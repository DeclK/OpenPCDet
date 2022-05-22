import torch
import torch.nn as nn
import numpy as np
from torch.nn.init import kaiming_normal_
from ...model_utils import centernet_utils
from ....utils import loss_utils, box_utils, common_utils
from ....ops.roiaware_pool3d.roiaware_pool3d_utils import points_in_boxes_cpu   # or we can try gpu?

class SeparateHead(nn.Module):
    def __init__(self, input_channels, sep_head_dict, init_bias=-2.19, use_bias=False):
        super().__init__()
        self.sep_head_dict = sep_head_dict

        for cur_name in self.sep_head_dict:
            output_channels = self.sep_head_dict[cur_name]['out_channels']
            num_conv = self.sep_head_dict[cur_name]['num_conv']

            fc_list = []
            for k in range(num_conv - 1):
                fc_list.append(nn.Sequential(
                    nn.Conv2d(input_channels, input_channels, kernel_size=3, stride=1, padding=1, bias=use_bias),
                    nn.BatchNorm2d(input_channels),
                    nn.ReLU()
                ))
            fc_list.append(nn.Conv2d(input_channels, output_channels, kernel_size=3, stride=1, padding=1, bias=True))
            fc = nn.Sequential(*fc_list)
            if 'hm' in cur_name:
                fc[-1].bias.data.fill_(init_bias)
            else:
                for m in fc.modules():
                    if isinstance(m, nn.Conv2d):
                        kaiming_normal_(m.weight.data)
                        if hasattr(m, "bias") and m.bias is not None:
                            nn.init.constant_(m.bias, 0)

            self.__setattr__(cur_name, fc)

    def forward(self, x):
        ret_dict = {}
        for cur_name in self.sep_head_dict:
            ret_dict[cur_name] = self.__getattr__(cur_name)(x)

        return ret_dict


class CGAM(nn.Module):
    def __init__(self, model_cfg, input_channels, class_names, grid_size=None, voxel_size=None, point_cloud_range=None) -> None:
        super().__init__()
        self.model_cfg = model_cfg
        self.point_cloud_range = point_cloud_range
        self.voxel_size = voxel_size
        self.class_names = class_names
        self.corner_types = model_cfg.CORNER_TYPES
        self.forward_ret_dict = {}

        self.shared_conv = nn.Sequential(
            nn.Conv2d(input_channels, self.model_cfg.SHARED_CONV_CHANNEL, 3, stride=1, padding=1),
            nn.BatchNorm2d(self.model_cfg.SHARED_CONV_CHANNEL),
            nn.ReLU()
        )

        # build seperate head
        head_dict = model_cfg.HEAD_DICT
        head_dict['hm'].update(dict(out_channels=len(model_cfg.CORNER_TYPES)))
        assert len(model_cfg.CORNER_TYPES) == 3, 'only 3 types of corners are implemented'
        self.seperate_head = SeparateHead(
            input_channels=model_cfg.SHARED_CONV_CHANNEL,
            sep_head_dict=head_dict,
            init_bias=-2.19,
            use_bias=True
        )
        self.build_losses()

    def build_losses(self):
        self.add_module('hm_loss_func', loss_utils.FocalLossCenterNet())
        self.add_module('reg_loss_func', loss_utils.RegLossCenterNet())

    def corner_selection(self, corners_xy, gt_boxes, points, pts_of_box):
        """
        for single sample, select corners of gt
        """
        INDEX = [(2, 3, 1), (3, 0, 2), (0, 1, 3), (1, 2, 0)]    # 按照逆时针魔改
        M = len(gt_boxes)
        c_invis = gt_boxes.new_zeros(M, 2)
        c_part_vis_l = gt_boxes.new_zeros(M, 2)
        c_part_vis_w = gt_boxes.new_zeros(M, 2)
        for box_idx in range(M):
            cur_box = gt_boxes[box_idx, :7]
            points_in_box = points[pts_of_box == box_idx, :3]  # (N, 3)
            # canonical transformation
            points_in_box = points_in_box - cur_box[:3]
            points_in_box = common_utils.rotate_points_along_z(
                points_in_box.unsqueeze(0), angle=-cur_box[6:7]).squeeze(0)
            x, y = points_in_box[:, 0], points_in_box[:, 1]
            q0 = torch.sum(torch.logical_and(x > 0, y > 0)).item()
            q1 = torch.sum(torch.logical_and(x > 0, y < 0)).item()
            q2 = torch.sum(torch.logical_and(x < 0, y < 0)).item()
            q3 = torch.sum(torch.logical_and(x < 0, y > 0)).item()
            q = [q0, q1, q2, q3]
            sub_q = [q3 + q0 + q1, 
                     q0 + q1 + q2,
                     q1 + q2 + q3,
                     q2 + q3 + q0]
            valid_q = sum([1 if qi > 0 else 0 for qi in q])
            if valid_q > 2:
                max_i = np.argmax(sub_q)
            else: max_i = np.argmax(q)
            c_invis[box_idx] = corners_xy[box_idx, INDEX[max_i][0]]
            c_part_vis_l[box_idx] = corners_xy[box_idx, INDEX[max_i][1]]
            c_part_vis_w[box_idx] = corners_xy[box_idx, INDEX[max_i][2]] 
        return c_invis, c_part_vis_l, c_part_vis_w

    def assign_corners_of_single_sample(self, corner_types, gt_boxes, feature_map_size, 
        feature_map_stride, points, num_max_objs=500, gaussian_overlap=0.1, min_radius=2):
        """
        1. build placeholder heatmap
        2. assign heatmap for boxes in this sample
        """
        C = len(corner_types)
        heatmap = gt_boxes.new_zeros(C, feature_map_size[1], feature_map_size[0])
        corner_offset = gt_boxes.new_zeros(C, num_max_objs, 2)
        inds = gt_boxes.new_zeros(C, num_max_objs).long()
        mask = gt_boxes.new_zeros(C, num_max_objs).long()
        # get corner points and foreground points
        corners = box_utils.boxes_to_corners_3d(gt_boxes[:, :7])  # (M, 8, 3)
        corners_xy = corners[:, :4, :2]    # (M, 4, 2)
        pts_in_flag = points_in_boxes_cpu(points[:, :3], gt_boxes[:, :7])
        in_label, pts_of_box = pts_in_flag.max(0)
        fg_pts, pts_of_box = points[in_label.bool()], pts_of_box[in_label.bool()]

        corners_i, corners_l, corners_w = self.corner_selection(corners_xy, gt_boxes, fg_pts, pts_of_box)
        sub_corner = torch.stack([corners_i, corners_l, corners_w], dim=0) # (3, M, 2)

        x, y = sub_corner[:, :, 0], sub_corner[:, :, 1]
        coord_x = (x - self.point_cloud_range[0]) / self.voxel_size[0] / feature_map_stride
        coord_y = (y - self.point_cloud_range[1]) / self.voxel_size[1] / feature_map_stride
        coord_x = torch.clamp(coord_x, min=0, max=feature_map_size[0] - 0.5)  # bugfixed: 1e-6 does not work for center.int()
        coord_y = torch.clamp(coord_y, min=0, max=feature_map_size[1] - 0.5)  #
        center = torch.stack((coord_x, coord_y), dim=-1)    # (3, M, 2)
        center_int = center.int()
        center_int_float = center_int.float()   # (3, M, 2)

        dx, dy = gt_boxes[:, 3], gt_boxes[:, 4]
        dx = dx / self.voxel_size[0] / feature_map_stride
        dy = dy / self.voxel_size[1] / feature_map_stride

        radius = centernet_utils.gaussian_radius(dx, dy, min_overlap=gaussian_overlap)
        radius = torch.clamp_min(radius.int(), min=min_radius)

        for k in range(min(num_max_objs, gt_boxes.shape[0])):
            if dx[k] <= 0 or dy[k] <= 0:
                continue
            for i in range(C):
                if not (0 <= center_int[i][k][0] <= feature_map_size[0] and 0 <= center_int[i][k][1] <= feature_map_size[1]):
                    continue    # insurance
                centernet_utils.draw_gaussian_to_heatmap(heatmap[i], center[i][k], radius[k].item())
                corner_offset[i][k] = center[i][k] - center_int_float[i][k].float()    # use .float redundent!
                inds[i][k] = center_int[i][k][1] * feature_map_size[0] + center_int[i][k][0]
                mask[i][k] = 1

        return heatmap, corner_offset, inds, mask

    def assign_targets(self, gt_boxes, feature_map_size, points):
        """
        return: 
            heatmap: (B, 3, H, W)
            center_offset: (B, 3, K, 2), K is num_max_objs
            inds: (B, 3, K)
            masks: (B, 3, K)
        """
        feature_map_size = feature_map_size[::-1]  # [H, W] ==> [x, y]
        target_assigner_cfg = self.model_cfg.TARGET_ASSIGNER_CONFIG
        ret_dict = {}

        all_names = np.array(['none', *self.class_names])
        batch_size = gt_boxes.shape[0]
        heatmap_list, corner_offset_list, inds_list, masks_list = [], [], [], []
        for bs_idx in range(batch_size):
            cur_gt_boxes = gt_boxes[bs_idx]
            p_idx = torch.nonzero(points[:, 0] == bs_idx).view(-1)
            cur_points = points[p_idx, 1:]
            gt_class_names = all_names[cur_gt_boxes[:, -1].cpu().long().numpy()]
        
            gt_boxes_single_head = []
            for idx, name in enumerate(gt_class_names): # filter empty gt_boxes
                if name not in self.class_names:
                    continue
                temp_box = cur_gt_boxes[idx]
                gt_boxes_single_head.append(temp_box[None, :])

            if len(gt_boxes_single_head) == 0:
                gt_boxes_single_head = cur_gt_boxes[:0, :]  # return an empty tensor
            else:
                gt_boxes_single_head = torch.cat(gt_boxes_single_head, dim=0)

            heatmap, corner_offset, inds, mask = self.assign_corners_of_single_sample(
                corner_types=self.corner_types, gt_boxes=gt_boxes_single_head.cpu(),
                feature_map_size=feature_map_size, feature_map_stride=target_assigner_cfg.FEATURE_MAP_STRIDE,
                gaussian_overlap=target_assigner_cfg.GAUSSIAN_OVERLAP,
                min_radius=target_assigner_cfg.MIN_RADIUS,
                points=cur_points.cpu()
            )
            heatmap_list.append(heatmap.to(gt_boxes_single_head.device))
            corner_offset_list.append(corner_offset.to(gt_boxes_single_head.device))
            inds_list.append(inds.to(gt_boxes_single_head.device))
            masks_list.append(mask.to(gt_boxes_single_head.device))

        ret_dict['heatmap'] = torch.stack(heatmap_list, dim=0)   # stack batch
        ret_dict['corner_offset'] = torch.stack(corner_offset_list, dim=0)
        ret_dict['inds'] = torch.stack(inds_list, dim=0)
        ret_dict['masks'] = torch.stack(masks_list, dim=0)
        return ret_dict

    def sigmoid(self, x):
        y = torch.clamp(x.sigmoid(), min=1e-4, max=1 - 1e-4)
        return y

    def get_loss(self):
        pred_dict = self.forward_ret_dict['pred_dict']
        target_dict = self.forward_ret_dict['target_dict']
        tb_dict = {}

        if not self.model_cfg.HM_NORMALIZATION:
            pred_dict['hm'] = self.sigmoid(pred_dict['hm'])
        hm_loss = self.hm_loss_func(pred_dict['hm'], target_dict['heatmap'])
        hm_loss *= self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['cls_weight']

        offset_target = target_dict['corner_offset']
        C = offset_target.size(1)
        reg_loss = 0
        for i in range(C):  # for each type corner
            mask_i = target_dict['masks'][:, i]
            ind_i = target_dict['inds'][:, i]
            offset_target_i = offset_target[:, i]
            reg_loss += self.reg_loss_func(pred_dict['corner'], mask_i, ind_i, offset_target_i)
        loc_loss = (reg_loss * reg_loss.new_tensor(self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['code_weights'])).sum()
        loc_loss = loc_loss * self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['loc_weight']

        loss = hm_loss + loc_loss
        tb_dict['corner_hm_loss'] = hm_loss.item()
        tb_dict['corner_offset_loss'] = loc_loss.item()
        return loss, tb_dict

    def forward(self, data_dict):
        spatial_features_2d = data_dict['spatial_features_2d']
        x = self.shared_conv(spatial_features_2d)

        pred_dict = self.seperate_head(x)
        
        if self.model_cfg.HM_NORMALIZATION:
            pred_dict['hm'] = self.sigmoid(pred_dict['hm'])
        self.forward_ret_dict['pred_dict'] = pred_dict

        if self.training:
            target_dict = self.assign_targets(
                data_dict['gt_boxes'], feature_map_size=spatial_features_2d.size()[2:],
                points=data_dict['points']
            )
            self.forward_ret_dict['target_dict'] = target_dict

        return pred_dict