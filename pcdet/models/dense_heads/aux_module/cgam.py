import copy
import torch
import torch.nn as nn
from torch.nn.init import kaiming_normal_
from ...model_utils import centernet_utils
from ....utils import loss_utils, box_utils

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
        self.corner_types = len(model_cfg.CORNER_TYPES)
        self.forward_ret_dict = {}

        self.shared_conv = nn.Sequential(
            nn.Conv2d(input_channels, self.model_cfg.SHARED_CONV_CHANNEL, 3, stride=1, padding=1),
            nn.BatchNorm2d(self.model_cfg.SHARED_CONV_CHANNEL),
            nn.ReLU()
        )

        self.head_list = nn.ModuleList()
        self.head_dict = model_cfg.HEAD_DICT
        for _ in range(self.corner_types):  #
            cur_head_dict = copy.deepcopy(self.head_dict)
            cur_head_dict['hm']['out_channels'] = len(class_names)
            seperate_head = SeparateHead(
                input_channels=model_cfg.SHARED_CONV_CHANNEL,
                sep_head_dict=cur_head_dict,
                init_bias=-2.19,
                use_bias=True
            )
            self.head_list.append(seperate_head)
        self.build_losses()

    def build_losses(self):
        self.add_module('hm_loss_func', loss_utils.FocalLossCenterNet())
        self.add_module('reg_loss_func', loss_utils.RegLossCenterNet())

    def corner_selection(self, corners, gt_sparse_corner):
        """
        For a single sample, select corners of gt boxes, not used
        input:
            corners: (M, 4, 3)
            gt_sparse_corner: (M, 3)
        return: 
            ret_corners: (3, M, 3), the first 3 means 3 types of corners
        """
        INDEX = torch.tensor([(0, 1, 3), (1, 0, 2), (2, 3, 1), (3, 2, 0)])
        INDEX = INDEX.to(corners.device)
        M = len(gt_sparse_corner)
        # finde sparse corner index
        dist = corners - gt_sparse_corner.unsqueeze(1)
        dist = torch.norm(dist, dim=2)  # (M, 4)
        sparse_index = torch.argmin(dist, dim=1)    # (M,)
        all_index = INDEX[sparse_index].unsqueeze(2).expand((M, 3, 3)) # (M, 3, 3)
        ret_corners = torch.gather(corners, index=all_index, dim=1)    # (M, 3, 3)
        return ret_corners.permute(1, 0, 2).contiguous()

    def assign_corners_of_single_sample(self, gt_boxes, sub_corner, feature_map_size,
        feature_map_stride, num_max_objs=500, gaussian_overlap=0.1, min_radius=2):
        """
        Input:
            gt_boxes: (N, 7+C)
            sub_corner: (N, 3)
        Return:
            heatmap & corner offset for a single sample of a single head
        """
        N = len(self.class_names)
        heatmap = gt_boxes.new_zeros(N, feature_map_size[1], feature_map_size[0])
        corner_offset = gt_boxes.new_zeros(num_max_objs, 2)
        inds = gt_boxes.new_zeros(num_max_objs).long()
        mask = gt_boxes.new_zeros(num_max_objs).long()

        x, y = sub_corner[:, 0], sub_corner[:, 1]
        coord_x = (x - self.point_cloud_range[0]) / self.voxel_size[0] / feature_map_stride
        coord_y = (y - self.point_cloud_range[1]) / self.voxel_size[1] / feature_map_stride
        coord_x = torch.clamp(coord_x, min=0, max=feature_map_size[0] - 0.5)  # bugfixed: 1e-6 does not work for center.int()
        coord_y = torch.clamp(coord_y, min=0, max=feature_map_size[1] - 0.5)  #
        center = torch.stack((coord_x, coord_y), dim=-1)                # (M, 2)
        center_int = center.int()
        center_int_float = center_int.float()

        dx, dy = gt_boxes[:, 3], gt_boxes[:, 4]
        dx = dx / self.voxel_size[0] / feature_map_stride
        dy = dy / self.voxel_size[1] / feature_map_stride

        radius = centernet_utils.gaussian_radius(dx, dy, min_overlap=gaussian_overlap)
        radius = torch.clamp_min(radius.int(), min=min_radius)

        for k in range(min(num_max_objs, gt_boxes.shape[0])):
            if dx[k] <= 0 or dy[k] <= 0:
                continue
            if not (0 <= center_int[k, 0] <= feature_map_size[0] and 0 <= center_int[k, 1] <= feature_map_size[1]):
                continue    # insurance
            cur_class_id = (gt_boxes[k, -1] - 1).long()
            centernet_utils.draw_gaussian_to_heatmap(heatmap[cur_class_id], center[k], radius[k].item())
            corner_offset[k] = center[k] - center_int_float[k]
            inds[k] = center_int[k, 1] * feature_map_size[0] + center_int[k, 0]
            mask[k] = 1

        return heatmap, corner_offset, inds, mask

    def assign_targets(self, gt_boxes, feature_map_size):
        """
        return: a dict of list, each member of list contains results for single corner head
            heatmap: (B, N, H, W), N is num_classes
            center_offset: (B, K, 2), K is num_max_objs
            inds: (B, K)
            masks: (B, K)
            corners: debug use
        """
        feature_map_size = feature_map_size[::-1]  # [H, W] ==> [x, y]
        target_assigner_cfg = self.model_cfg.TARGET_ASSIGNER_CONFIG
        batch_size = gt_boxes.shape[0]
        ret_dict = {
            'heatmaps': [],
            'corner_offsets': [],
            'inds': [],
            'masks': [],
        }

        for idx in range(self.corner_types):
            heatmap_list, corner_offset_list, inds_list, masks_list = [], [], [], []
            for bs_idx in range(batch_size):
                cur_gt_boxes = gt_boxes[bs_idx]             # (M, 8)
                # filter empty gt
                non_empty_mask = cur_gt_boxes[:, -1].int() != 0
                gt_boxes_single_head = cur_gt_boxes[non_empty_mask]

                # build sub corner for single head
                corners = box_utils.boxes_to_corners_3d(gt_boxes_single_head)       # (M, 8, 3)
                sub_corner = corners[:, idx]    # (M, 3)

                heatmap, corner_offset, inds, mask = self.assign_corners_of_single_sample(
                    gt_boxes=gt_boxes_single_head.cpu(),
                    sub_corner=sub_corner.cpu(),
                    feature_map_size=feature_map_size, 
                    feature_map_stride=target_assigner_cfg.FEATURE_MAP_STRIDE,
                    gaussian_overlap=target_assigner_cfg.GAUSSIAN_OVERLAP,
                    min_radius=target_assigner_cfg.MIN_RADIUS,
                )
                heatmap_list.append(heatmap.to(gt_boxes_single_head.device))
                corner_offset_list.append(corner_offset.to(gt_boxes_single_head.device))
                inds_list.append(inds.to(gt_boxes_single_head.device))
                masks_list.append(mask.to(gt_boxes_single_head.device))

            ret_dict['heatmaps'].append(torch.stack(heatmap_list, dim=0) ) # stack batch
            ret_dict['corner_offsets'].append(torch.stack(corner_offset_list, dim=0))
            ret_dict['inds'].append(torch.stack(inds_list, dim=0))
            ret_dict['masks'].append(torch.stack(masks_list, dim=0))
        return ret_dict

    def sigmoid(self, x):
        y = torch.clamp(x.sigmoid(), min=1e-4, max=1 - 1e-4)
        return y

    def get_loss(self):
        pred_dicts = self.forward_ret_dict['pred_dicts']
        target_dicts = self.forward_ret_dict['target_dicts']
        tb_dict = {}
        loss = 0

        for idx, pred_dict in enumerate(pred_dicts):
            if not self.model_cfg.HM_NORMALIZATION:
                pred_dict['hm'] = self.sigmoid(pred_dict['hm'])

            pred_hm = pred_dict['hm']
            target_hm = target_dicts['heatmaps'][idx]
            hm_loss = self.hm_loss_func(pred_hm, target_hm)
            hm_loss *= self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['cls_weight']

            pred_offset = pred_dict['corner']
            target_offset = target_dicts['corner_offsets'][idx]
            mask = target_dicts['masks'][idx]
            ind = target_dicts['inds'][idx]
            reg_loss = self.reg_loss_func(pred_offset, mask, ind, target_offset)
            code_weights = self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['code_weights']
            loc_loss = (reg_loss * reg_loss.new_tensor(code_weights)).sum()
            loc_loss *= self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['loc_weight']

            loss += hm_loss + loc_loss
            tb_dict[f'corner_hm_loss_{idx}'] = hm_loss.item()
            tb_dict[f'corner_offset_loss_{idx}'] = loc_loss.item()
        return loss, tb_dict

    def forward(self, data_dict):
        spatial_features_2d = data_dict['spatial_features_2d']
        x = self.shared_conv(spatial_features_2d)

        pred_dicts = []
        for head in self.head_list:
            pred_dicts.append(head(x))
        
        if self.model_cfg.HM_NORMALIZATION:
            for pred_dict in pred_dicts:
                pred_dict['hm'] = self.sigmoid(pred_dict['hm'])
        self.forward_ret_dict['pred_dicts'] = pred_dicts

        if self.training:
            target_dicts = self.assign_targets(
                gt_boxes=data_dict['gt_boxes'], 
                feature_map_size=spatial_features_2d.size()[2:])
            self.forward_ret_dict['target_dicts'] = target_dicts

        merge_feat_list = []    # concat feat to merge with bev
        for pred_dict in pred_dicts:
            for _, feat in pred_dict.items():
                merge_feat_list.append(feat)
        merge_feat = torch.cat(merge_feat_list, dim=1)
        return merge_feat