import copy
import numpy as np
import torch
import torch.nn as nn
from torch.nn.init import kaiming_normal_
from pcdet.models.model_utils import centernet_utils


class BaseBEVBackbone(nn.Module):
    def __init__(self, model_cfg, input_channels):
        super().__init__()
        self.model_cfg = model_cfg

        if self.model_cfg.get('LAYER_NUMS', None) is not None:
            assert len(self.model_cfg.LAYER_NUMS) == len(self.model_cfg.LAYER_STRIDES) == len(self.model_cfg.NUM_FILTERS)
            layer_nums = self.model_cfg.LAYER_NUMS
            layer_strides = self.model_cfg.LAYER_STRIDES
            num_filters = self.model_cfg.NUM_FILTERS
        else:
            layer_nums = layer_strides = num_filters = []

        if self.model_cfg.get('UPSAMPLE_STRIDES', None) is not None:
            assert len(self.model_cfg.UPSAMPLE_STRIDES) == len(self.model_cfg.NUM_UPSAMPLE_FILTERS)
            num_upsample_filters = self.model_cfg.NUM_UPSAMPLE_FILTERS
            upsample_strides = self.model_cfg.UPSAMPLE_STRIDES
        else:
            upsample_strides = num_upsample_filters = []

        num_levels = len(layer_nums)
        c_in_list = [input_channels, *num_filters[:-1]]
        self.blocks = nn.ModuleList()
        self.deblocks = nn.ModuleList()
        for idx in range(num_levels):
            cur_layers = [
                nn.ZeroPad2d(1),
                nn.Conv2d(
                    c_in_list[idx], num_filters[idx], kernel_size=3,
                    stride=layer_strides[idx], padding=0, bias=False
                ),
                nn.BatchNorm2d(num_filters[idx], eps=1e-3, momentum=0.01),
                nn.ReLU()
            ]
            for k in range(layer_nums[idx]):
                cur_layers.extend([
                    nn.Conv2d(num_filters[idx], num_filters[idx], kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(num_filters[idx], eps=1e-3, momentum=0.01),
                    nn.ReLU()
                ])
            self.blocks.append(nn.Sequential(*cur_layers))
            if len(upsample_strides) > 0:
                stride = upsample_strides[idx]
                if stride >= 1:
                    self.deblocks.append(nn.Sequential(
                        nn.ConvTranspose2d(
                            num_filters[idx], num_upsample_filters[idx],
                            upsample_strides[idx],
                            stride=upsample_strides[idx], bias=False
                        ),
                        nn.BatchNorm2d(num_upsample_filters[idx], eps=1e-3, momentum=0.01),
                        nn.ReLU()
                    ))
                else:
                    stride = np.round(1 / stride).astype(np.int)
                    self.deblocks.append(nn.Sequential(
                        nn.Conv2d(
                            num_filters[idx], num_upsample_filters[idx],
                            stride,
                            stride=stride, bias=False
                        ),
                        nn.BatchNorm2d(num_upsample_filters[idx], eps=1e-3, momentum=0.01),
                        nn.ReLU()
                    ))

        c_in = sum(num_upsample_filters)
        if len(upsample_strides) > num_levels:
            self.deblocks.append(nn.Sequential(
                nn.ConvTranspose2d(c_in, c_in, upsample_strides[-1], stride=upsample_strides[-1], bias=False),
                nn.BatchNorm2d(c_in, eps=1e-3, momentum=0.01),
                nn.ReLU(),
            ))

        self.num_bev_features = c_in

    def forward(self, x, **kwags):
        ups = []
        for i in range(len(self.blocks)):
            x = self.blocks[i](x)
            ups.append(self.deblocks[i](x))
        x = torch.cat(ups, dim=1)
    
        return x


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


class CenterHeadIoU(nn.Module):
    def __init__(self, model_cfg, input_channels, num_class, class_names, grid_size, point_cloud_range, voxel_size,
                 predict_boxes_when_training=True):
        super().__init__()
        self.model_cfg = model_cfg
        self.num_class = num_class
        self.grid_size = grid_size
        self.point_cloud_range = point_cloud_range
        self.voxel_size = voxel_size
        self.feature_map_stride = self.model_cfg.TARGET_ASSIGNER_CONFIG.get('FEATURE_MAP_STRIDE', None)

        self.class_names = class_names
        self.class_names_each_head = []
        self.class_id_mapping_each_head = []

        for cur_class_names in self.model_cfg.CLASS_NAMES_EACH_HEAD:
            self.class_names_each_head.append([x for x in cur_class_names if x in class_names])
            cur_class_id_mapping = torch.from_numpy(np.array(
                [self.class_names.index(x) for x in cur_class_names if x in class_names]
            )).cuda()
            self.class_id_mapping_each_head.append(cur_class_id_mapping)

        total_classes = sum([len(x) for x in self.class_names_each_head])
        assert total_classes == len(self.class_names), f'class_names_each_head={self.class_names_each_head}'

        self.shared_conv = nn.Sequential(
            nn.Conv2d(
                input_channels, self.model_cfg.SHARED_CONV_CHANNEL, 3, stride=1, padding=1,
                bias=self.model_cfg.get('USE_BIAS_BEFORE_NORM', False)
            ),
            nn.BatchNorm2d(self.model_cfg.SHARED_CONV_CHANNEL),
            nn.ReLU(),
        )

        self.heads_list = nn.ModuleList()
        self.separate_head_cfg = self.model_cfg.SEPARATE_HEAD_CFG
        for idx, cur_class_names in enumerate(self.class_names_each_head):
            cur_head_dict = copy.deepcopy(self.separate_head_cfg.HEAD_DICT)
            cur_head_dict['hm'] = dict(out_channels=len(cur_class_names), num_conv=self.model_cfg.NUM_HM_CONV)
            self.heads_list.append(
                SeparateHead(
                    input_channels=self.model_cfg.SHARED_CONV_CHANNEL,
                    sep_head_dict=cur_head_dict,
                    init_bias=-2.19,
                    use_bias=self.model_cfg.get('USE_BIAS_BEFORE_NORM', False)
                )
            )
        self.predict_boxes_when_training = predict_boxes_when_training
        self.forward_ret_dict = {}

    def sigmoid(self, x):
        y = torch.clamp(x.sigmoid(), min=1e-4, max=1 - 1e-4)
        return y

    def get_loss(self):
        pred_dicts = self.forward_ret_dict['pred_dicts']
        target_dicts = self.forward_ret_dict['target_dicts']

        tb_dict = {}
        loss = 0

        for idx, pred_dict in enumerate(pred_dicts):
            pred_dict['hm'] = self.sigmoid(pred_dict['hm'])
            hm_loss = self.hm_loss_func(pred_dict['hm'], target_dicts['heatmaps'][idx])
            hm_loss *= self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['cls_weight']

            target_boxes = target_dicts['target_boxes'][idx]
            gt_boxes = target_dicts['gt_boxes'][idx]
            mask = target_dicts['masks'][idx]
            ind = target_dicts['inds'][idx]

            # ususal box regression
            pred_boxes = torch.cat([pred_dict[head_name] for head_name in self.separate_head_cfg.HEAD_ORDER], dim=1)
            reg_loss = self.reg_loss_func(pred_boxes, mask, ind, target_boxes)
            loc_loss = (reg_loss * reg_loss.new_tensor(self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['code_weights'])).sum()
            loc_loss = loc_loss * self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['loc_weight']

            # IoU based regression loss
            pred_boxes = centernet_utils.generate_dense_boxes(
                pred_dict=pred_dict,
                feature_map_stride=self.feature_map_stride,
                voxel_size=self.voxel_size,
                point_cloud_range=self.point_cloud_range)
            pred_boxes = torch.clamp(pred_boxes, min=-200., max=200.)   # avoid large number          
            # iou_reg_loss = self.iou_reg_loss_func(pred_boxes, mask, ind, gt_boxes)
            # iou_reg_loss *= self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['iou_reg_weight']
            iou_reg_loss = 0

            # IoU prediction loss
            if pred_dict.get('iou', None) is not None:
                pred_boxes_for_iou = pred_boxes.detach()
                iou_loss = self.iou_loss_func(pred_dict['iou'], mask, ind, pred_boxes_for_iou, gt_boxes)
                iou_loss *= self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['iou_weight']
                tb_dict['iou_loss_%d' % idx] = iou_loss.item()
            else: iou_loss = 0

            loss += hm_loss + loc_loss + iou_reg_loss + iou_loss
            # tb_dict['iou_reg_loss_head_%d' % idx] = iou_reg_loss.item()
            tb_dict['hm_loss_head_%d' % idx] = hm_loss.item()
            tb_dict['loc_loss_head_%d' % idx] = loc_loss.item()

        tb_dict['rpn_loss'] = loss.item()
        return loss, tb_dict

    def generate_predicted_boxes(self, batch_size, pred_dicts):
        post_process_cfg = self.model_cfg.POST_PROCESSING
        ret_dict = [{
            'pred_boxes': [],
            'pred_scores': [],
            'pred_labels': [],
        } for i in range(batch_size)]

        for idx, pred_dict in enumerate(pred_dicts):           # for each head
            batch_box_preds = self.generate_dense_boxes(
                pred_dict=pred_dict,
                feature_map_stride=self.feature_map_stride,
                voxel_size=self.voxel_size,
                point_cloud_range=self.point_cloud_range
            )
            B, _, H, W = batch_box_preds.size()
            batch_box_preds = batch_box_preds.permute(0, 2, 3, 1).view(B, H*W, -1)
            batch_hm = pred_dict['hm'].sigmoid().permute(0, 2, 3, 1).view(B, H*W, -1)

            if 'iou' in pred_dict.keys():
                batch_iou = pred_dict['iou'].permute(0, 2, 3, 1).view(B, H*W)
                batch_iou = (batch_iou + 1) * 0.5
            else: batch_iou = torch.ones((B, H*W)).to(batch_hm.device)

            for i in range(B):                                  # for each batch
                box_preds = batch_box_preds[i]
                hm_preds = batch_hm[i]
                iou_preds = batch_iou[i]
                scores, labels = torch.max(hm_preds, dim=-1)    # (H*W,)

                rectifier = post_process_cfg.get('RECTIFIER', 0.0)
                rectifier = torch.tensor(rectifier).view(-1).to(scores.device)
                if rectifier.size(0) > 1:                       # class specific rectifier
                    assert rectifier.size(0) == self.num_class
                    rectifier = rectifier[labels]               # (H*W,)

                scores = torch.pow(scores, 1 - rectifier) \
                       * torch.pow(iou_preds, rectifier)

                labels = self.class_id_mapping_each_head[idx][labels.long()]

                ret_dict[i]['pred_boxes'].append(box_preds)
                ret_dict[i]['pred_scores'].append(scores)
                ret_dict[i]['pred_labels'].append(labels)

        for i in range(batch_size): # concat head results
            ret_dict[i]['pred_boxes'] = torch.cat(ret_dict[i]['pred_boxes'], dim=0)
            ret_dict[i]['pred_scores'] = torch.cat(ret_dict[i]['pred_scores'], dim=0)
            ret_dict[i]['pred_labels'] = torch.cat(ret_dict[i]['pred_labels'], dim=0) + 1

        return ret_dict

    def generate_dense_boxes(self, pred_dict, feature_map_stride, voxel_size, point_cloud_range):
        """
        Generate boxes for single sample pixel-wise
        Input pred_dict:
            center      (B, 2, H, W)
            center_z    (B, 1, H, W)
            dim         (B, 3, H, W)
            cos/sin     (B, 1, H, W)
        Return:
            (B, 7, H, W)
        """
        batch_offset = pred_dict['center']
        batch_z = pred_dict['center_z']
        batch_dim = torch.clamp(pred_dict['dim'], min=-3, max=3).exp()  # avoid large gradient
        batch_rot_cos = pred_dict['rot'][:, 0].unsqueeze(dim=1)
        batch_rot_sin = pred_dict['rot'][:, 1].unsqueeze(dim=1)
        # batch_rot = torch.atan2(batch_rot_sin, batch_rot_cos)
        batch_rot = torch.atan(batch_rot_sin / (batch_rot_cos + 1e-6))  # ONNX doesn't support atan2

        B, _, H, W = batch_dim.size()
        ys, xs = torch.meshgrid([torch.arange(0, H), torch.arange(0, W)])
        ys = ys.view(1, 1, H, W).repeat(B, 1, 1, 1).to(batch_dim)
        xs = xs.view(1, 1, H, W).repeat(B, 1, 1, 1).to(batch_dim)
        xs = xs + batch_offset[:, 0:1]    # (B, 1, H, W)
        ys = ys + batch_offset[:, 1:2]
        xs = xs * feature_map_stride * voxel_size[0] + point_cloud_range[0]
        ys = ys * feature_map_stride * voxel_size[1] + point_cloud_range[1]

        pred_boxes = torch.cat((xs, ys, batch_z, batch_dim, batch_rot), dim=1)
        return pred_boxes

    def forward(self, x):
        x = self.shared_conv(x)

        pred_dicts = []
        for head in self.heads_list:
            pred_dicts.append(head(x))

        self.forward_ret_dict['pred_dicts'] = pred_dicts

        batch_size = x.size(0)
        pred_dicts = self.generate_predicted_boxes(
            batch_size, pred_dicts
        )
        scores = [pred_dicts[0]['pred_scores']]               # for aw format need
        boxes = pred_dicts[0]['pred_boxes'].unsqueeze(0)      # (1, H*W, 7)

        return scores, boxes
