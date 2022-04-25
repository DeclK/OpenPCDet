from functools import partial

import torch.nn as nn
import torch

from ...utils.spconv_utils import replace_feature, spconv
from pcdet.ops.pointnet2.pointnet2_stack import pointnet2_utils
from ...ops.roiaware_pool3d.roiaware_pool3d_utils import points_in_boxes_cpu
from ...utils import loss_utils

def post_act_block(in_channels, out_channels, kernel_size, indice_key=None, stride=1, padding=0,
                   conv_type='subm', norm_fn=None):

    if conv_type == 'subm':
        conv = spconv.SubMConv3d(in_channels, out_channels, kernel_size, bias=False, indice_key=indice_key)
    elif conv_type == 'spconv':
        conv = spconv.SparseConv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding,
                                   bias=False, indice_key=indice_key)
    elif conv_type == 'inverseconv':
        conv = spconv.SparseInverseConv3d(in_channels, out_channels, kernel_size, indice_key=indice_key, bias=False)
    else:
        raise NotImplementedError

    m = spconv.SparseSequential(
        conv,
        norm_fn(out_channels),
        nn.ReLU(),
    )

    return m

class AuxVoxelBackBone8x(nn.Module):
    def __init__(self, model_cfg, input_channels, grid_size, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        # Please remember adjust point_range & voxel_size according to used dataset
        self.point_range = [-75.2, -75.2, -5.0, 75.2, 75.2, 3.0]
        self.voxel_size = [0.1, 0.1, 0.2]
        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)
        self.sparse_shape = grid_size[::-1] + [1, 0, 0]

        self.conv_input = spconv.SparseSequential(
            spconv.SubMConv3d(input_channels, 16, 3, padding=1, bias=False, indice_key='subm1'),
            norm_fn(16),
            nn.ReLU(),
        )
        block = post_act_block

        self.conv1 = spconv.SparseSequential(
            block(16, 16, 3, norm_fn=norm_fn, padding=1, indice_key='subm1'),
        )

        self.conv2 = spconv.SparseSequential(
            # [1600, 1408, 41] <- [800, 704, 21]
            block(16, 32, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv2', conv_type='spconv'),
            block(32, 32, 3, norm_fn=norm_fn, padding=1, indice_key='subm2'),
            block(32, 32, 3, norm_fn=norm_fn, padding=1, indice_key='subm2'),
        )

        self.conv3 = spconv.SparseSequential(
            # [800, 704, 21] <- [400, 352, 11]
            block(32, 64, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv3', conv_type='spconv'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm3'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm3'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm3'),
        )

        self.conv4 = spconv.SparseSequential(
            # [400, 352, 11] <- [200, 176, 5]
            block(64, 64, 3, norm_fn=norm_fn, stride=2, padding=(0, 1, 1), indice_key='spconv4', conv_type='spconv'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm4'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm4'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm4'),
        )

        last_pad = 0
        last_pad = self.model_cfg.get('last_pad', last_pad)
        self.conv_out = spconv.SparseSequential(
            # [200, 150, 5] -> [200, 150, 2]
            spconv.SparseConv3d(64, 64, (3, 1, 1), stride=(2, 1, 1), padding=last_pad,
                                bias=False, indice_key='spconv_down2'),
            norm_fn(64),
            nn.ReLU(),
        )
        self.num_point_features = 64
        self.backbone_channels = {
            'x_conv1': 16,
            'x_conv2': 32,
            'x_conv3': 64,
            'x_conv4': 64
        }

        # Auxiliary network
        input_channel = self.backbone_channels['x_conv2'] + self.backbone_channels['x_conv3'] + self.backbone_channels['x_conv4']
        self.point_fc = nn.Linear(input_channel, 64, bias=False)
        self.point_cls = nn.Linear(64, 1, bias=False)
        self.point_reg = nn.Linear(64, 3, bias=False)
        self.forward_ret_dict = {}

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size: int
                vfe_features: (num_voxels, C)
                voxel_coords: (num_voxels, 4), [batch_idx, z_idx, y_idx, x_idx]
        Returns:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
        """
        voxel_features, voxel_coords = batch_dict['voxel_features'], batch_dict['voxel_coords']
        batch_size = batch_dict['batch_size']
        input_sp_tensor = spconv.SparseConvTensor(
            features=voxel_features,
            indices=voxel_coords.int(),
            spatial_shape=self.sparse_shape,
            batch_size=batch_size
        )

        x = self.conv_input(input_sp_tensor)

        x_conv1 = self.conv1(x)
        x_conv2 = self.conv2(x_conv1)
        x_conv3 = self.conv3(x_conv2)
        x_conv4 = self.conv4(x_conv3)

        # for detection head
        # [200, 176, 5] -> [200, 176, 2]
        out = self.conv_out(x_conv4)

        batch_dict.update({
            'encoded_spconv_tensor': out,
            'encoded_spconv_tensor_stride': 8
        })
        batch_dict.update({
            'multi_scale_3d_features': {
                'x_conv1': x_conv1,
                'x_conv2': x_conv2,
                'x_conv3': x_conv3,
                'x_conv4': x_conv4,
            }
        })

        if not self.training:
            return batch_dict
        else:
            points_mean = torch.zeros_like(voxel_features)
            points_mean[:, 0] = voxel_coords[:, 0]
            points_mean[:, 1:] = voxel_features[:, :3]

            middle = batch_dict['multi_scale_3d_features']
            middle = [middle['x_conv2'], middle['x_conv3'], middle['x_conv4']]

            # scene voxel count
            unknown_batch_cnt = points_mean.new_zeros(batch_size).int()
            for bs_idx in range(batch_size):
                unknown_batch_cnt[bs_idx] = (voxel_coords[:, 0] == bs_idx).sum()
            # vx_feat: (num_non-empty_voxels, channels) vx_nxyz: (num_non-empty_voxels, 4) lidar 坐标系
            offset = torch.Tensor(self.point_range[:3])
            voxel_size = torch.Tensor(self.voxel_size)
            vx_feat, vx_nxyz, known_batch_cnt = tensor2points(middle[0], offset, voxel_size * 2)
            p0 = nearest_neighbor_interpolate(points_mean, unknown_batch_cnt, vx_nxyz, known_batch_cnt, vx_feat)

            vx_feat, vx_nxyz, known_batch_cnt = tensor2points(middle[1], offset, voxel_size * 4)
            p1 = nearest_neighbor_interpolate(points_mean, unknown_batch_cnt, vx_nxyz, known_batch_cnt, vx_feat)

            vx_feat, vx_nxyz, known_batch_cnt = tensor2points(middle[2], offset, voxel_size * 8)
            p2 = nearest_neighbor_interpolate(points_mean, unknown_batch_cnt, vx_nxyz, known_batch_cnt, vx_feat)

            pointwise = self.point_fc(torch.cat([p0, p1, p2], dim=-1))
            point_cls = self.point_cls(pointwise)
            self.forward_ret_dict['point_cls'] = point_cls
            point_reg = self.point_reg(pointwise)
            self.forward_ret_dict['point_reg'] = point_reg

            # save them for aux loss computation
            self.forward_ret_dict['points_mean'] = points_mean
            self.forward_ret_dict['gt_boxes'] = batch_dict['gt_boxes']

            return batch_dict
    
    def build_aux_target(self, nxyz, gt_boxes3d, enlarge=1.0):
        """ build target for each point in voxel 
        Args:
            - nxyz: points (N1 + N2 ..., 4)
            - gt_boxes: (B, N, 8) extra 1 for class feature
        Return:
            - 
        """
        center_offsets = list()
        pts_labels = list()

        for i in range(len(gt_boxes3d)):
            boxes3d = gt_boxes3d[i,:,:7].cpu()
            idx = torch.nonzero(nxyz[:, 0] == i).view(-1)
            new_xyz = nxyz[idx, 1:].cpu()

            boxes3d[:, 3:6] *= enlarge

            pts_in_flag = points_in_boxes_cpu(new_xyz, boxes3d)
            pts_label, pts_of_box = pts_in_flag.max(0)
            pts_label = pts_label.byte()
            center_offset = new_xyz - boxes3d[pts_of_box, :3]

            pts_labels.append(pts_label)
            center_offsets.append(center_offset)

        center_offsets = torch.cat(center_offsets).cuda()
        pts_labels = torch.cat(pts_labels).cuda()

        return pts_labels, center_offsets

    def get_loss(self):
        """ calculate aux loss 
        (points_mean, point_cls, point_reg) is the first three parameters
        gt_boxes come from batch_dict: (N, max_num_per_sample, 8)
        """
        points = self.forward_ret_dict['points_mean']
        gt_bboxes = self.forward_ret_dict['gt_boxes']

        N = len(gt_bboxes)

        # pts_labels, center_targets = self.build_aux_target(points, gt_bboxes)
        pts_labels, center_targets = self.build_aux_target(points, gt_bboxes)

        rpn_cls_target = pts_labels.float()
        pos = (pts_labels > 0).float()
        neg = (pts_labels == 0).float()

        pos_normalizer = pos.sum()
        pos_normalizer = torch.clamp(pos_normalizer, min=1.0)

        cls_weights = pos + neg
        cls_weights = cls_weights / pos_normalizer

        reg_weights = pos
        reg_weights = reg_weights / pos_normalizer

        point_cls = self.forward_ret_dict['point_cls']
        aux_loss_cls = loss_utils.weighted_sigmoid_focal_loss(
                        point_cls.view(-1), rpn_cls_target, weight=cls_weights, avg_factor=1.)
        aux_loss_cls /= N

        point_reg = self.forward_ret_dict['point_reg']
        aux_loss_reg = loss_utils.weighted_smoothl1(
                        point_reg, center_targets, beta=1 / 9., weight=reg_weights[..., None], avg_factor=1.)
        aux_loss_reg /= N

        aux_loss = aux_loss_reg + aux_loss_cls

        tb_dict = dict(aux_loss_cls = aux_loss_cls.item(), 
                       aux_loss_reg = aux_loss_reg.item(),
                       aux_loss = aux_loss.item())

        return aux_loss, tb_dict
    
def nearest_neighbor_interpolate(unknown, unknown_batch_cnt, known, known_batch_cnt, known_feats):
    """
    three_nn returns:
        dist: (N1 + N2 ..., 3)  l2 distance to the three nearest neighbors
        idx: (N1 + N2 ..., 3)  index of the three nearest neighbors, range [0, M1+M2+...]
    three_interpolate returns:
        out_tensor: (N1 + N2 ..., C)
    """
    dist, idx = pointnet2_utils.three_nn(unknown[:, 1:4], unknown_batch_cnt, known[:, 1:4], known_batch_cnt)
    dist_recip = 1.0 / (dist + 1e-8)
    norm = torch.sum(dist_recip, dim=1, keepdim=True)
    weight = dist_recip / norm
    interpolated_feats = pointnet2_utils.three_interpolate(known_feats, idx, weight)

    return interpolated_feats

def tensor2points(tensor, offset: torch.Tensor, voxel_size: torch.Tensor):
    # feature source voxel batch count
    batch_size = tensor.batch_size
    indices = tensor.indices.float()
    offset, voxel_size = offset.to(indices.device), voxel_size.to(indices.device)
    indices[:, 1:] = indices[:, [3, 2, 1]] * voxel_size + offset + .5 * voxel_size

    known_batch_cnt = indices.new_zeros(batch_size).int()
    for bs_ids in range(tensor.batch_size):
        known_batch_cnt[bs_ids] = (tensor.indices[:, 0] == bs_ids).sum()

    return tensor.features, indices, known_batch_cnt