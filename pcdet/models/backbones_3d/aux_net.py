from torch import nn

# SA-SSD is [N, 4] but OPENPCDET is [B, N, 4]
from pcdet.ops.pointnet2.pointnet2_stack import pointnet2_utils
import torch
from ...ops.roiaware_pool3d.roiaware_pool3d_utils import points_in_boxes_cpu
from pcdet.utils.loss_utils import weighted_smoothl1, weighted_sigmoid_focal_loss


class AuxilNet(nn.Module):
    def __init__(self, model_cfg):
        super(AuxilNet, self).__init__()

        self.model_cfg = model_cfg

        self.forward_aux_dict = {}

        self.channel = model_cfg.POINT_FC_CLS_REG
        self.point_range = model_cfg.POINT_CLOUD_RANGE
        self.voxel_size = model_cfg.VOXEL_SIZE

        self.voxel_x = self.voxel_size[0]
        self.voxel_y = self.voxel_size[1]
        self.voxel_z = self.voxel_size[2]

        self.x_offset = self.voxel_x / 2 + self.point_range[0]
        self.y_offset = self.voxel_y / 2 + self.point_range[1]
        self.z_offset = self.voxel_z / 2 + self.point_range[2]

        self.point_fc = \
            nn.Linear(self.channel[0], self.channel[1], bias=False)
        self.point_cls = \
            nn.Linear(self.channel[1], self.channel[2], bias=False)
        self.point_reg = \
            nn.Linear(self.channel[1], self.channel[3], bias=False)

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

    def aux_loss(self):

        points, point_cls, point_reg, gt_bboxes = \
            self.forward_aux_dict["points_mean"], \
            self.forward_aux_dict["point_cls"], \
            self.forward_aux_dict["point_reg"], \
            self.forward_aux_dict["gt_boxes"]

        N = len(gt_bboxes)

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

        aux_loss_cls = weighted_sigmoid_focal_loss(
            point_cls.view(-1), rpn_cls_target, weight=cls_weights, avg_factor=1.)
        aux_loss_cls /= N

        aux_loss_reg = weighted_smoothl1(
            point_reg, center_targets, beta=1 / 9., weight=reg_weights[..., None], avg_factor=1.)
        aux_loss_reg /= N

        return aux_loss_cls + aux_loss_reg, dict(
            aux_loss_cls=aux_loss_cls.item(),
            aux_loss_reg=aux_loss_reg.item(),
        )

    def forward(self, batch_dict):
        voxel_features, voxel_num_points, voxel_coords = \
            batch_dict['voxels'], batch_dict['voxel_num_points'], batch_dict['voxel_coords']
        voxel_points_mean = voxel_features[:, :, :3].sum(dim=1, keepdim=True) \
            / voxel_num_points.type_as(voxel_features).view(-1, 1, 1)

        center = torch.zeros([voxel_features.size()[0], 3]).to(
            voxel_features.device)

        center[:, 0] = \
            voxel_coords[:, 3] * self.voxel_x + self.x_offset
        center[:, 1] = \
            voxel_coords[:, 2] * self.voxel_y + self.y_offset
        center[:, 2] = \
            voxel_coords[:, 1] * self.voxel_z + self.z_offset

        points_mean = torch.zeros([voxel_features.size()[0], 4]).to(
            voxel_features.device)
        voxel_points_mean = voxel_points_mean.squeeze(dim=1)
        points_mean[:, 0] = voxel_coords[:, 0]
        points_mean[:, 1:] = voxel_points_mean[:, :3]

        # auxiliary network
        vx_feat, vx_nxyz = batch_dict['pillar_features'], center

        unknown_batch_cnt = torch.zeros(
            points_mean.shape[0], dtype=torch.int).to(points_mean.device)
        known_batch_cnt = torch.zeros(
            vx_feat.shape[0], dtype=torch.int).to(vx_feat.device)

        unknown_batch_cnt = points_mean.new_zeros(
            batch_dict["batch_size"]).int()
        known_batch_cnt = vx_feat.new_zeros(
            batch_dict["batch_size"]).int()

        for idx in range(batch_dict["batch_size"]):
            unknown_batch_cnt[idx] = (voxel_coords[:, 0] == idx).sum()
            known_batch_cnt[idx] = (voxel_coords[:, 0] == idx).sum()

        unkown = points_mean[:, 1:]
        p0 = nearest_neighbor_interpolate(
            unkown, unknown_batch_cnt, vx_nxyz, vx_feat, known_batch_cnt)

        pointwise = self.point_fc(p0)
        point_cls = self.point_cls(pointwise)
        point_reg = self.point_reg(pointwise)

        self.forward_aux_dict["points_mean"] = points_mean
        self.forward_aux_dict["point_cls"] = point_cls
        self.forward_aux_dict["point_reg"] = point_reg
        self.forward_aux_dict["gt_boxes"] = batch_dict["gt_boxes"][:, :, :7]

        return batch_dict


def nearest_neighbor_interpolate(unknown, unknown_batch_cnt, known, known_feats, known_batch_cnt):
    #  unknown, unknown_batch_cnt, known, known_batch_cnt
    """
    :param pts: (n, 4) tensor of the bxyz positions of the unknown features
    :param ctr: (m, 4) tensor of the bxyz positions of the known features
    :param ctr_feats: (m, C) tensor of features to be propigated
    :return:
        new_features: (n, C) tensor of the features of the unknown features
    """

    """
    Args:
        ctx:
        unknown: (N1 + N2..., 3)
        unknown_batch_cnt: (batch_size), [N1, N2, ...]
        known: (M1 + M2..., 3)
        known_batch_cnt: (batch_size), [M1, M2, ...]

    Returns:
        dist: (N1 + N2 ..., 3)  l2 distance to the three nearest neighbors
        idx: (N1 + N2 ..., 3)  index of the three nearest neighbors, range [0, M1+M2+...]
    """

    dist, idx = pointnet2_utils.three_nn(
        unknown, unknown_batch_cnt, known, known_batch_cnt)
    dist_recip = 1.0 / (dist + 1e-8)
    norm = torch.sum(dist_recip, dim=1, keepdim=True)
    weight = dist_recip / norm
    interpolated_feats = pointnet2_utils.three_interpolate(
        known_feats, idx, weight)

    return interpolated_feats

