import torch
from functools import partial
try:
    import spconv.pytorch as spconv
    from spconv.pytorch import ops
    from spconv.pytorch import SparseConv2d, SparseMaxPool2d, SparseInverseConv2d
except:
    import spconv
    from spconv import ops
    from spconv import SparseConv2d, SparseMaxPool2d, SparseInverseConv2d

from torch import nn
from .pillar_encoder_utils import bev_spatial_shape, Sparse2DBasicBlock, Sparse2DBasicBlockV


class SpMiddlePillarEncoder(nn.Module):
    def __init__(
            self, model_cfg, input_channels, grid_size, **kwargs):
        super(SpMiddlePillarEncoder, self).__init__()
        self.model_cfg = model_cfg
        self.input_channels = input_channels
        self.bev_width, self.bev_height, _ = grid_size
        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)

        self.conv1 = spconv.SparseSequential(
            Sparse2DBasicBlockV(32, 32, norm_fn=norm_fn, indice_key="res0"),
            Sparse2DBasicBlock(32, 32, norm_fn=norm_fn, indice_key="res0"),
        )

        self.conv2 = spconv.SparseSequential(
            SparseConv2d(
                32, 64, 3, 2, padding=1, bias=False
            ),  # [752, 752] -> [376, 376]
            norm_fn(64),
            nn.ReLU(),
            Sparse2DBasicBlock(64, 64, norm_fn=norm_fn, indice_key="res1"),
            Sparse2DBasicBlock(64, 64, norm_fn=norm_fn, indice_key="res1"),
        )

        self.conv3 = spconv.SparseSequential(
            SparseConv2d(
                64, 128, 3, 2, padding=1, bias=False
            ),  # [376, 376] -> [188, 188]
            norm_fn(128),
            nn.ReLU(),
            Sparse2DBasicBlock(128, 128, norm_fn=norm_fn, indice_key="res2"),
            Sparse2DBasicBlock(128, 128, norm_fn=norm_fn, indice_key="res2"),
        )

        self.conv4 = spconv.SparseSequential(
            SparseConv2d(
                128, 128, 3, 2, padding=1, bias=False
            ),
            norm_fn(128),
            nn.ReLU(),
            Sparse2DBasicBlock(128, 128, norm_fn=norm_fn, indice_key="res3"),
            Sparse2DBasicBlock(128, 128, norm_fn=norm_fn, indice_key="res3"),
        )

        self.num_point_features = 128

    def forward(self, batch_dict):
        # transfer pillar features into sparse tensor
        pillar_features = batch_dict['pillar_features']
        pillar_indices = batch_dict['voxel_coords'][:, [0, 2, 3]].int()
        batch_size = batch_dict['batch_size']
        sp_tensor = spconv.SparseConvTensor(pillar_features, pillar_indices, (self.bev_height, self.bev_width), batch_size)

        x_conv1 = self.conv1(sp_tensor)
        x_conv2 = self.conv2(x_conv1)
        x_conv3 = self.conv3(x_conv2)
        x_conv4 = self.conv4(x_conv3)

        batch_dict['encoded_spconv_tensor'] = x_conv4
        batch_dict['encoded_spconv_tensor_stride'] = 8
        return batch_dict