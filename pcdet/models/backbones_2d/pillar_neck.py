import numpy as np
from functools import partial

try:
    import spconv.pytorch as spconv
except:
    import spconv

import torch
from torch import nn

norm_fn = partial(nn.BatchNorm2d, eps=1e-3, momentum=0.01)

class NECKV2(nn.Module):
    def __init__(self, model_cfg, input_channels, **kwargs):
        super(NECKV2, self).__init__()
        self.model_cfg = model_cfg

        self._layer_strides = model_cfg.LAYER_STRIDES
        self._num_filters = model_cfg.NUM_FILTERS
        self._layer_nums = model_cfg.LAYER_NUMS
        self._upsample_strides = model_cfg.UPSAMPLE_STRIDES
        self._num_upsample_filters = model_cfg.NUM_UPSAMPLE_FILTERS
        self._num_input_features = model_cfg.NUM_INPUT_FEATURES

        assert len(self._layer_strides) == len(self._layer_nums)
        assert len(self._num_filters) == len(self._layer_nums)
        assert len(self._num_upsample_filters) == len(self._upsample_strides)

        self._upsample_start_idx = len(self._layer_nums) - len(self._upsample_strides)

        must_equal_list = []
        for i in range(len(self._upsample_strides)):
            must_equal_list.append(
                self._upsample_strides[i]
                / np.prod(self._layer_strides[: i + self._upsample_start_idx + 1])
            )

        for val in must_equal_list:
            assert val == must_equal_list[0]

        self.block_5, num_out_filters = self._make_layer(
            self._num_input_features[1],
            self._num_filters[1],
            self._layer_nums[1],
            stride=1,
        )
        self.deblock_5 = nn.Sequential(
            nn.ConvTranspose2d(
                num_out_filters,
                self._num_upsample_filters[1],
                2,
                stride=2,
                bias=False,
            ),
            norm_fn(self._num_upsample_filters[1]),
            nn.ReLU(),
        )

        self.deblock_4 = nn.Sequential(
            nn.ZeroPad2d(1),
            nn.Conv2d(self._num_input_features[0], self._num_upsample_filters[0], 3, stride=1, bias=False),
            norm_fn(self._num_upsample_filters[0]),
            nn.ReLU(),
        )
        self.block_4, num_out_filters = self._make_layer(
            self._num_upsample_filters[0] + self._num_upsample_filters[1],
            self._num_upsample_filters[0] + self._num_upsample_filters[1],
            self._layer_nums[0],
            stride=1,
        )
        self.num_bev_features = self._num_upsample_filters[0] + self._num_upsample_filters[1]

    @property
    def downsample_factor(self):
        factor = np.prod(self._layer_strides)
        if len(self._upsample_strides) > 0:
            factor /= self._upsample_strides[-1]
        return factor

    def _make_layer(self, inplanes, planes, num_blocks, stride=1):
        block = [
            nn.ZeroPad2d(1),
            nn.Conv2d(inplanes, planes, 3, stride=stride, bias=False),
            norm_fn(planes),
            nn.ReLU()
        ]

        for _ in range(num_blocks):
            block.append(nn.Conv2d(planes, planes, 3, padding=1, bias=False))
            block.append(norm_fn(planes))
            block.append(nn.ReLU())

        return nn.Sequential(*block), planes

    def forward(self, batch_dict, **kwargs):
        x_conv4 = batch_dict['encoded_tensor_8x']
        x_conv5 = batch_dict['encoded_tensor_16x']

        ups = [self.deblock_4(x_conv4)]
        x = self.block_5(x_conv5)
        ups.append(self.deblock_5(x))
        x = torch.cat(ups, dim=1)
        x = self.block_4(x)

        batch_dict['spatial_features_2d'] = x
        return batch_dict

