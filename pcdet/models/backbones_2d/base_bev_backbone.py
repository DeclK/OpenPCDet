import numpy as np
import torch
import torch.nn as nn

from functools import partial
norm_fn = partial(nn.BatchNorm2d, eps=1e-3, momentum=0.01)

# CHK MARK, rewritten BaseBEVBackbone for ONNX need, structure is the same, but code is simplified

class BaseBEVBackbone(nn.Module):
    def __init__(self, model_cfg, input_channels):
        super().__init__()
        self.model_cfg = model_cfg
        assert len(self.model_cfg.LAYER_NUMS) == len(self.model_cfg.LAYER_STRIDES) == len(self.model_cfg.NUM_FILTERS)
        assert len(self.model_cfg.LAYER_NUMS) == len(self.model_cfg.UPSAMPLE_STRIDES), 'must have upsample process'

        layer_nums = self.model_cfg.LAYER_NUMS
        layer_strides = self.model_cfg.LAYER_STRIDES
        num_filters = self.model_cfg.NUM_FILTERS
        num_upsample_filters = self.model_cfg.NUM_UPSAMPLE_FILTERS
        upsample_strides = self.model_cfg.UPSAMPLE_STRIDES

        num_levels = len(layer_nums) # normally is 2
        c_in_list = [input_channels, *num_filters[:-1]]
        self.blocks = nn.ModuleList()
        self.deblocks = nn.ModuleList()

        for idx in range(num_levels):
            # downsample
            cur_c_out = num_filters[idx]
            cur_c_in = c_in_list[idx]
            cur_stride = layer_strides[idx]

            cur_layers = [
                nn.Conv2d(cur_c_in, cur_c_out, 3,
                    stride=cur_stride, padding=1, bias=False),
                norm_fn(cur_c_out),
                nn.ReLU()
            ]
            for _ in range(layer_nums[idx]):
                cur_layers.extend([
                    nn.Conv2d(cur_c_out, cur_c_out, 3,
                              padding=1, bias=False),
                    norm_fn(cur_c_out),
                    nn.ReLU()
                ])
            self.blocks.append(nn.Sequential(*cur_layers))

            # upsample
            cur_up_stride = upsample_strides[idx]
            cur_c_up_out= num_upsample_filters[idx]
            if cur_up_stride > 1:
                self.deblocks.append(nn.Sequential(
                    nn.ConvTranspose2d(cur_c_out, cur_c_up_out, 
                        cur_up_stride,stride=cur_up_stride,bias=False),
                    norm_fn(cur_c_up_out),
                    nn.ReLU(),
                ))
            else:
                cur_up_stride = np.round(1 / cur_up_stride).astype(np.int)
                self.deblocks.append(nn.Sequential(
                    nn.Conv2d(cur_c_out, cur_c_up_out, cur_up_stride,
                        stride=cur_up_stride, bias=False),
                    norm_fn(cur_c_up_out),
                    nn.ReLU(),
                ))
        c_in = sum(num_upsample_filters)
        self.num_bev_features = c_in

    def forward(self, batch_dict, **kwags):
        x = batch_dict['spatial_features']
        ups = []
        for i in range(len(self.blocks)):
            x = self.blocks[i](x)
            ups.append(self.deblocks[i](x))
        x = torch.cat(ups, dim=1)
        batch_dict['spatial_features_2d'] = x
    
        return batch_dict
