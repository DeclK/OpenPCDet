# Implement SC-Conv backbone in AFDet v2
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from functools import partial
norm_fn = partial(nn.BatchNorm2d, eps=1e-3, momentum=0.01)


class SCConv(nn.Module):
    """
    Self-calibrate Module
    Params: 
        - inplanes & planes: normally inplanes == planes == original planes // 2
        - stride: normally is 1
        - norm_fn: norm_fn
    """
    def __init__(self, inplanes, planes, stride=1, pooling_r=4, norm_fn=None):
        super(SCConv, self).__init__()
        self.k2 = nn.Sequential(
                    nn.AvgPool2d(pooling_r, stride=pooling_r), 
                    nn.Conv2d(inplanes, planes, 3, stride=1, padding=1, bias=False),
                    norm_fn(planes),
                    )
        self.k3 = nn.Sequential(
                    nn.Conv2d(inplanes, planes, 3, stride=1, padding=1, bias=False),
                    norm_fn(planes),
                    )
        self.k4 = nn.Sequential(
                    nn.Conv2d(inplanes, planes, 3, stride=stride, padding=1, bias=False),
                    norm_fn(planes),
                    )

    def forward(self, x):
        identity = x

        out = torch.sigmoid(torch.add(identity, F.interpolate(self.k2(x), identity.size()[2:]))) # sigmoid(identity + k2)
        out = torch.mul(self.k3(x), out) # k3 * sigmoid(identity + k2)
        out = self.k4(out) # k4

        return out

class SCBottleneck(nn.Module):
    """ 
    SC-Block, to replace original convolution blcok
    Params:
        - inplanes & planes: normally inplanes == planes
        - stride: normally is 1
    """
    def __init__(self, inplanes, planes, stride=1, norm_fn=None):
        super(SCBottleneck, self).__init__()
        # split channels
        half_planes = planes // 2
        self.relu = nn.ReLU(inplace=True)
        self.conv1_a = nn.Sequential(
                            nn.Conv2d(inplanes, half_planes, 1, bias=False),
                            norm_fn(half_planes),
                            self.relu,
        )
        self.conv1_b = nn.Sequential(
                        nn.Conv2d(inplanes, half_planes, 1, bias=False),
                        norm_fn(half_planes),
                        self.relu,
        )
        self.k1 = nn.Sequential(
                    nn.Conv2d(half_planes, half_planes, 3, stride=stride, padding=1, bias=False),
                    norm_fn(half_planes),
                    self.relu,
        )
        self.scconv = nn.Sequential(
                        SCConv(half_planes, half_planes, stride=stride, norm_fn=norm_fn),
                        self.relu,
        )
        self.conv3 = nn.Sequential(
                        nn.Conv2d(half_planes * 2, planes, 1, bias=False),
                        norm_fn(planes),
        )

    def forward(self, x):
        residual = x

        out_a= self.conv1_a(x)
        out_b = self.conv1_b(x)

        out_a = self.k1(out_a)
        out_b = self.scconv(out_b)

        out = self.conv3(torch.cat([out_a, out_b], dim=1))

        out += residual
        out = self.relu(out)

        return out

class SCBEVBackbone(nn.Module):
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
                nn.Conv2d(cur_c_in, cur_c_out, 3, stride=cur_stride, padding=1, bias=False),
                norm_fn(cur_c_out),
                nn.ReLU()
            ]
            for _ in range(layer_nums[idx]):
                cur_layers.append(SCBottleneck(cur_c_out, cur_c_out, stride=1, norm_fn=norm_fn))
            self.blocks.append(nn.Sequential(*cur_layers))

            # upsample
            cur_up_stride = upsample_strides[idx]
            cur_c_up_out= num_upsample_filters[idx]
            if cur_up_stride > 1:
                self.deblocks.append(nn.Sequential(
                    nn.ConvTranspose2d(cur_c_out, cur_c_up_out, cur_up_stride, stride=cur_up_stride, bias=False),
                    norm_fn(cur_c_up_out),
                    nn.ReLU(),
                ))
            else:
                cur_up_stride = np.round(1 / cur_up_stride).astype(np.int)
                self.deblocks.append(nn.Sequential(
                    nn.Conv2d(cur_c_out, cur_c_up_out, cur_up_stride, stride=cur_up_stride, bias=False),
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