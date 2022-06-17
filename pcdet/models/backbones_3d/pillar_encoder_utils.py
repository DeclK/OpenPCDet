import torch
import numpy as np
from torch import nn
try:
    import spconv.pytorch as spconv
    from spconv.pytorch import ops
except:
    import spconv

def replace_feature(out, new_features):
    if "replace_feature" in out.__dir__():
        # spconv 2.x behaviour
        return out.replace_feature(new_features)
    else:
        out.features = new_features
        return out

        
def conv2D3x3(in_planes, out_planes, stride=1, dilation=1, indice_key=None, bias=True):
    """3x3 convolution with padding to keep the same input and output"""
    assert stride >= 1
    padding = dilation
    if stride == 1:
        return spconv.SubMConv2d(
            in_planes,
            out_planes,
            kernel_size=3,
            stride=stride,
            dilation=dilation,
            padding=padding,
            bias=bias,
            indice_key=indice_key,
        )
    else:
        return spconv.SparseConv2d(
            in_planes,
            out_planes,
            kernel_size=3,
            stride=stride,
            dilation=dilation,
            padding=padding,
            bias=bias,
            indice_key=indice_key,
        )

def conv2D1x1(in_planes, out_planes, bias=False):
    """1x1 convolution"""
    return spconv.SubMConv2d(
            in_planes,
            out_planes,
            kernel_size=1,
            stride=1,
            dilation=1,
            padding=0,
            bias=bias,
            # indice_key=indice_key,
        )

class Sparse2DBasicBlock(spconv.SparseModule):
    expansion = 1

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        norm_fn=None,
        indice_key=None,
    ):
        super(Sparse2DBasicBlock, self).__init__()
        bias = norm_fn is not None

        self.batch_norm1 = norm_fn(planes)
        self.batch_norm2 = norm_fn(planes)

        self.conv1 = spconv.SparseSequential(
            conv2D3x3(inplanes, planes, stride, indice_key=indice_key, bias=bias),
            self.batch_norm1,
        )
        self.conv2 = spconv.SparseSequential(
            conv2D3x3(planes, planes, indice_key=indice_key, bias=bias),
            self.batch_norm2,
        )
        self.relu = nn.ReLU()
        self.stride = stride

    def forward(self, x):
        identity = x.features

        out = self.conv1(x)
        out = replace_feature(out, self.relu(out.features))
        out = self.conv2(out)

        out = replace_feature(out, out.features + identity)
        out = replace_feature(out, self.relu(out.features))

        return out

class Sparse2DBasicBlockV(spconv.SparseModule):
    expansion = 1

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        norm_fn=None,
        indice_key=None,
    ):
        super(Sparse2DBasicBlockV, self).__init__()
        bias = norm_fn is not None

        self.batch_norm1 = norm_fn(planes)
        self.batch_norm2 = norm_fn(planes)
        self.batch_norm3 = norm_fn(planes)

        self.conv0 = spconv.SparseSequential(
            conv2D3x3(inplanes, planes, stride, indice_key=indice_key, bias=bias),
            self.batch_norm1,
        )
        self.conv1 = spconv.SparseSequential(
            conv2D3x3(planes, planes, stride, indice_key=indice_key, bias=bias),
            self.batch_norm2,
        )
        self.conv2 = spconv.SparseSequential(
            conv2D3x3(planes, planes, indice_key=indice_key, bias=bias),
            self.batch_norm3,
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv0(x)
        identity = x.features

        out = self.conv1(x)
        out = replace_feature(out, self.relu(out.features))
        out = self.conv2(out)

        out = replace_feature(out, out.features + identity)
        out = replace_feature(out, self.relu(out.features))

        return out

def bev_spatial_shape(point_cloud_range, bev_size):
    W = round((point_cloud_range[3] - point_cloud_range[0]) / bev_size)
    H = round((point_cloud_range[4] - point_cloud_range[1]) / bev_size)
    return int(H), int(W)

class Dense2DBasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        norm_fn=None,
    ):
        super(Dense2DBasicBlock, self).__init__()
        self.batch_norm1 = norm_fn(planes)
        self.batch_norm2 = norm_fn(planes)

        self.conv1 = nn.Sequential(
            nn.Conv2d(inplanes, planes, 3, stride=stride, padding=1, bias=False),
            self.batch_norm1,
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(planes, planes, 3, stride=stride, padding=1, bias=False),
            self.batch_norm2,
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)

        out += identity
        out = self.relu(out)

        return out

class Dense2DBasicBlockV(nn.Module):
    expansion = 1

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        norm_fn=None,
    ):
        super(Dense2DBasicBlockV, self).__init__()
        self.batch_norm1 = norm_fn(planes)
        self.batch_norm2 = norm_fn(planes)
        self.batch_norm3 = norm_fn(planes)

        self.conv0 = nn.Sequential(
            nn.Conv2d(inplanes, planes, 3, stride=stride, padding=1, bias=False),
            self.batch_norm1,
            nn.ReLU()
        )
        self.conv1 = nn.Sequential(
            nn.Conv2d(planes, planes, 3, stride=1, padding=1, bias=False),
            self.batch_norm2,
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(planes, planes, 3, stride=1, padding=1, bias=False),
            self.batch_norm3,
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv0(x)

        identity = x

        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)

        out += identity
        out = self.relu(out)

        return out