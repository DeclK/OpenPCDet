import torch
import torch.nn as nn
import numpy as np
import argparse
from pathlib import Path

from pcdet.config import cfg, cfg_from_yaml_file
from onnx_networks import BaseBEVBackbone, CenterHeadIoU


class BackboneWithDenseHead(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        backbone_channels = cfg.MODEL.MAP_TO_BEV.NUM_BEV_FEATURES
        backbone_cfg = cfg.MODEL.BACKBONE_2D
        self.backbone_2d = BaseBEVBackbone(backbone_cfg, backbone_channels)

        dense_head_cfg = cfg.MODEL.DENSE_HEAD
        input_channels = sum(backbone_cfg.NUM_UPSAMPLE_FILTERS)
        num_class = len(cfg.CLASS_NAMES)
        class_names = cfg.CLASS_NAMES
        point_cloud_range = cfg.DATA_CONFIG.POINT_CLOUD_RANGE
        voxel_size = cfg.DATA_CONFIG.DATA_PROCESSOR[2].VOXEL_SIZE

        self.dense_head = CenterHeadIoU(
            model_cfg=dense_head_cfg,
            input_channels=input_channels,
            num_class=num_class,
            class_names=class_names,
            grid_size=None,
            point_cloud_range=point_cloud_range,
            voxel_size=voxel_size,
        )
    def forward(self, x):
        x = self.backbone_2d(x)
        scores, boxes = self.dense_head(x)
        return scores, boxes

def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default=None, help='specify the config for training')
    parser.add_argument('--ckpt', type=str, default=None, help='checkpoint to start from')
    parser.add_argument('--output', type=str, default=None, help='output directory of onnx')
    args = parser.parse_args()

    cfg_ = cfg_from_yaml_file(args.cfg_file, cfg) if args.cfg_file is not None else cfg
    return args, cfg_

def build_backbone(ckpt, cfg):

    backbone = BackboneWithDenseHead(cfg)
    backbone.to('cuda').eval()

    checkpoint = torch.load(ckpt, map_location='cuda')
    dicts = {}
    for key in checkpoint["model_state"].keys():
        if "backbone_2d" in key:
            dicts[key] = checkpoint["model_state"][key]
        if "dense_head" in key:
            dicts[key] = checkpoint["model_state"][key]
    backbone.load_state_dict(dicts, strict=False)

    pc_range = np.array(cfg.DATA_CONFIG.POINT_CLOUD_RANGE)
    voxel_size = np.array(cfg.DATA_CONFIG.DATA_PROCESSOR[2]['VOXEL_SIZE'])
    grid_size = (pc_range[3:] - pc_range[:3]) / voxel_size
    gridx = np.round(grid_size[0]).astype(np.int32)
    gridy = np.round(grid_size[1]).astype(np.int32)
    channels = cfg.MODEL.MAP_TO_BEV.NUM_BEV_FEATURES
    batch_size = 1
    dummy_input = torch.ones((batch_size, channels,
                    gridy, gridx), dtype=torch.float32).cuda()
    return backbone, dummy_input

if __name__ == "__main__":
    args, cfg = parse_config()

    cfg_file = args.cfg_file
    ckpt_file = args.ckpt
    if args.output is None:
        output_dir = Path(__file__).resolve().parents[2] / 'onnx_model'
    else:
        output_dir = Path(args.output).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_onnx = output_dir / 'backbone.onnx'

    print('Build backbone')
    backbone, dummy_input = build_backbone(ckpt_file, cfg)
    
    print('Export to onnx')
    torch.onnx.export(backbone,
                      dummy_input,
                      output_onnx,
                      opset_version=12,
                      verbose=True,
                      do_constant_folding=True)

    print(f'saving to: {output_dir}')
    