import sys
import torch
import torch.nn as nn
from pcdet.models.backbones_3d.vfe.pillar_vfe import PFNLayer, VFETemplate
import argparse
from pathlib import Path

from pcdet.config import cfg, cfg_from_yaml_file


def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default=None, help='specify the config for training')
    parser.add_argument('--ckpt', type=str, default=None, help='checkpoint to start from')
    parser.add_argument('--output', type=str, default=None, help='output directory of onnx')
    args = parser.parse_args()

    cfg_ = cfg_from_yaml_file(args.cfg_file, cfg) if args.cfg_file is not None else cfg
    return args, cfg_


class PillarVFE(VFETemplate):
    def __init__(self, model_cfg, num_point_features, voxel_size, point_cloud_range, **kwargs):
        super().__init__(model_cfg=model_cfg)

        self.use_norm = self.model_cfg.USE_NORM
        self.with_distance = self.model_cfg.WITH_DISTANCE
        self.use_absolute_xyz = self.model_cfg.USE_ABSLOTE_XYZ
        num_point_features += 5 if self.use_absolute_xyz else 3
        self.num_point_features = num_point_features

        self.num_filters = self.model_cfg.NUM_FILTERS
        assert len(self.num_filters) > 0
        num_filters = [num_point_features] + list(self.num_filters)

        pfn_layers = []
        for i in range(len(num_filters) - 1):
            in_filters = num_filters[i]
            out_filters = num_filters[i + 1]
            pfn_layers.append(
                PFNLayer(in_filters, out_filters, self.use_norm, last_layer=(i >= len(num_filters) - 2))
            )
        self.pfn_layers = nn.ModuleList(pfn_layers)

        self.voxel_x = voxel_size[0]
        self.voxel_y = voxel_size[1]
        self.voxel_z = voxel_size[2]
        self.x_offset = self.voxel_x / 2 + point_cloud_range[0]
        self.y_offset = self.voxel_y / 2 + point_cloud_range[1]
        self.z_offset = self.voxel_z / 2 + point_cloud_range[2]

    def forward(self, features, **kwargs):
        """
        We need to rewrite the forward path because ONNX requries 
        forward input only contains Tensor.
        So, assuming the features are already processed as
        (max_num_pillars, max_points_per_pillars, num_point_features + 6)
        extra 6 feats are offset to mean position (3) and voxel center (2)
        """
        for pfn in self.pfn_layers:
            features = pfn(features)
        features = features[:, 0, :]    # unsqueeze
        return features

def build_pfe(ckpt, cfg):
    pfe = PillarVFE(
        model_cfg=cfg.MODEL.VFE,
        num_point_features=cfg.DATA_CONFIG.DATA_AUGMENTOR.AUG_CONFIG_LIST[0]['NUM_POINT_FEATURES'],
        point_cloud_range=cfg.DATA_CONFIG.POINT_CLOUD_RANGE,
        voxel_size=cfg.DATA_CONFIG.DATA_PROCESSOR[2].VOXEL_SIZE)

    pfe.to('cuda').eval()

    checkpoint = torch.load(ckpt, map_location='cuda')
    dicts = {}
    for key in checkpoint["model_state"].keys():
        if "vfe" in key:
            dicts[key[4:]] = checkpoint["model_state"][key]
    pfe.load_state_dict(dicts, strict=False)

    max_num_pillars = cfg.DATA_CONFIG.DATA_PROCESSOR[2].MAX_NUMBER_OF_VOXELS['test']
    max_points_per_pillars = cfg.DATA_CONFIG.DATA_PROCESSOR[2].MAX_POINTS_PER_VOXEL
    dims_feature = pfe.num_point_features
    dummy_input = torch.ones((max_num_pillars, max_points_per_pillars,
                             dims_feature), dtype=torch.float32).cuda()
    return pfe, dummy_input

if __name__ == "__main__":
    args, cfg = parse_config()

    cfg_file = args.cfg_file
    ckpt_file = args.ckpt
    if args.output is None:
        output_dir = Path(__file__).resolve().parents[2] / 'onnx_model'
    else:
        output_dir = Path(args.output).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_onnx = output_dir / 'pfe.onnx'

    print('Build pfe')
    pfe, dummy_input = build_pfe(ckpt_file, cfg)
    
    print('Export to onnx')
    torch.onnx.export(pfe,
                      dummy_input,
                      output_onnx,
                      opset_version=12,
                      verbose=True,
                      do_constant_folding=True)

    print(f'saving to: {output_dir}')
    
