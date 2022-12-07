from pcdet.datasets import build_dataloader
from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.utils.common_utils import EasyPickle
from viewer.viewer import Viewer
from tqdm import tqdm

import os

def build_viewer(box_type="OpenPCDet", bg=(255,255,255), offscreen=False):
    return Viewer(box_type=box_type, bg=bg, offscreen=offscreen)

cfg_file = '/OpenPCDet/tools/cfgs/once_models/second.yaml'
result_file = '/OpenPCDet/data/seca/semi_cg_ssd_lr_0.003_60_epoch_best/OpenPCDet/tools/cfgs/once_semi_models/mean_teacher_cg_ssd/default/eval/eval_with_teacher_model/epoch_60/val/result.pkl'

os.chdir("/OpenPCDet/tools")    # build cfg need
cfg = cfg_from_yaml_file(cfg_file, cfg)
test_set, test_loader, sampler = build_dataloader(
    dataset_cfg=cfg.DATA_CONFIG,
    class_names=cfg.CLASS_NAMES,
    batch_size=1,
    dist=False,
    training=False
)

# sample keys
"""
points
frame_id
gt_boxes
use_lead_xyz
voxels
voxel_coords
voxel_num_points
batch_size
"""

# result
result = EasyPickle.load(result_file)
"""
name
score
boxes_3d
frame_id
"""
n = len(test_loader)
pbar = tqdm(range(n))
pbar.set_description('Visualizing')
vi = build_viewer(offscreen=True)

for idx, sample in enumerate(test_loader):
    pbar.update()
    # get items
    pred = result[idx]
    pred_frame_id, pred_boxes_3d, pred_score, pred_name = \
    pred['frame_id'], pred['boxes_3d'], pred['score'], pred['name']
    points, frame_id, gt_boxes = \
    sample['points'], sample['frame_id'][0], sample['gt_boxes']
    
    assert frame_id == pred_frame_id    # check frame id consistency
    # vis
    # vi.add_points(points)
    vi.add_3D_boxes(pred_boxes_3d)
    vi.show_3D()
    break