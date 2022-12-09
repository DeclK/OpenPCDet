from pcdet.datasets import build_dataloader
from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.utils.common_utils import EasyPickle
from viewer.viewer import Viewer
from tqdm import tqdm

import os

def build_viewer(box_type="OpenPCDet", bg=(255,255,255), offscreen=False, remote=False):
    # in case you are working on a remote machine, specify DISPLAY value like this
    # checkout https://github.com/marcomusy/vedo/issues/64  
    if remote: os.environ['DISPLAY'] = ':99.0'
    return Viewer(box_type=box_type, bg=bg, offscreen=offscreen)

def main():
    cfg_file = '/OpenPCDet/tools/cfgs/once_models/second.yaml'
    seca = '/OpenPCDet/data/seca/semi_cg_ssd_lr_0.003_60_epoch_best/OpenPCDet/tools/cfgs/once_semi_models/mean_teacher_cg_ssd/default/eval/eval_with_teacher_model/epoch_60/val/result.pkl'
    second = '/OpenPCDet/data/seca/second_res/OpenPCDet/tools/cfgs/once_models/second_res/default/eval/eval_with_train/result.pkl'

    os.chdir("/OpenPCDet/tools")    # build cfg need
    cfg = cfg_from_yaml_file(cfg_file, cfg)
    test_set, test_loader, sampler = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        batch_size=1,
        dist=False,
        training=False
    )

    """ 
    Sample Keys:
    points, frame_id, gt_boxes, use_lead_xyz, voxels,
    voxel_coords, voxel_num_points, batch_size

    Result Keys:
    name, score, boxes_3d, frame_id
    """

    seca = EasyPickle.load(seca)
    second = EasyPickle.load(second)
    n = len(test_loader)
    pbar = tqdm(range(n))
    pbar.set_description('Visualizing')
    vi = build_viewer(offscreen=True)

    for idx, sample in enumerate(test_loader):
        # get item
        pred_seca = seca[idx]
        pred_frame_id, pred_boxes_3d, pred_score, pred_name = \
        pred_seca['frame_id'], pred_seca['boxes_3d'], pred_seca['score'], pred_seca['name']
        vi.add_3D_boxes(pred_boxes_3d, color='green')

        pred_second = second[idx]
        pred_frame_id, pred_boxes_3d, pred_score, pred_name = \
        pred_second['frame_id'], pred_second['boxes_3d'], pred_second['score'], pred_second['name']
        vi.add_3D_boxes(pred_boxes_3d, color='red')

        points, frame_id, gt_boxes = \
        sample['points'], sample['frame_id'][0], sample['gt_boxes'][0][:, :7]
        vi.add_3D_boxes(gt_boxes, color='blue')
        
        assert frame_id == pred_frame_id    # check frame id consistency
        # vis
        points = points[:, [1,2,3]]
        vi.add_points(points)
        out_file = '/OpenPCDet/data/vis_results/seca/' + frame_id + '.png'
        vi.show_3D(out_file, scale=2)
        
        pbar.update()

if __name__ == '__main__':
    main()