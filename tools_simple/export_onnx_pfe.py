import os

cfg_file = '/home/chk/OpenPCDet/tools/cfgs/aw_models/centerpoint_iou.yaml'
ckpt = '/home/chk/OpenPCDet/data/output_aw/aw_centerpoint_v1.1_iou/cfgs/aw_models/centerpoint_iou/default/ckpt/checkpoint_epoch_100.pth'

os.chdir('/home/chk/OpenPCDet/tools')

# os.system(f'python onnx_utils/trans_pfe.py \
#             --cfg_file {cfg_file} \
#             --ckpt {ckpt}')

os.system(f'python onnx_utils/trans_backbone.py \
            --cfg_file {cfg_file} \
            --ckpt {ckpt}')