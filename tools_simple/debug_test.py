import os

CONFIG_FILE = './cfgs/aw_models/centerpoint_iou.yaml'
CKPT = '/home/chk/OpenPCDet/data/output_aw/aw_centerpoint_aux_v1.0_half_repv2/cfgs/aw_models/centerpoint_aux/default/ckpt/checkpoint_epoch_80.pth'
gpu_index = [5]
sample_per_gpu = 4

NUM_GPUS = len(gpu_index)
BATCH_SIZE = NUM_GPUS * sample_per_gpu
CUDA_INDEX = ','.join(map(str, gpu_index))

os.chdir('/home/chk/OpenPCDet/tools')
os.system(f"export CUDA_VISIBLE_DEVICES='{CUDA_INDEX}' && \
            python test.py  \
            --cfg_file {CONFIG_FILE}    \
            --batch_size {BATCH_SIZE}   \
            --ckpt {CKPT}")