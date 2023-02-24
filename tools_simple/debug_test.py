import os
from pathlib import Path
workd_dir = Path(__file__).resolve().parents[1] / 'tools'

CONFIG_FILE = '/OpenPCDet/tools/cfgs/kitti_models/cg_ssd.yaml'
CKPT = '/OpenPCDet/data/output_with_fixed_seed/cg_ssd_and_second_sc/cfgs/kitti_models/cg_ssd/default/ckpt/checkpoint_epoch_80.pth'
gpu_index = [0]
sample_per_gpu = 1

NUM_GPUS = len(gpu_index)
BATCH_SIZE = NUM_GPUS * sample_per_gpu
CUDA_INDEX = ','.join(map(str, gpu_index))

os.chdir(workd_dir)
os.system(f"export CUDA_VISIBLE_DEVICES='{CUDA_INDEX}' && \
            python test.py  \
            --cfg_file {CONFIG_FILE}    \
            --batch_size {BATCH_SIZE}   \
            --ckpt {CKPT}")