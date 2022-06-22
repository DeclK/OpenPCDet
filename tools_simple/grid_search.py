import os
import time
from pathlib import Path
import re

# This is a script for grid search, it enumerates grid values, 
# and use these values in the corresponding key in config file 
# Please specific the parameters below:
    # 1. grid values: list
    # 2. config file information
    # 3. pattern to substitude
    # 4. running scripts


grid = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
config_file = Path('/home/chk/OpenPCDet/tempt_cfg.yaml')

CKPT = '/home/chk/OpenPCDet/data/output/centerpoint_iou_v1.2_only_iou_rectify/cfgs/once_models/centerpoint_iou/default/ckpt/checkpoint_epoch_80.pth'
gpu_index = [1]
sample_per_gpu = 4
NUM_GPUS = len(gpu_index)
BATCH_SIZE = NUM_GPUS * sample_per_gpu
CUDA_INDEX = ','.join(map(str, gpu_index))

os.chdir('/home/chk/OpenPCDet/tools')
# modify config
for value in grid:
    content = config_file.read_text()
    pattern = re.compile(r'RECTIFIER: 0.\d*')
    ret = pattern.sub(f'RECTIFIER: {value}', content)
    config_file.write_text(ret)

    time.sleep(0.5)
    os.system(f"export CUDA_VISIBLE_DEVICES='{CUDA_INDEX}' && \
                python test.py  \
                --cfg_file {config_file}    \
                --batch_size {BATCH_SIZE}   \
                --ckpt {CKPT}")