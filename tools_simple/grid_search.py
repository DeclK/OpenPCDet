import os
import numpy as np
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


grid = np.round(np.linspace(0.4, 0.8, num=41), 2)
grid = grid.tolist()
config_file = Path('/home/chk/OpenPCDet/tools/cfgs/aw_models/centerpoint_iou.yaml')

CKPT = '/home/chk/OpenPCDet/data/output_aw/aw_centerpoint_v1.1_iou/cfgs/aw_models/centerpoint_iou/default/ckpt/checkpoint_epoch_100.pth'
gpu_index = [0, 1, 2, 3]
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
                bash scripts/dist_test.sh {NUM_GPUS}          \
                --cfg_file {config_file}    \
                --batch_size {BATCH_SIZE}   \
                --ckpt {CKPT}"
                )

# Get result from output

# from pathlib import Path
# import re
# dir = '/home/chk/OpenPCDet/output/home/chk/OpenPCDet/tools/cfgs/aw_models/centerpoint_iou/default/eval/epoch_100/test/default'
# dir = Path(dir)
# for f in dir.iterdir():
#     str = f.read_text()
#     pattern = re.compile(r'\|[\w\W]*\|')
#     thresh = re.search(r'RECTIFIER: 0.\d*', str).group()
#     ret = pattern.search(str).group()
#     print(thresh)
#     print(ret)