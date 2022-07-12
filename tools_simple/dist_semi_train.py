import os
from pathlib import Path

CONFIG_FILE = ''
pretrained_model = ''
gpu_index = []

NUM_GPUS = len(gpu_index)
CUDA_INDEX = ','.join(map(str, gpu_index))

os.chdir('')
os.system(f"export CUDA_VISIBLE_DEVICES='{CUDA_INDEX}' &&   \
            bash scripts/dist_semi_train.sh {NUM_GPUS}      \
            --cfg_file {CONFIG_FILE} \
            --pretrained_model {pretrained_model}\
            --fix_random_seed")
