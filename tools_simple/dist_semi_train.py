import os
from pathlib import Path

CONFIG_FILE = './cfgs/once_semi_models/mean_teacher_centerpoint.yaml'
pretrained_model = '/home/chk/OpenPCDet/checkpoints/centerpoint_epoch_80.pth'
gpu_index = [5,6,7]

NUM_GPUS = len(gpu_index)
CUDA_INDEX = ','.join(map(str, gpu_index))

os.chdir('/OpenPCDet/tools')
os.system(f"export CUDA_VISIBLE_DEVICES='{CUDA_INDEX}' &&   \
            bash scripts/dist_semi_train.sh {NUM_GPUS}      \
            --cfg_file {CONFIG_FILE} \
            --pretrained_model {pretrained_model}\
            --fix_random_seed")
