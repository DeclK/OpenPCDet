import os
from pathlib import Path
work_dir = Path(__file__).resolve().parents[1] / 'tools'

CONFIG_FILE = '/OpenPCDet/tools/cfgs/once_models/centerpoint_aux_multi_head.yaml'
gpu_index = [0,1,2]
samples_per_gpu = 4

NUM_GPUS = len(gpu_index)
BATCH_SIZE = NUM_GPUS * samples_per_gpu
CUDA_INDEX = ','.join(map(str, gpu_index))

# Given ckpt dir

# CKPT_DIR = '/OpenPCDet/data/once_output/centerpoint_multi_head_aux/OpenPCDet/tools/cfgs/once_models/centerpoint_aux/default/ckpt'
# START_EPOCH = 1

# os.chdir(work_dir)
# os.system(f"export CUDA_VISIBLE_DEVICES='{CUDA_INDEX}' && \
#             bash scripts/dist_test.sh {NUM_GPUS}          \
#             --cfg_file {CONFIG_FILE}    \
#             --batch_size {BATCH_SIZE}   \
#             --start_epoch {START_EPOCH} \
#             --ckpt_dir {CKPT_DIR}       \
#             --eval_all")

# Given ckpt file

CKPT = '/OpenPCDet/data/seca/centerpoint_aux_v1.0/OpenPCDet/tools/cfgs/once_models/centerpoint_aux_multi_head/default/ckpt/checkpoint_epoch_80.pth'
os.chdir(work_dir)
os.system(f"export CUDA_VISIBLE_DEVICES='{CUDA_INDEX}' && \
            bash scripts/dist_test.sh {NUM_GPUS}          \
            --cfg_file {CONFIG_FILE}    \
            --batch_size {BATCH_SIZE}   \
            --ckpt {CKPT}"
            )