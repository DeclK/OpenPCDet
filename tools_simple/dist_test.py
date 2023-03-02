import os
from pathlib import Path
work_dir = Path(__file__).resolve().parents[1] / 'tools'

CONFIG_FILE = '/OpenPCDet/tools/cfgs/once_semi_models/mean_teacher_cg_ssd.yaml'
gpu_index = [0,1,2]
samples_per_gpu = 4

NUM_GPUS = len(gpu_index)
BATCH_SIZE = NUM_GPUS * samples_per_gpu
CUDA_INDEX = ','.join(map(str, gpu_index))

# Given ckpt dir

# CKPT_DIR = '/OpenPCDet/output/OpenPCDet/tools/cfgs/kitti_models/second_car/default/ckpt'
# START_EPOCH = 70

# os.chdir(work_dir)
# os.system(f"export CUDA_VISIBLE_DEVICES='{CUDA_INDEX}' && \
#             bash scripts/dist_test.sh {NUM_GPUS}          \
#             --cfg_file {CONFIG_FILE}    \
#             --batch_size {BATCH_SIZE}   \
#             --start_epoch {START_EPOCH} \
#             --ckpt_dir {CKPT_DIR}       \
#             --eval_all")

# Given ckpt file

CKPT = '/OpenPCDet/data/seca/semi_cg_ssd_lr_0.003_60_epoch_best/OpenPCDet/tools/cfgs/once_semi_models/mean_teacher_cg_ssd/default/ssl_ckpt/teacher/checkpoint_epoch_60.pth'
os.chdir(work_dir)
os.system(f"export CUDA_VISIBLE_DEVICES='{CUDA_INDEX}' && \
            bash scripts/dist_test.sh {NUM_GPUS}          \
            --cfg_file {CONFIG_FILE}    \
            --batch_size {BATCH_SIZE}   \
            --ckpt {CKPT}"
            )