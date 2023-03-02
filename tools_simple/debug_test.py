import os
from pathlib import Path
workd_dir = Path(__file__).resolve().parents[1] / 'tools'

CONFIG_FILE = '/OpenPCDet/tools/cfgs/once_semi_models/mean_teacher_cg_ssd.yaml'
CKPT = '/OpenPCDet/data/seca/semi_cg_ssd_lr_0.003_60_epoch_best/OpenPCDet/tools/cfgs/once_semi_models/mean_teacher_cg_ssd/default/ssl_ckpt/teacher/checkpoint_epoch_60.pth'
gpu_index = [0]
sample_per_gpu = 4

NUM_GPUS = len(gpu_index)
BATCH_SIZE = NUM_GPUS * sample_per_gpu
CUDA_INDEX = ','.join(map(str, gpu_index))

os.chdir(workd_dir)
os.system(f"export CUDA_VISIBLE_DEVICES='{CUDA_INDEX}' && \
            python test.py  \
            --cfg_file {CONFIG_FILE}    \
            --batch_size {BATCH_SIZE}   \
            --ckpt {CKPT}")