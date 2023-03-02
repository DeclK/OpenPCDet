import os
from pathlib import Path
work_dir = Path(__file__).resolve().parents[1] / 'tools'

CONFIG_FILE = '/OpenPCDet/tools/cfgs/once_models/cg_ssd.yaml'
# pretrained_model = '/OpenPCDet/data/seca/00kitti/seca_kitti/OpenPCDet/tools/cfgs/kitti_models/cg_ssd/default/ckpt/checkpoint_epoch_80.pth'
gpu_index = [0]
sample_per_gpu = 2

NUM_GPUS = len(gpu_index)
BATCH_SIZE = NUM_GPUS * sample_per_gpu
CUDA_INDEX = ','.join(map(str, gpu_index))

os.chdir(work_dir)
os.system(f"export CUDA_VISIBLE_DEVICES='{CUDA_INDEX}' && \
            python train.py --cfg_file {CONFIG_FILE} --batch_size {BATCH_SIZE} \
            --workers 0\
            --fix_random_seed")