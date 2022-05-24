import os

CONFIG_FILE = './cfgs/once_models/second.yaml'
CKPT = '/OpenPCDet/checkpoints/second_epoch_80.pth'
gpu_index = [1]
sample_per_gpu = 4

NUM_GPUS = len(gpu_index)
BATCH_SIZE = NUM_GPUS * sample_per_gpu
CUDA_INDEX = ','.join(map(str, gpu_index))

os.chdir('/OpenPCDet/tools')
os.system(f"export CUDA_VISIBLE_DEVICES='{CUDA_INDEX}' && \
            python test.py  \
            --cfg_file {CONFIG_FILE}    \
            --batch_size {BATCH_SIZE}   \
            --ckpt {CKPT}")