import os

CONFIG_FILE = './cfgs/aw_models/centerpoint.yaml'
gpu_index = [1]
sample_per_gpu = 8

NUM_GPUS = len(gpu_index)
BATCH_SIZE = NUM_GPUS * sample_per_gpu
CUDA_INDEX = ','.join(map(str, gpu_index))

os.chdir('/home/chk/OpenPCDet/tools')
os.system(f"export CUDA_VISIBLE_DEVICES='{CUDA_INDEX}' && \
            python train_with_eval.py --cfg_file {CONFIG_FILE} --batch_size {BATCH_SIZE} \
            --fix_random_seed")