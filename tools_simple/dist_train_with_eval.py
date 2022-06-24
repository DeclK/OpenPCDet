import os

CONFIG_FILE = './cfgs/aw_models/centerpoint.yaml'
gpu_index = [2,3]
sample_per_gpu = 8
CKPT_CHEK_INTERVAL = 1

NUM_GPUS = len(gpu_index)
BATCH_SIZE = NUM_GPUS * sample_per_gpu
CUDA_INDEX = ','.join(map(str, gpu_index))

os.chdir('/home/chk/OpenPCDet/tools')
os.system(f"export CUDA_VISIBLE_DEVICES='{CUDA_INDEX}' && \
            bash scripts/dist_train_with_eval.sh {NUM_GPUS}         \
            --cfg_file {CONFIG_FILE}  \
            --batch_size {BATCH_SIZE} \
            --ckpt_save_interval {CKPT_CHEK_INTERVAL} \
            --fix_random_seed")
