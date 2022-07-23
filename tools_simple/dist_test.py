import os

CONFIG_FILE = ''
gpu_index = []
samples_per_gpu = 4

NUM_GPUS = len(gpu_index)
BATCH_SIZE = NUM_GPUS * samples_per_gpu
CUDA_INDEX = ','.join(map(str, gpu_index))

# Given ckpt dir

# CKPT_DIR = ''
# START_EPOCH = 158

# os.chdir('')
# os.system(f"export CUDA_VISIBLE_DEVICES='{CUDA_INDEX}' && \
#             bash scripts/dist_test.sh {NUM_GPUS}          \
#             --cfg_file {CONFIG_FILE}    \
#             --batch_size {BATCH_SIZE}   \
#             --start_epoch {START_EPOCH} \
#             --ckpt_dir {CKPT_DIR}       \
#             --eval_all")

# Given ckpt file

CKPT = ''
os.chdir('')
os.system(f"export CUDA_VISIBLE_DEVICES='{CUDA_INDEX}' && \
            bash scripts/dist_test.sh {NUM_GPUS}          \
            --cfg_file {CONFIG_FILE}    \
            --batch_size {BATCH_SIZE}   \
            --ckpt {CKPT}"
            )