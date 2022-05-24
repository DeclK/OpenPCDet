import os

CONFIG_FILE = './cfgs/once_semi_models/mean_teacher_second.yaml'
CKPT_DIR = '/home/chk/OpenPCDet/output/cfgs/once_semi_models/mean_teacher_second/default/ssl_ckpt/student'
START_EPOCH = 21
gpu_index = [4,5,6,7]

NUM_GPUS = len(gpu_index)
BATCH_SIZE = NUM_GPUS * 4
CUDA_INDEX = ','.join(map(str, gpu_index))

os.chdir('/home/chk/OpenPCDet/tools')
os.system(f"export CUDA_VISIBLE_DEVICES='{CUDA_INDEX}' && \
            bash scripts/dist_test.sh {NUM_GPUS}          \
            --cfg_file {CONFIG_FILE}    \
            --batch_size {BATCH_SIZE}   \
            --start_epoch {START_EPOCH} \
            --ckpt_dir {CKPT_DIR}       \
            --workers 0\
            --eval_all")