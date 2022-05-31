import os

CONFIG_FILE = './cfgs/once_semi_models/mean_teacher_centerpoint.yaml'
CKPT_DIR = '/OpenPCDet/output/cfgs/once_semi_models/mean_teacher_centerpoint/default/ssl_ckpt/student'
START_EPOCH = 20
gpu_index = [0,1,2]

NUM_GPUS = len(gpu_index)
BATCH_SIZE = NUM_GPUS * 4
CUDA_INDEX = ','.join(map(str, gpu_index))

os.chdir('/OpenPCDet/tools')
os.system(f"export CUDA_VISIBLE_DEVICES='{CUDA_INDEX}' && \
            bash scripts/dist_test.sh {NUM_GPUS}          \
            --cfg_file {CONFIG_FILE}    \
            --batch_size {BATCH_SIZE}   \
            --start_epoch {START_EPOCH} \
            --ckpt_dir {CKPT_DIR}       \
            --workers 0\
            --eval_all")