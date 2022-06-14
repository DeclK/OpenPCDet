import os

CONFIG_FILE = './cfgs/once_models/centerpoint_iou.yaml'
CKPT_DIR = '/home/chk/OpenPCDet/output/cfgs/once_models/centerpoint_iou/default/ckpt'
START_EPOCH = 70
gpu_index = [0,1,2,3]

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
            --eval_all")