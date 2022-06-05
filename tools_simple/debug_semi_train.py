import os

CONFIG_FILE = './cfgs/once_semi_models/mean_teacher_centerpoint.yaml'
pretrained_model = '/home/chk/OpenPCDet/checkpoints/centerpoint_epoch_80.pth'
# CONFIG_FILE = './cfgs/once_semi_models/mean_teacher_second.yaml'
# pretrained_model = '/OpenPCDet/checkpoints/second_epoch_80.pth'
gpu_index = [1]
NUM_GPUS = len(gpu_index)
CUDA_INDEX = ','.join(map(str, gpu_index))

os.chdir('/home/chk/OpenPCDet/tools')
os.system(f"export CUDA_VISIBLE_DEVICES='{CUDA_INDEX}' && \
            python semi_train.py    \
            --cfg_file {CONFIG_FILE}\
            --pretrained_model {pretrained_model}\
            --fix_random_seed")