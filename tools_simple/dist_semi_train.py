import os
from pathlib import Path
work_dir = Path(__file__).resolve().parents[1] / 'tools'

CONFIG_FILE = '/OpenPCDet/tools/cfgs/once_semi_models/mean_teacher_cg_ssd.yaml'
pretrained_model = '/OpenPCDet/data/secg/cg_ssd_single_head/OpenPCDet/tools/cfgs/once_models/cg_ssd/default/ckpt/checkpoint_epoch_80.pth'
gpu_index = [0,1,2]

NUM_GPUS = len(gpu_index)
CUDA_INDEX = ','.join(map(str, gpu_index))

os.chdir(work_dir)
os.system(f"export CUDA_VISIBLE_DEVICES='{CUDA_INDEX}' &&   \
            bash scripts/dist_semi_train.sh {NUM_GPUS}      \
            --cfg_file {CONFIG_FILE} \
            --pretrained_model {pretrained_model}\
            --fix_random_seed")
