import os
import time
from pathlib import Path
import re
# This is a script for grid search, it enumerates grid values, 
# and use these values in the corresponding key in config file 
# Please specific the parameters below:
# 1. grid values: list
# 2. config file information
# 3. pattern to substitude
# 4. running scripts


grid = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
config_file = Path('/OpenPCDet/tools/cfgs/once_semi_models/mean_teacher_cg_ssd.yaml')

CKPT = '/OpenPCDet/data/seca/semi_cg_ssd_lr_0.003_60_epoch_best/OpenPCDet/tools/cfgs/once_semi_models/mean_teacher_cg_ssd/default/ssl_ckpt/teacher/checkpoint_epoch_60.pth'
gpu_index = [0, 1, 2]
sample_per_gpu = 4
NUM_GPUS = len(gpu_index)
BATCH_SIZE = NUM_GPUS * sample_per_gpu
CUDA_INDEX = ','.join(map(str, gpu_index))

os.chdir('/OpenPCDet/tools')

def grid_search():
    for value in grid:
        content = config_file.read_text()
        pattern = re.compile(r'RECTIFIER: 0.\d*')
        ret = pattern.sub(f'RECTIFIER: {value}', content)
        config_file.write_text(ret)

        time.sleep(0.5)
        os.system(f"CUDA_VISIBLE_DEVICES={CUDA_INDEX} && MKL_SERVICE_FORCE_INTEL=1 &&\
                    bash scripts/dist_test.sh {NUM_GPUS}          \
                    --cfg_file {config_file}    \
                    --batch_size {BATCH_SIZE}   \
                    --ckpt {CKPT}"
                    )

def print_result_from_grid_search(dir):
    dir = Path(dir)
    result = ''
    for f in dir.iterdir():
        if f.suffix != '.txt': continue
        if f.name == 'all.txt': continue
        str = f.read_text()
        if len(str) < 10: continue
        pattern = re.compile(r'\|[\w\W]*\|')
        thresh = re.search(r'RECTIFIER: [01].\d*', str).group()
        ret = pattern.search(str).group()
        result += thresh + '\n' + ret + '\n'
    print(result)
    out_file = dir / 'all.txt'
    out_file.touch(exist_ok=True)
    out_file.write_text(result)


if __name__ == '__main__':
    output = '/OpenPCDet/output'
    print_result_from_grid_search(output)