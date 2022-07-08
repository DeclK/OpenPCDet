import os

cfg_file = ''
ckpt = ''

os.chdir('/home/chk/OpenPCDet/tools')

os.system(f'python onnx_utils/trans_pfe.py \
            --cfg_file {cfg_file} \
            --ckpt {ckpt}')

os.system(f'python onnx_utils/trans_backbone.py \
            --cfg_file {cfg_file} \
            --ckpt {ckpt}')