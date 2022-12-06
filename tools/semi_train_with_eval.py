import argparse
import datetime
import glob
import os
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn
from tensorboardX import SummaryWriter

from pcdet.config import cfg, cfg_from_list, cfg_from_yaml_file, log_config_to_file
from pcdet.datasets import build_semi_dataloader
from pcdet.models import build_network
from pcdet.utils import common_utils
from train_utils.optimization import build_optimizer, build_scheduler
from ssl_utils.semi_train_utils import train_ssl_model_with_eval
from test import repeat_eval_ckpt

def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default=None, help='specify the config for training')
    parser.add_argument('--workers', type=int, default=8, help='number of workers for dataloader')
    parser.add_argument('--extra_tag', type=str, default='default', help='extra tag for this experiment')
    parser.add_argument('--ckpt', type=str, default=None, help='checkpoint to start from')
    parser.add_argument('--pretrained_model', type=str, default=None, help='pretrained_model')
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm'], default='none')
    parser.add_argument('--tcp_port', type=int, default=8888, help='tcp port for distrbuted training')
    parser.add_argument('--sync_bn', action='store_true', default=False, help='whether to use sync bn')
    parser.add_argument('--fix_random_seed', action='store_true', default=False, help='')
    parser.add_argument('--ckpt_save_interval', type=int, default=1, help='number of training epochs')
    parser.add_argument('--local_rank', type=int, default=0, help='local rank for distributed training')
    parser.add_argument('--max_ckpt_save_num', type=int, default=30, help='max number of saved checkpoint')
    parser.add_argument('--merge_all_iters_to_one_epoch', action='store_true', default=False, help='')
    parser.add_argument('--set', dest='set_cfgs', default=None, nargs=argparse.REMAINDER,
                        help='set extra config keys if needed')

    parser.add_argument('--max_waiting_mins', type=int, default=1, help='max waiting minutes')
    parser.add_argument('--start_epoch', type=int, default=0, help='')
    parser.add_argument('--save_to_file', action='store_true', default=False, help='')

    parser.add_argument('--runs_on', type=str, default='server', choices=['server', 'cloud'],help='runs on server or cloud')

    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)
    cfg.TAG = Path(args.cfg_file).stem
    cfg.EXP_GROUP_PATH = '/'.join(args.cfg_file.split('/')[1:-1])  # remove 'cfgs' and 'xxxx.yaml'

    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs, cfg)

    return args, cfg

# CHK MARK, add if no unlabeled data situation
class DistStudent(nn.Module):
    def __init__(self, student):
        super().__init__()
        self.onepass = student

    def forward(self, ld_batch, ud_batch):
        if ld_batch is not None and ud_batch is not None:
            return self.onepass(ld_batch), self.onepass(ud_batch)
        elif ld_batch is None:
            return None, self.onepass(ud_batch)
        else: return self.onepass(ld_batch), None

class DistTeacher(nn.Module):
    def __init__(self, teacher):
        super().__init__()
        self.onepass = teacher

    def forward(self, ld_batch, ud_batch):
        if ld_batch is not None and ud_batch is not None:
            return self.onepass(ld_batch), self.onepass(ud_batch)
        elif ld_batch is None:
            return None, self.onepass(ud_batch)
        else: return self.onepass(ld_batch), None

def main():
    args, cfg = parse_config()
    
    if args.runs_on == 'cloud':
        cfg.DATA_CONFIG.DATA_PATH = cfg.DATA_CONFIG.CLOUD_DATA_PATH

    if args.launcher == 'none':
        dist_train = False
    else:
        total_gpus, cfg.LOCAL_RANK = getattr(common_utils, 'init_dist_%s' % args.launcher)(
            args.tcp_port, args.local_rank, backend='nccl'
        )
        dist_train = True

    if args.fix_random_seed:
        common_utils.set_random_seed(666)

    output_dir = cfg.ROOT_DIR / 'output' / cfg.EXP_GROUP_PATH / cfg.TAG / args.extra_tag
    if args.runs_on == 'cloud':
        output_dir = Path('/cache/output/') / cfg.EXP_GROUP_PATH / cfg.TAG

    pretrain_ckpt_dir = output_dir / 'pretrain_ckpt'
    ssl_ckpt_dir = output_dir / 'ssl_ckpt'
    student_ckpt_dir = output_dir / 'ssl_ckpt' / 'student'
    teacher_ckpt_dir = output_dir / 'ssl_ckpt' / 'teacher'
    eval_output_dir = output_dir / 'eval'   # add eval output dir

    output_dir.mkdir(parents=True, exist_ok=True)
    pretrain_ckpt_dir.mkdir(parents=True, exist_ok=True)
    student_ckpt_dir.mkdir(parents=True, exist_ok=True)
    teacher_ckpt_dir.mkdir(parents=True, exist_ok=True)
    eval_output_dir.mkdir(parents=True, exist_ok=True)

    log_file = output_dir / ('log_train_%s.txt' % datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
    logger = common_utils.create_logger(log_file, rank=cfg.LOCAL_RANK)

    # log to file
    logger.info('**********************Start logging**********************')
    gpu_list = os.environ['CUDA_VISIBLE_DEVICES'] if 'CUDA_VISIBLE_DEVICES' in os.environ.keys() else 'ALL'
    logger.info('CUDA_VISIBLE_DEVICES=%s' % gpu_list)

    for key, val in vars(args).items():
        logger.info('{:16} {}'.format(key, val))
    log_config_to_file(cfg, logger=logger)
    if cfg.LOCAL_RANK == 0:
        os.system('cp %s %s' % (args.cfg_file, output_dir))

    tb_log = SummaryWriter(log_dir=str(output_dir / 'tensorboard')) if cfg.LOCAL_RANK == 0 else None

    batch_size = {
        'pretrain': cfg.OPTIMIZATION.PRETRAIN.BATCH_SIZE_PER_GPU,
        'labeled': cfg.OPTIMIZATION.SEMI_SUP_LEARNING.LD_BATCH_SIZE_PER_GPU,
        'unlabeled': cfg.OPTIMIZATION.SEMI_SUP_LEARNING.UD_BATCH_SIZE_PER_GPU,
        'test': cfg.OPTIMIZATION.TEST.BATCH_SIZE_PER_GPU,
    }
    # -----------------------create dataloader & network & optimizer---------------------------
    datasets, dataloaders, samplers = build_semi_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        batch_size=batch_size,
        dist=dist_train,
        root_path=cfg.DATA_CONFIG.DATA_PATH,
        workers=args.workers,
        logger=logger,
        merge_all_iters_to_one_epoch=args.merge_all_iters_to_one_epoch,
    )

    # --------------------------------stage II SSL training---------------------------------------
    logger.info('************************Stage II SSL training************************')
    teacher_model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=datasets['labeled'])
    """
    for param in teacher_model.parameters(): # ema teacher model
        param.detach_()
    """
    student_model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=datasets['labeled'])
    teacher_model.set_model_type('teacher')
    student_model.set_model_type('student')

    teacher_model.cuda()
    student_model.cuda()

    # only update student model by gradient descent, teacher model are updated by EMA
    student_optimizer = build_optimizer(student_model, cfg.OPTIMIZATION.SEMI_SUP_LEARNING.STUDENT)

    last_epoch = -1
    start_epoch = it = 0
    if args.pretrained_model or cfg.get('USE_PRETRAIN_MODEL', False):   # CHK MARK, modify pretrained model
        cfg.PRETRAIN_CKPT=  args.pretrained_model
        pretrained_model = cfg.PRETRAIN_CKPT
        if args.runs_on == 'cloud':
            pretrained_model = cfg.CLOUD_PRETRAIN_CKPT
    else:
        ckpt_list = glob.glob(str(pretrain_ckpt_dir / '*checkpoint_epoch_*.pth'))
        if len(ckpt_list) > 0:
            ckpt_list.sort(key=os.path.getmtime)
            pretrained_model = ckpt_list[-1]
            it, start_epoch = student_model.load_params_with_optimizer(
                ckpt_list[-1], to_cpu=dist_train, optimizer=student_optimizer, logger=logger
            )
            last_epoch = start_epoch + 1
        else: raise RuntimeError('No pretrained model or ckpt found')

    teacher_model.load_params_from_file(filename=pretrained_model, to_cpu=dist, logger=logger)
    student_model.load_params_from_file(filename=pretrained_model, to_cpu=dist, logger=logger)

    if dist_train:
        student_model = DistStudent(student_model) # add wrapper for dist training
        student_model = nn.parallel.DistributedDataParallel(student_model, device_ids=[cfg.LOCAL_RANK % torch.cuda.device_count()])
        # teacher doesn't need dist train
        teacher_model = DistTeacher(teacher_model)
        teacher_model = nn.parallel.DistributedDataParallel(teacher_model, device_ids=[cfg.LOCAL_RANK % torch.cuda.device_count()])

    student_model.train()
    """
    Notes: we found for pseudo labels, teacher_model.eval() is better; for EMA update and consistency, teacher_model.train() is better
    """
    if cfg.OPTIMIZATION.SEMI_SUP_LEARNING.TEACHER.NUM_ITERS_PER_UPDATE == -1: # for pseudo label
        teacher_model.eval() # Set to eval mode to avoid BN update and dropout
    else: # for EMA teacher with consistency
        teacher_model.train() # Set to train mode
    for t_param in teacher_model.parameters():
        t_param.requires_grad = False

    logger.info(student_model)

    # use unlabeled data as epoch counter
    student_lr_scheduler, student_lr_warmup_scheduler = build_scheduler(
        student_optimizer, total_iters_each_epoch=len(dataloaders['labeled']), total_epochs=cfg.OPTIMIZATION.SEMI_SUP_LEARNING.NUM_EPOCHS,
        last_epoch=last_epoch, optim_cfg=cfg.OPTIMIZATION.SEMI_SUP_LEARNING.STUDENT
    )
    logger.info('**********************Start ssl-training %s/%s(%s)**********************'
                % (cfg.EXP_GROUP_PATH, cfg.TAG, args.extra_tag))

    eval_info = dict(cfg=cfg, dist_test=dist_train, save_to_file=args.save_to_file, logger=logger, result_dir=eval_output_dir)

    train_ssl_model_with_eval(
        teacher_model = teacher_model,
        student_model = student_model,
        student_optimizer = student_optimizer,
        labeled_loader = dataloaders['labeled'],
        unlabeled_loader = dataloaders['unlabeled'],
        test_loader = dataloaders['test'],   # add test loader
        eval_info = eval_info,             # add eval info
        lr_scheduler=student_lr_scheduler,
        ssl_cfg=cfg.OPTIMIZATION.SEMI_SUP_LEARNING,
        start_epoch=start_epoch,
        total_epochs=cfg.OPTIMIZATION.SEMI_SUP_LEARNING.NUM_EPOCHS,
        start_iter=it,
        rank=cfg.LOCAL_RANK,
        tb_log=tb_log,
        ckpt_save_dir=ssl_ckpt_dir,
        labeled_sampler=samplers['labeled'],
        unlabeled_sampler=samplers['unlabeled'],
        lr_warmup_scheduler=student_lr_warmup_scheduler,
        ckpt_save_interval=args.ckpt_save_interval,
        max_ckpt_save_num=args.max_ckpt_save_num,
        merge_all_iters_to_one_epoch=args.merge_all_iters_to_one_epoch,
        dist = dist_train
    )

    logger.info('**********************End ssl-training %s/%s(%s)**********************\n\n\n'
                % (cfg.EXP_GROUP_PATH, cfg.TAG, args.extra_tag))

    # logger.info('**********************Start evaluation for student model %s/%s(%s)**********************' %
    #             (cfg.EXP_GROUP_PATH, cfg.TAG, args.extra_tag))
    # eval_ssl_dir = output_dir / 'eval' / 'eval_with_student_model'
    # eval_ssl_dir.mkdir(parents=True, exist_ok=True)
    # args.start_epoch = cfg.OPTIMIZATION.SEMI_SUP_LEARNING.NUM_EPOCHS - 5
    # repeat_eval_ckpt(
    #     model = student_model.module.onepass if dist_train else student_model,
    #     test_loader = dataloaders['test'],
    #     args = args,
    #     eval_output_dir = eval_ssl_dir,
    #     logger = logger,
    #     ckpt_dir = ssl_ckpt_dir / 'student',
    #     dist_test=dist_train
    # )
    # logger.info('**********************End evaluation for student model %s/%s(%s)**********************' %
    #             (cfg.EXP_GROUP_PATH, cfg.TAG, args.extra_tag))

    logger.info('**********************Start evaluation for teacher model %s/%s(%s)**********************' %
                (cfg.EXP_GROUP_PATH, cfg.TAG, args.extra_tag))
    eval_ssl_dir = output_dir / 'eval' / 'eval_with_teacher_model'
    eval_ssl_dir.mkdir(parents=True, exist_ok=True)
    args.start_epoch = cfg.OPTIMIZATION.SEMI_SUP_LEARNING.NUM_EPOCHS - 5
    if dist_train:
        teacher_model.module.onepass.set_model_type('origin') # ret filtered boxes
    else:
        teacher_model.set_model_type('origin')
    for t_param in teacher_model.parameters(): # Add this to avoid errors
        t_param.requires_grad = True
    repeat_eval_ckpt(
        model = teacher_model.module.onepass if dist_train else teacher_model,
        test_loader = dataloaders['test'],
        args = args,
        eval_output_dir = eval_ssl_dir,
        logger = logger,
        ckpt_dir = ssl_ckpt_dir / 'teacher',
        dist_test=dist_train
    )
    logger.info('**********************End evaluation for teacher model %s/%s(%s)**********************' %
                (cfg.EXP_GROUP_PATH, cfg.TAG, args.extra_tag))

if __name__ == '__main__':
    main()