#
# Created by ZhangYuyang on 2019/8/9
#
import os
import argparse

from utils.logger import get_logger
from utils.magicpoint_trainer import MagicPointTrainer


class Parameters:

    synthetic_dataset_dir = '/data/MegPoint/dataset/synthetic'

    ckpt_root = './magicpoint_ckpt'
    ckpt_dir = ''
    log_root = './magicpoint_log'
    log_dir = ''
    logger = None
    gpus = None

    lr = 0.001
    batch_size = 32
    epoch_num = 10
    log_freq = 1
    prefix = 'exp1'

    height = 240
    width = 320
    do_augmentation = True
    homography_params = {
        'do_translation': True,
        'do_rotation': True,
        'do_scaling': True,
        'do_perspective': True,
        'scaling_amplitude': 0.2,
        'perspective_amplitude_x': 0.2,
        'perspective_amplitude_y': 0.2,
        'patch_ratio': 0.8,
        'rotation_max_angle': 1.57,  # 3.14
        'allow_artifacts': False,
    }


def myparser():
    parser = argparse.ArgumentParser(description="Pytorch MagicPoint Training")
    parser.add_argument("--gpus", type=str, default='0')
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--prefix", type=str, default='exp1')
    return parser.parse_args()


args = myparser()
params = Parameters()

# read some params from bash
params.gpus = args.gpus
params.lr = args.lr
params.batch_size = args.batch_size
params.prefix = args.prefix

# set and mkdir relative dir when necessary
if not os.path.exists(params.ckpt_root):
    os.mkdir(params.ckpt_root)
if not os.path.exists(params.log_root):
    os.mkdir(params.log_root)
params.ckpt_dir = os.path.join(params.ckpt_root, params.prefix + '_%.4f' % params.lr + '_%d' % params.batch_size)
params.log_dir = os.path.join(params.log_root, params.prefix + '_%.4f' % params.lr + '_%d' % params.batch_size)
if not os.path.exists(params.ckpt_dir):
    os.mkdir(params.ckpt_dir)
if not os.path.exists(params.log_dir):
    os.mkdir(params.log_dir)
params.logger = get_logger(params.log_dir)

# set gpu devices
os.environ['CUDA_VISIBLE_DEVICES'] = params.gpus
params.gpus = [i for i in range(len(params.gpus.split(',')))]
params.logger.info("Set CUDA_VISIBLE_DEVICES to %s" % params.gpus)

# log the parameters
params.logger.info('batch size is %d' % params.batch_size)
params.logger.info('training epoch is %d' % params.epoch_num)
params.logger.info('input size is [%d, %d]' % (params.height, params.width))
params.logger.info('learning rate is %.4f' % params.lr)
params.logger.info('prefix is %s' % params.prefix)

# initialize the trainer and train
magicpoint_trainer = MagicPointTrainer(params)
magicpoint_trainer.train()













