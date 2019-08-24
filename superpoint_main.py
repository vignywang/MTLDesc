# 
# Created by ZhangYuyang on 2019/8/23
#


import os
import torch
import numpy as np
import argparse
import glob

from utils.logger import get_logger
from utils.trainers import SuperPointTrainer

# make the result reproducible
torch.manual_seed(3928)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(2933)


class Parameters:

    coco_dataset_dir = '/data/MegPoint/dataset/coco'
    hpatch_dataset_dir = '/data/MegPoint/dataset/hpatch'

    ckpt_root = './magicpoint_ckpt'
    ckpt_dir = ''
    log_root = './magicpoint_log'
    log_dir = ''
    logger = None
    gpus = None

    # common params
    height = 240
    width = 320

    # training params
    lr = 0.001
    descriptor_weight = 0.0001
    batch_size = 64
    epoch_num = 50
    log_freq = 100
    num_workers = 8
    prefix = 'exp1'
    do_augmentation = True

    # testing relating params
    save_threshold_curve = True

    # HPatch validate/test relating params
    detection_threshold = 0.005
    correct_epsilon = 3
    rep_top_k = 300
    desp_top_k = 1000

    # homography & photometric relating params using in training
    homography_params = {
        'patch_ratio': 0.9,  # 0.8,
        'perspective_amplitude_x': 0.1,  # 0.2,
        'perspective_amplitude_y': 0.1,  # 0.2,
        'scaling_sample_num': 5,
        'scaling_amplitude': 0.2,
        'translation_overflow': 0.05,
        'rotation_sample_num': 25,
        'rotation_max_angle': np.pi/3,  # np.pi / 2,
        'do_perspective': True,
        'do_scaling': True,
        'do_rotation': True,
        'do_translation': True,
        'allow_artifacts': True
    }

    photometric_params = {
        'gaussian_noise_mean': 0,  # 10,
        'gaussian_noise_std': 5,
        'speckle_noise_min_prob': 0,
        'speckle_noise_max_prob': 0.0035,
        'brightness_max_abs_change': 15,  # 25,
        'contrast_min': 0.7,  # 0.3,
        'contrast_max': 1.3,  # 1.5,
        'shade_transparency_range': (-0.5, 0.5),  # (-0.5, 0.8),
        'shade_kernel_size_range': (50, 100),
        'shade_nb_ellipese': 20,
        'motion_blur_max_kernel_size': 7,
        'do_gaussian_noise': True,
        'do_speckle_noise': True,
        'do_random_brightness': True,
        'do_random_contrast': True,
        'do_shade': True,
        'do_motion_blur': True
    }


def myparser():
    parser = argparse.ArgumentParser(description="Pytorch SuperPoint Training")
    parser.add_argument("--gpus", type=str, default='0')
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--save_threshold_curve", type=bool, default=True)
    parser.add_argument("--prefix", type=str, default='SuperPoint')
    return parser.parse_args()


args = myparser()
params = Parameters()

# read some params from bash
params.gpus = args.gpus
params.lr = args.lr
params.batch_size = args.batch_size
params.prefix = args.prefix
params.num_workers = args.num_workers
params.save_threshold_curve = args.save_threshold_curve

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
params.logger.info('number worker is %d' % params.num_workers)
params.logger.info('prefix is %s' % params.prefix)


trainer = SuperPointTrainer(params)
trainer.train()

