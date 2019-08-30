# 
# Created by ZhangYuyang on 2019/8/27
#
import os
import argparse
import numpy as np
from utils.logger import get_logger


class BasicParameters(object):

    def __init__(self):
        self.synthetic_dataset_dir = '/data/MegPoint/dataset/synthetic'
        self.coco_dataset_dir = '/data/MegPoint/dataset/coco'
        self.hpatch_dataset_dir = '/data/MegPoint/dataset/hpatch'

        self.ckpt_root = './ckpt'
        self.ckpt_dir = ''
        self.log_root = './log'
        self.log_dir = ''
        self.logger = None
        self.gpus = None

        # common params
        self.height = 240
        self.width = 320
        self.hpatch_height = 480
        self.hpatch_width = 640

        # training params
        self.lr = 0.001
        self.descriptor_weight = 1.0
        self.batch_size = 64
        self.epoch_num = 100
        self.log_freq = 100
        self.num_workers = 8
        self.prefix = 'exp1'
        self.do_augmentation = True

        # testing relating params
        self.save_threshold_curve = True
        self.nms_threshold = 4

        # HPatch validate/test relating params
        self.detection_threshold = 0.005
        self.correct_epsilon = 3
        self.top_k = 1000

        # homography & photometric relating params using in training
        self.homography_params = {
            'patch_ratio': 0.8,
            'perspective_amplitude_x': 0.2,
            'perspective_amplitude_y': 0.2,
            'scaling_sample_num': 5,
            'scaling_amplitude': 0.2,
            'translation_overflow': 0.05,
            'rotation_sample_num': 25,
            'rotation_max_angle': np.pi / 2,
            'do_perspective': True,
            'do_scaling': True,
            'do_rotation': True,
            'do_translation': True,
            'allow_artifacts': True
        }

        self.photometric_params = {
            'gaussian_noise_mean': 10,
            'gaussian_noise_std': 5,
            'speckle_noise_min_prob': 0,
            'speckle_noise_max_prob': 0.0035,
            'brightness_max_abs_change': 25,
            'contrast_min': 0.3,
            'contrast_max': 1.5,
            'shade_transparency_range': (-0.5, 0.8),
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

    def initialize(self):
        # read some params from bash
        args = self.my_parser()
        self.gpus = args.gpus
        self.lr = args.lr
        self.batch_size = args.batch_size
        self.prefix = args.prefix
        self.num_workers = args.num_workers

        # set and mkdir relative dir when necessary
        if not os.path.exists(self.ckpt_root):
            os.mkdir(self.ckpt_root)
        if not os.path.exists(self.log_root):
            os.mkdir(self.log_root)
        self.ckpt_dir = os.path.join(self.ckpt_root,
                                     self.prefix + '_%.4f' % self.lr + '_%d' % self.batch_size)
        self.log_dir = os.path.join(self.log_root, self.prefix + '_%.4f' % self.lr + '_%d' % self.batch_size)
        if not os.path.exists(self.ckpt_dir):
            os.mkdir(self.ckpt_dir)
        if not os.path.exists(self.log_dir):
            os.mkdir(self.log_dir)
        self.logger = get_logger(self.log_dir)

        # set gpu devices
        os.environ['CUDA_VISIBLE_DEVICES'] = self.gpus
        self.gpus = [i for i in range(len(self.gpus.split(',')))]
        self.logger.info("Set CUDA_VISIBLE_DEVICES to %s" % self.gpus)

        # log the parameters
        self.logger.info('batch size is %d' % self.batch_size)
        self.logger.info('training epoch is %d' % self.epoch_num)
        self.logger.info('training input size is [%d, %d]' % (self.height, self.width))
        self.logger.info('non-maximum suppression threshold: %d' % self.nms_threshold)
        self.logger.info('haptch testing input size is [%d, %d]' % (self.hpatch_height, self.hpatch_width))
        self.logger.info('learning rate is %.4f' % self.lr)
        self.logger.info('number worker is %d' % self.num_workers)
        self.logger.info('prefix is %s' % self.prefix)

    @staticmethod
    def my_parser():
        parser = argparse.ArgumentParser(description="Pytorch Training")
        parser.add_argument("--gpus", type=str, default='0')
        parser.add_argument("--lr", type=float, default=0.001)
        parser.add_argument("--batch_size", type=int, default=64)
        parser.add_argument("--num_workers", type=int, default=8)
        parser.add_argument("--prefix", type=str, default='exp1')
        return parser.parse_args()






