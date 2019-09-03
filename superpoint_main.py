# 
# Created by ZhangYuyang on 2019/8/23
#
import torch
import argparse
import numpy as np

from basic_parameters import BasicParameters
from utils.trainers import SuperPointTrainer

# make the result reproducible
torch.manual_seed(3928)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(2933)


class SuperPointParameters(BasicParameters):

    def __init__(self):
        super(SuperPointParameters, self).__init__()

        self.ckpt_root = './superpoint_ckpt'
        self.log_root = './superpoint_log'
        self.coco_pseudo_idx = '0'
        self.loss_type = 'triplet'  # 'pairwise'

        # homography & photometric relating params using in training
        self.homography_params = {
            'patch_ratio': 0.9,
            'perspective_amplitude_x': 0.1,
            'perspective_amplitude_y': 0.1,
            'scaling_sample_num': 5,
            'scaling_amplitude': 0.2,
            'translation_overflow': 0.05,
            'rotation_sample_num': 25,
            'rotation_max_angle': np.pi/3,
            'do_perspective': True,
            'do_scaling': True,
            'do_rotation': True,
            'do_translation': True,
            'allow_artifacts': True
        }

        self.photometric_params = {
            'gaussian_noise_mean': 0,
            'gaussian_noise_std': 5,
            'speckle_noise_min_prob': 0,
            'speckle_noise_max_prob': 0.0035,
            'brightness_max_abs_change': 15,
            'contrast_min': 0.7,
            'contrast_max': 1.3,
            'shade_transparency_range': (-0.5, 0.5),
            'shade_kernel_size_range': (100, 150),
            'shade_nb_ellipese': 20,
            'motion_blur_max_kernel_size': 3,
            'do_gaussian_noise': True,
            'do_speckle_noise': True,
            'do_random_brightness': True,
            'do_random_contrast': True,
            'do_shade': True,
            'do_motion_blur': True
        }

    @staticmethod
    def my_parser():
        parser = argparse.ArgumentParser(description="Pytorch Training")
        parser.add_argument("--lr", type=float, default=0.001)
        parser.add_argument("--batch_size", type=int, default=32)
        parser.add_argument("--gpus", type=str, default='0')
        parser.add_argument("--prefix", type=str, default='superpoint')
        parser.add_argument("--num_workers", type=int, default=8)
        parser.add_argument("--coco_pseudo_idx", type=str, default='0')
        parser.add_argument("--loss_type", type=str, default='triplet')

        return parser.parse_args()


if __name__ == '__main__':
    params = SuperPointParameters()
    params.initialize()

    trainer = SuperPointTrainer(params)
    trainer.train()

