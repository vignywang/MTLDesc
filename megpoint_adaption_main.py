# 
# Created by ZhangYuyang on 2019/9/18
#
import argparse
import torch
import numpy as np

from basic_parameters import BasicParameters
from utils.megpoint_trainers import MegPointAdaptionTrainer


def setup_seed():
    # make the result reproducible
    torch.manual_seed(3928)
    torch.cuda.manual_seed_all(2342)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(2933)


class MegPointAdaptionParameters(BasicParameters):

    def __init__(self):
        super(MegPointAdaptionParameters, self).__init__()
        self.ckpt_root = './megpoint_ckpt'
        self.log_root = './megpoint_log'
        self.magicpoint_ckpt_file = "/home/zhangyuyang/project/development/MegPoint/magicpoint_ckpt/good_results/synthetic_new_0.0010_64/model_59.pt"
        self.coco_pseudo_idx = '0'

        # 采样检测相关
        self.do_augmentation = False
        self.sample_num = 10
        self.detection_threshold = 0.005
        self.train_top_k = 150

        # homography & photometric relating params using in training
        self.homography_params = {
            'patch_ratio': 0.8,  # 0.8,
            'perspective_amplitude_x': 0.2,  # 0.2,
            'perspective_amplitude_y': 0.2,  # 0.2,
            'scaling_sample_num': 5,
            'scaling_amplitude': 0.2,
            'translation_overflow': 0.05,
            'rotation_sample_num': 25,
            'rotation_max_angle': np.pi / 2,  # np.pi / 2,
            'do_perspective': True,
            'do_scaling': True,
            'do_rotation': True,
            'do_translation': True,
            'allow_artifacts': True
        }

        self.aug_homography_params = {
            'patch_ratio': 0.8,  # 0.8,  # 0.9,
            'perspective_amplitude_x': 0.3,  # 0.2,  # 0.1,
            'perspective_amplitude_y': 0.3,  # 0.2,  # 0.1,
            'scaling_sample_num': 5,
            'scaling_amplitude': 0.2,
            'translation_overflow': 0.05,
            'rotation_sample_num': 25,
            'rotation_max_angle': np.pi/3,  # np.pi/2.,  # np.pi/3,
            'do_perspective': True,
            'do_scaling': True,
            'do_rotation': True,
            'do_translation': True,
            'allow_artifacts': True
        }

        self.photometric_params = {
            'gaussian_noise_mean': 0,  # 10,
            'gaussian_noise_std': 5,
            'speckle_noise_min_prob': 0,
            'speckle_noise_max_prob': 0.0035,
            'brightness_max_abs_change': 25,  # 25,
            'contrast_min': 0.5,  # 0.3,
            'contrast_max': 1.5,  # 1.5,
            'shade_transparency_range': (-0.5, 0.5),  # (-0.5, 0.8),
            'shade_kernel_size_range': (100, 150),  # (50, 100),
            'shade_nb_ellipese': 20,
            'motion_blur_max_kernel_size': 7,
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
        parser.add_argument("--gpus", type=str, default='0')
        parser.add_argument("--batch_size", type=int, default=32)
        parser.add_argument("--num_workers", type=int, default=4)
        parser.add_argument("--log_freq", type=int, default=100)
        parser.add_argument("--lr", type=float, default=0.001)
        parser.add_argument("--prefix", type=str, default='adaption')
        parser.add_argument("--coco_pseudo_idx", type=str, default='0')
        return parser.parse_args()


def main():
    setup_seed()
    params = MegPointAdaptionParameters()
    params.initialize()

    # initialize the trainer and train
    megpoint_trainer = MegPointAdaptionTrainer(params)
    megpoint_trainer.train()


if __name__ == '__main__':
    main()



