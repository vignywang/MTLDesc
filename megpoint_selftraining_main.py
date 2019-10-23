#
# Created by ZhangYuyang on 2019/10/13
#
import argparse
import torch
import numpy as np

from basic_parameters import BasicParameters
from utils.megpoint_trainers import MegPointSeflTrainingTrainer


def setup_seed():
    # make the result reproducible
    torch.manual_seed(3928)
    torch.cuda.manual_seed_all(2342)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(2933)


class MegPointSelfTrainingParameters(BasicParameters):

    def __init__(self):
        super(MegPointSelfTrainingParameters, self).__init__()
        self.ckpt_root = './megpoint_ckpt'
        self.log_root = './megpoint_log'
        # self.magicpoint_ckpt = "/home/zhangyuyang/project/development/MegPoint/megpoint_ckpt/good_results/only_detector_re_50_0.0010_48/model_49.pt"
        self.magicpoint_ckpt = "/home/zhangyuyang/project/development/MegPoint/magicpoint_ckpt/good_results/synthetic_new_0.0010_64/model_59.pt"
        # self.magicpoint_ckpt = "/home/zhangyuyang/project/development/MegPoint/magicpoint_ckpt/good_results/synthetic_ed_0.0010_32/model_09.pt"

        # training相关
        self.round_num = 50
        self.epoch_each_round = 5  # 2
        self.raw_batch_size = 32
        self.initial_region_portion = 0.5  # 阈值排序为前50%的区域会被标注为有效区域
        self.initial_point_portion = 0.05
        self.max_region_portion = 0.6
        self.max_point_portion = 0.08
        self.only_target_dataset = False
        self.only_detector = False
        self.reinitialize_each_round = True
        # self.initial_portion = 0.002  # 每一轮按1%递增，最高0.005
        self.src_detector_weight = 0.5
        self.tgt_detector_weight = 1 - self.src_detector_weight

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

        parser.add_argument("--round_num", type=int, default=100)
        parser.add_argument("--epoch_each_round", type=int, default=5)
        parser.add_argument("--only_detector", type=int, default=0)
        parser.add_argument("--only_target_dataset", type=int, default=0)
        parser.add_argument("--reinitialize_each_round", type=int, default=0)
        parser.add_argument("--initial_region_portion", type=float, default=0.5)
        parser.add_argument("--initial_point_portion", type=float, default=0.2)
        parser.add_argument("--max_region_portion", type=float, default=0.8)
        parser.add_argument("--max_point_portion", type=float, default=0.5)
        return parser.parse_args()

    def initialize(self):
        super(MegPointSelfTrainingParameters, self).initialize()
        self.logger.info("------------------------------------------")
        self.logger.info("self-training important params:")

        self.only_target_dataset = bool(self.only_target_dataset)
        self.only_detector = bool(self.only_detector)
        self.reinitialize_each_round = bool(self.reinitialize_each_round)

        self.logger.info("round_num: %d" % self.round_num)
        self.logger.info("epoch_each_round: %d" % self.epoch_each_round)
        self.logger.info("only_target_detector: %s" % self.only_target_dataset)
        self.logger.info("only_detector: %s" % self.only_detector)
        self.logger.info("reinitialize_each_round: %s" % self.reinitialize_each_round)
        self.logger.info("epoch_each_round: %d" % self.epoch_each_round)
        self.logger.info(
            "src_detector_weight=%.2f, tgt_detector_weight=%.2f" % (
                self.src_detector_weight,
                self.tgt_detector_weight
            )
        )
        self.logger.info("------------------------------------------")


def main():
    setup_seed()
    params = MegPointSelfTrainingParameters()
    params.initialize()

    # initialize the trainer and train
    megpoint_trainer = MegPointSeflTrainingTrainer(params)
    megpoint_trainer.train()


if __name__ == '__main__':
    main()


