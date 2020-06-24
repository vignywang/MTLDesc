# 
# Created by ZhangYuyang on 2019/10/31
#
import argparse
import torch
import numpy as np

from basic_parameters import BasicParameters
from utils.megpoint_trainers import MegPointHeatmapTrainer
from utils.utils import generate_testing_file


def setup_seed():
    # make the result reproducible
    torch.manual_seed(3928)
    torch.cuda.manual_seed_all(2342)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(2933)


class MegPointHeatmapParameters(BasicParameters):

    def __init__(self):
        super(MegPointHeatmapParameters, self).__init__()
        self.ckpt_root = './megpoint_ckpt'
        self.log_root = './megpoint_log'
        self.ckpt_folder = ""

        self.detection_threshold = 0.9
        self.dataset_dir = None

        self.adjust_lr = "False"
        self.tmp_ckpt_file = ""
        self.extractor_ckpt_file = ""
        self.weight_decay = 1e-4

        self.model_type = "MegPoint"
        self.fix_grid_option = "400"
        self.fix_sample = "False"
        self.rotation_option = "none"

        # homography & photometric relating params using in training
        self.homography_params = {
            'patch_ratio': 0.8,  # 1.0
            'perspective_amplitude_x': 0.3,  # 0.1
            'perspective_amplitude_y': 0.3,  # 0.1
            'scaling_sample_num': 5,
            'scaling_low': 0.8,
            'scaling_up': 2.0,
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
            'gaussian_noise_mean': 0,
            'gaussian_noise_std': 5,
            'speckle_noise_min_prob': 0,
            'speckle_noise_max_prob': 0.0035,
            'brightness_max_abs_change': 15,
            'contrast_min': 0.5,  # 0.7,
            'contrast_max': 1.5,  # 1.3,
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
        parser.add_argument("--gpus", type=str, default='0')
        parser.add_argument("--dataset_dir", type=str, default='')
        parser.add_argument("--hpatch_dataset_dir", type=str, required=True)
        parser.add_argument("--batch_size", type=int, default=8)
        parser.add_argument("--num_workers", type=int, default=8)
        parser.add_argument("--epoch_num", type=int, default=15)
        parser.add_argument("--log_freq", type=int, default=50)
        parser.add_argument("--lr", type=float, default=0.001)
        parser.add_argument("--prefix", type=str, default='exp')
        parser.add_argument("--detection_threshold", type=float, default=0.9)

        parser.add_argument("--height", type=int, default=240)
        parser.add_argument("--width", type=int, default=320)
        parser.add_argument("--run_mode", type=str, default="train")
        parser.add_argument("--ckpt_file", type=str, default="")
        parser.add_argument("--ckpt_folder", type=str, default="")
        parser.add_argument("--tmp_ckpt_file", type=str, default="")
        parser.add_argument("--extractor_ckpt_file", type=str, default="")

        parser.add_argument("--model_type", type=str, default="SuperPointBackbone")
        parser.add_argument("--fix_grid_option", type=str, default="400")
        parser.add_argument("--fix_sample", type=str, default="False")
        parser.add_argument("--rotation_option", type=str, default="none")

        return parser.parse_args()

    def initialize(self):
        super(MegPointHeatmapParameters, self).initialize()
        self.logger.info("------------------------------------------")
        self.logger.info("heatmap important params:")

        self.logger.info("dataset_dir: %s" % self.dataset_dir)

        if self.adjust_lr == "True":
            self.adjust_lr = True
        else:
            self.adjust_lr = False

        if self.do_augmentation == "True":
            self.do_augmentation = True
        else:
            self.do_augmentation = False

        if self.fix_sample == "True":
            self.fix_sample = True
        else:
            self.fix_sample = False

        self.logger.info("adjust_lr: %s" % self.adjust_lr)
        self.logger.info("do_augmentation: %s" % self.do_augmentation)
        self.logger.info("model_type: %s" % self.model_type)
        self.logger.info("fix_grid_option: %s " % self.fix_grid_option)
        self.logger.info("fix_sample: %s" % self.fix_sample)
        self.logger.info("rotation_option: %s" % self.rotation_option)

        self.logger.info("------------------------------------------")


def main():
    setup_seed()
    params = MegPointHeatmapParameters()
    params.initialize()

    # initialize the trainer to train or test
    megpoint_trainer = MegPointHeatmapTrainer(params)

    if params.run_mode == "train":
        megpoint_trainer.train()
    elif params.run_mode == "test":
        megpoint_trainer.test(params.tmp_ckpt_file, params.extractor_ckpt_file)
    elif params.run_mode == "test_folder":
        models = generate_testing_file(params.ckpt_folder, params.extractor_ckpt_file)
        for m in models:
            megpoint_trainer.test(m)


if __name__ == '__main__':
    main()

