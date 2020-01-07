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
        self.ckpt_root = '/data/MegPoint/megpoint_ckpt'
        self.log_root = './megpoint_log'
        self.ckpt_folder = ""

        self.detection_threshold = 0.9
        self.dataset_dir = None

        self.network_arch = "baseline"
        self.train_mode = "with_gt"
        self.detection_mode = "use_network"
        self.desp_loss = "triplet"
        self.align_weight = 1.0
        self.align_type = "general"
        self.adjust_lr = "False"
        self.point_type = "general"
        self.fn_scale = 1.0
        self.point_gamma = 2.0
        self.homo_weight = 0.1
        self.half_region_size = 7
        self.repro_weight = 0.1
        self.tmp_ckpt_file = ""
        self.dataset_type = "coco"
        self.sample_num = 100

        # debug related
        self.debug_mode = "MegPoint"
        self.debug_ckpt = None

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
        parser.add_argument("--dataset_dir", type=str, default="/data/MegPoint/dataset/coco/train2014/pseudo_image_points_0")
        parser.add_argument("--batch_size", type=int, default=16)
        parser.add_argument("--num_workers", type=int, default=8)
        parser.add_argument("--epoch_num", type=int, default=60)
        parser.add_argument("--log_freq", type=int, default=50)
        parser.add_argument("--lr", type=float, default=0.001)
        parser.add_argument("--prefix", type=str, default='exp')
        parser.add_argument("--detection_threshold", type=float, default=0.9)

        parser.add_argument("--height", type=int, default=240)
        parser.add_argument("--width", type=int, default=320)
        parser.add_argument("--network_arch", type=str, default="baseline")  # 目前为两种 baseline or resnet50
        parser.add_argument("--train_mode", type=str, default="with_gt")  # with_gt or without_gt
        parser.add_argument("--desp_loss", type=str, default="triplet")  # triplet or tuplet
        parser.add_argument("--detection_mode", type=str, default="use_network")  # use_network or use_sift
        parser.add_argument("--run_mode", type=str, default="train")
        parser.add_argument("--ckpt_file", type=str, default="")
        parser.add_argument("--ckpt_folder", type=str, default="")
        parser.add_argument("--homo_pred_mode", type=str, default="RANSAC")
        parser.add_argument("--match_mode", type=str, default="NN")
        parser.add_argument("--align_weight", type=float, default=1.0)
        parser.add_argument("--align_type", type=str, default="general")  # general or weighted
        parser.add_argument("--adjust_lr", type=str, default="False")  # True or False
        parser.add_argument("--point_type", type=str, default="general")  # general or spatial
        parser.add_argument("--fn_scale", type=float, default=1.0)
        parser.add_argument("--point_gamma", type=float, default=2.0)
        parser.add_argument("--homo_weight", type=float, default=0.1)
        parser.add_argument("--repro_weight", type=float, default=0.1)
        parser.add_argument("--half_region_size", type=int, default=7)
        parser.add_argument("--tmp_ckpt_file", type=str, default="")
        parser.add_argument("--dataset_type", type=str, default="coco")
        parser.add_argument("--sample_num", type=int, default=100)

        # debug related
        parser.add_argument("--debug_mode", type=str, default="MegPoint")
        parser.add_argument("--debug_ckpt", type=str, default=None)

        return parser.parse_args()

    def initialize(self):
        super(MegPointHeatmapParameters, self).initialize()
        self.logger.info("------------------------------------------")
        self.logger.info("heatmap important params:")

        self.logger.info("dataset_dir: %s" % self.dataset_dir)
        self.logger.info("train mode: %s" % self.train_mode)
        self.logger.info("network_arch: %s" % self.network_arch)
        self.logger.info("align_weight: %.4f" % self.align_weight)
        self.logger.info("align_type: %s" % self.align_type)
        self.logger.info("point_type: %s" % self.point_type)
        self.logger.info("fn_scale: %.1f" % self.fn_scale)
        self.logger.info("point_gamma: %.1f" % self.point_gamma)
        self.logger.info("homo_weight: %.3f" % self.homo_weight)
        self.logger.info("repro_weight: %.3f" % self.repro_weight)
        self.logger.info("half_region_size: %d" % self.half_region_size)

        if self.adjust_lr == "True":
            self.adjust_lr = True
        else:
            self.adjust_lr = False

        self.logger.info("adjust_lr: %s" % self.adjust_lr)

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
        megpoint_trainer.test(params.tmp_ckpt_file)
    elif params.run_mode == "test_folder":
        models = generate_testing_file(params.ckpt_folder)
        for m in models:
            megpoint_trainer.test(m)
    elif params.run_mode == "test_debug":
        megpoint_trainer.test_debug(params.ckpt_file, params.debug_mode, params.debug_ckpt)

    # /home/zhangyuyang/project/development/MegPoint/megpoint_ckpt/coco_weight_bce_01_0.0010_24/model_59.pt
    # /home/zhangyuyang/project/development/MegPoint/megpoint_ckpt/coco_weight_bce_heatmap_00_0.0010_24/model_99.pt
    # /home/zhangyuyang/project/development/MegPoint/megpoint_ckpt/coco_weight_bce_residual_0.0010_16/model_59.pt
    # /home/zhangyuyang/project/development/MegPoint/megpoint_ckpt/good_results/coco_weight_bce_precise_0.0010_16/model_19.pt

if __name__ == '__main__':
    main()

