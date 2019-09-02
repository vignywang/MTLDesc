#
# Created by ZhangYuyang on 2019/8/9
#
import os
import argparse
import torch
import numpy as np
import glob

from basic_parameters import BasicParameters
from utils.testers import MagicPointSyntheticTester
from utils.testers import HPatchTester

# make the result reproducible
torch.manual_seed(3928)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(2933)


class MagicPointParameters(BasicParameters):

    def __init__(self):
        super(MagicPointParameters, self).__init__()
        self.ckpt_root = './magicpoint_ckpt'
        self.log_root = './magicpoint_log'
        self.nms_threshold = 4
        self.epoch_num = 60

    @staticmethod
    def my_parser():
        parser = argparse.ArgumentParser(description="Pytorch Training")
        parser.add_argument("--gpus", type=str, default='0')
        parser.add_argument("--lr", type=float, default=0.001)
        parser.add_argument("--batch_size", type=int, default=64)
        parser.add_argument("--num_workers", type=int, default=8)
        parser.add_argument("--prefix", type=str, default='exp1')
        return parser.parse_args()


if __name__ == '__main__':
    params = MagicPointParameters()
    params.initialize()

    # initialize the trainer and train
    # magicpoint_trainer = MagicPointSyntheticTrainer(params)
    # magicpoint_trainer.train()

    # initialize the tester and test all checkpoint file in the folder
    magicpoint_synthetic_tester = MagicPointSyntheticTester(params)
    magicpoint_hpatch_tester = HPatchTester(params)

    # # choose test mode
    # mode = 'all'
    # mode = 'only_hpatch'
    mode = 'only_synthetic'
    # mode = 'only_synthetic_one_image'

    # ckpt_file = '/home/zhangyuyang/project/development/MegPoint/magicpoint_ckpt/good_results/adam_adaption_0.0010_64/model_99.pt'
    # ckpt_file = '/home/zhangyuyang/project/development/MegPoint/magicpoint_ckpt/good_results/adam_0.0010_64/model_59.pt'
    ckpt_file = '/home/zhangyuyang/project/development/MegPoint/magicpoint_ckpt/good_results/adaption_0/model_99.pt'
    # ckpt_file = '/home/zhangyuyang/project/development/MegPoint/magicpoint_ckpt/good_results/superpoint_magicleap.pth'

    if mode == 'all':
        ckpt_files = glob.glob(os.path.join(params.ckpt_dir, "model_*"))
        ckpt_files = sorted(ckpt_files)
        for ckpt_file in ckpt_files:
            magicpoint_synthetic_tester.test(ckpt_file)
            magicpoint_hpatch_tester.test_model(ckpt_file)

    elif mode == 'only_hpatch':
        # magicpoint_hpatch_tester.test_FAST_repeatability()
        magicpoint_hpatch_tester.test_model(ckpt_file)
        # magicpoint_hpatch_tester.test_orb_descriptors()

    elif mode == 'only_synthetic':
        magicpoint_synthetic_tester.test(ckpt_file)

    elif mode == 'only_synthetic_one_image':
        image_dir = '/data/MegPoint/dataset/synthetic/draw_multiple_polygons/images/test/76.png'
        magicpoint_synthetic_tester.test_single_image(ckpt_file, image_dir)







