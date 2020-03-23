#
# Created by ZhangYuyang on 2019/8/9
#
import argparse
import torch
import numpy as np

from basic_parameters import BasicParameters
from utils.trainers import MagicPointSynthetic

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
        parser.add_argument("--synthetic_dataset_dir", type=str, required=True)
        parser.add_argument("--gpus", type=str, default='0')
        parser.add_argument("--lr", type=float, default=0.001)
        parser.add_argument("--batch_size", type=int, default=64)
        parser.add_argument("--num_workers", type=int, default=8)
        parser.add_argument("--prefix", type=str, default='exp1')
        parser.add_argument("--run_mode", type=str, default='train')
        parser.add_argument("--ckpt_file", type=str, default=None)
        return parser.parse_args()


def main():

    params = MagicPointParameters()
    params.initialize()

    magicpoint_synthetic = MagicPointSynthetic(params)
    if params.run_mode == 'train':
        magicpoint_synthetic.train()
    elif params.run_mode == 'test':
        if params.ckpt_file is None:
            params.logger.error("Please input the ckpt_file in bash like: --ckpt_file=/*****")
            return
        else:
            magicpoint_synthetic.test(params.ckpt_file)


if __name__ == '__main__':
    main()

