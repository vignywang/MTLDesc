#
# Created by ZhangYuyang on 2019/10/16
#
import argparse
import torch
import numpy as np

from basic_parameters import BasicParameters
from utils.megpoint_trainers import MagicPointHeatmapTrainer

# make the result reproducible
torch.manual_seed(3928)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(2933)


class MagicPointHeatmapParameters(BasicParameters):

    def __init__(self):
        super(MagicPointHeatmapParameters, self).__init__()
        self.ckpt_root = './magicpoint_ckpt'
        self.log_root = './magicpoint_log'
        self.nms_threshold = 4
        self.epoch_num = 60

    @staticmethod
    def my_parser():
        parser = argparse.ArgumentParser(description="Pytorch Training")
        parser.add_argument("--gpus", type=str, default='0')
        parser.add_argument("--lr", type=float, default=0.001)
        parser.add_argument("--batch_size", type=int, default=32)
        parser.add_argument("--num_workers", type=int, default=8)
        parser.add_argument("--log_freq", type=int, default=100)
        parser.add_argument("--epoch_num", type=int, default=60)
        parser.add_argument("--prefix", type=str, default='exp1')
        parser.add_argument("--run_mode", type=str, default="train")
        parser.add_argument("--ckpt_file", type=str, default="")
        return parser.parse_args()


def main():

    params = MagicPointHeatmapParameters()
    params.initialize()

    magicpoint_trainer = MagicPointHeatmapTrainer(params)

    if params.run_mode == "train":
        magicpoint_trainer.train()

    elif params.run_mode == "test":
        magicpoint_trainer.test(params.ckpt_file)

    else:
        assert False


if __name__ == '__main__':
    main()



