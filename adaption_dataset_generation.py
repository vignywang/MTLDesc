#
# Created by ZhangYuyang on 2019/8/19
#
import os
import argparse

import numpy as np
from utils.adaption_maker import AdaptionMaker

# make a reproducible pseudo ground truth
np.random.seed(3242)


def my_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_file", type=str, required=True)
    parser.add_argument("--dataset_type", type=str, required=True)
    parser.add_argument("--dataset_dir", type=str, required=True)
    parser.add_argument("--out_root", type=str, required=True)
    parser.add_argument("--adaption_num", type=int, default=100)
    parser.add_argument("--top_k", type=int, default=300)
    parser.add_argument("--nms_threshold", type=int, default=8)
    # for coco adaption, first round: 0.005, second round: 0.04
    parser.add_argument("--detection_threshold", type=float, default=0.04)

    return parser.parse_args()


if __name__ == "__main__":
    params = my_parser()

    print("ckpt_file: %s" % params.ckpt_file)
    print("dataset_type: %s" % params.dataset_type)
    print("dataset_dir: %s" % params.dataset_dir)
    print("out_root: %s" % params.out_root)
    print("adaption_num: %d" % params.adaption_num)
    print("top_k: %d" % params.top_k)
    print("nms_threshold: %d" % params.nms_threshold)
    print("detection_thresholds: %.4f" % params.detection_threshold)

    if not os.path.exists(params.out_root):
        os.mkdir(params.out_root)
    # use_gpu = False
    use_gpu = True

    adaption_maker = AdaptionMaker(params=params, use_gpu=use_gpu)
    adaption_maker.run()





