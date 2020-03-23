#
# Created by ZhangYuyang on 2019/8/19
#
import argparse

import numpy as np
from utils.adaption_maker import AdaptionMaker

# make a reproducible pseudo ground truth
np.random.seed(3242)


def my_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, default=0)
    parser.add_argument("--first_ckpt_file", type=str, default="")
    parser.add_argument("--second_ckpt_file", type=str, default="")
    parser.add_argument("--coco_dataset_dir", type=str, required=True)
    parser.add_argument("--height", type=int, default=240)
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--adaption_num", type=int, default=100)
    parser.add_argument("--top_k", type=int, default=300)
    parser.add_argument("--nms_threshold", type=int, default=8)
    parser.add_argument("--first_threshold", type=float, default=0.005)
    parser.add_argument("--second_threshold", type=float, default=0.04)

    return parser.parse_args()


if __name__ == "__main__":
    params = my_parser()
    if params.rounds == 0:
        if params.first_ckpt_file == "":
            print("Must have first_ckpt_file!")
            assert False
        params.ckpt_file = params.first_ckpt_file
        params.detection_threshold = params.first_threshold
    elif params.rounds == 1:
        if params.second_ckpt_file == "":
            print("Must have second_ckpt_file!")
            assert False
        params.ckpt_file = params.second_ckpt_file
        params.detection_threshold = params.second_threshold
    else:
        print("Only support rounds of 0 or 1, current is %d" % params.rounds)
        assert False
    params.out_root = params.coco_dataset_dir

    print("rounds: %d" % params.rounds)
    print("thresholds: %.4f" % params.detection_threshold)

    adaption_maker = AdaptionMaker(params=params)
    adaption_maker.run()





