#
# Created by ZhangYuyang on 2019/8/19
#
import yaml
import argparse

import numpy as np
from labelers import get_labeler

# make a reproducible pseudo ground truth
np.random.seed(3242)


def my_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)

    return parser.parse_args()


if __name__ == "__main__":
    params = my_parser()
    # read configs
    with open(params.config, 'r') as f:
        config = yaml.load(f)

    with get_labeler(config['labeler'])(**config) as labeler:
        labeler.build_dataset()




