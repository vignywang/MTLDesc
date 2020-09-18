#
# Created by ZhangYuyang on 2020/9/18
#
# used to create training dataset or testing dataset to train depth model
import argparse
import yaml

from data_utils.megadepth import MegaDepthRaw


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.load(f)

    dataset_raw = MegaDepthRaw(**config)
    dataset_raw.run()


