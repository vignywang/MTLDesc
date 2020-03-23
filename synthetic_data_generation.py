#
# Created by ZhangYuyang on 2020/3/23
#
import os
import argparse

from data_utils.synthetic_generation import drawing_primitives
from data_utils.synthetic_generation import default_config
from data_utils.synthetic_generation import dump_primitive_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True)
    args = parser.parse_args()

    if not os.path.exists(args.data_root):
        os.mkdir(args.data_root)

    for i in drawing_primitives:
        dump_primitive_data(i, default_config, args.data_root)



