#
# Created by ZhangYuyang on 2020/6/27
#
import argparse

import tensorflow as tf
import yaml
from tqdm import tqdm

from datasets import get_dataset
from models import get_model


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.load(f)
    keys = '*' if config['keys'] == '*' else config['keys'].split(',')

    dataset = get_dataset(config['dataset']['data_name'])(**config['dataset'])
    test_set = dataset.get_test_set()
    data_length = dataset.data_length

    fmbench_list_path = '/home/yuyang/project/development/github/hand-craft/scripts/fmbench_list.txt'

    with open(fmbench_list_path, 'w') as wf:
        try:
            for i in tqdm(range(data_length)):
                data = next(test_set)
                image_path = tf.compat.as_str_any(data['image_path'])[len(config['dataset']['data_root'])+1:]
                wf.write(image_path + '\n')

        except dataset.end_set:
            pass


