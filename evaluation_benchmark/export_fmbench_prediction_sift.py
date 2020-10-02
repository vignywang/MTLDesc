#
# Created by ZhangYuyang on 2020/6/27
#
import os
import struct
import argparse

import yaml
import numpy as np
import tensorflow as tf
from tqdm import tqdm

from datasets import get_dataset
from models import get_model


def decode_keypoints(keypoints_bin):
    with open(keypoints_bin, 'rb') as f:
        shape = []
        for i in range(2):
            shape.append(struct.unpack('i', f.read(4))[0])

        keypoints = []
        for i in range(shape[0]):
            x = struct.unpack('f', f.read(4))[0]
            y = struct.unpack('f', f.read(4))[0]
            _ = struct.unpack('f', f.read(4))[0]
            _ = struct.unpack('f', f.read(4))[0]
            keypoints.append([x, y])

        keypoints = np.stack(keypoints)

    return keypoints


def decode_descriptors(descriptor_bin):
    with open(descriptor_bin, 'rb') as f:
        shape = []
        for i in range(2):
            shape.append(struct.unpack('i', f.read(4))[0])

        descriptors = []
        for i in range(shape[0]):
            cur_desc = []
            for j in range(shape[1]):
                cur_desc.append(struct.unpack('f', f.read(4))[0])
            descriptors.append(cur_desc)
        descriptors = np.stack(descriptors)

    return descriptors


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

    dump_name = config['dataset']['dump_name']

    try:
        for i in tqdm(range(data_length)):
            data = next(test_set)

            image_rel_path = tf.compat.as_str_any(data['image_path'])[len(config['dataset']['data_root']) + 1:]
            feature_path = os.path.join(config['dataset']['feature_root'], image_rel_path.strip('.jpg'))

            keypoints_bin = feature_path + '_keypoints.bin'
            descriptors_bin = feature_path + '_descriptor.bin'

            keypoints = decode_keypoints(keypoints_bin)
            descriptors = decode_descriptors(descriptors_bin)

            dump_data = {}
            dump_data['dump_data'] = (
                descriptors,
                keypoints,
            )
            dump_data['image_path'] = data['image_path']
            dump_data['dump_path'] = data['dump_path']
            dump_data['dump_name'] = dump_name
            dataset.format_data(dump_data)
    except dataset.end_set:
        pass


