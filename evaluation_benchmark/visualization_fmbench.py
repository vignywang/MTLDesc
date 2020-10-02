#
# Created by ZhangYuyang on 2020/7/12
#
import os
import argparse
from pathlib import Path

import yaml
from tqdm import tqdm
import cv2 as cv

from datasets import get_sub_dataset
from models import get_model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--configs', type=str, required=True)
    parser.add_argument('--output_path', type=str, required=True)
    args = parser.parse_args()

    with open(args.configs, 'r') as cf:
        config = yaml.load(cf)
    keys = '*' if config['keys'] == '*' else config['keys'].split(',')

    dataset = get_sub_dataset(config['dataset']['data_name'])(**config['dataset'])
    test_set = dataset.get_test_set()
    data_length = dataset.data_length

    output_path = Path(args.output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    with get_model(config['model']['name'])(**config['model']) as net:
        for i in tqdm(range(data_length)):
            data = next(test_set)
            if not i % 5 == 0:
                continue
            image = data['image']

            predictions = net.predict(data['image'], keys=keys)
            keypoints = predictions['keypoints']
            keypoints = [(int(keypoints[k, 0]), int(keypoints[k, 1])) for k in range(keypoints.shape[0])]

            point_size = 1
            point_color = (0, 255, 0)
            thickness = 2
            for point in keypoints:
                cv.circle(image, point, point_size, point_color, thickness)

            cv.imwrite(str(Path(output_path, "%04d.jpg" % i)), image[:, :, ::-1])

