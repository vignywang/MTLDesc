#
# Created by ZhangYuyang on 2020/6/27
#
import os
from pathlib import Path
import argparse

import yaml
import numpy as np
import cv2 as cv
from tqdm import tqdm

from hpatch_related.hpatch_dataset import OrgHPatchDataset
from models import get_model


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--output_root', type=str, required=True)
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.load(f)
    keys = '*' if config['keys'] == '*' else config['keys'].split(',')

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    dataset = OrgHPatchDataset(**config['hpatches'])

    with get_model(config['model']['name'])(**config['model']) as net:
        net.size()
        for i, data in tqdm(enumerate(dataset)):
            data['image'] = cv.resize(data['image'], dsize=(640, 480), interpolation=cv.INTER_AREA)
            net.predict(data['image'], keys=keys)

        info = net.average_inference_time()
        file = open(os.path.join(output_root, config['model']['name']+'_speed.txt'), 'w')
        file.write(info)
        file.close()



