#
# Created by ZhangYuyang on 2020/6/27
#
from pathlib import Path
import argparse

import yaml
import numpy as np
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
        for i, data in tqdm(enumerate(dataset), total=len(dataset)):
            predictions = net.predict(data['image'], keys=keys)

            image_name = data['image_name']
            folder_name = data['folder_name']
            output_dir = Path(output_root, folder_name)
            output_dir.mkdir(parents=True, exist_ok=True)

            np.savez(Path(output_dir, image_name), **predictions)



