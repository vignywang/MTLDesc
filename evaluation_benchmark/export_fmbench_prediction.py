#
# Created by ZhangYuyang on 2020/6/27
#
import argparse

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

    dump_name = config['dataset']['dump_name']

    with get_model(config['model']['name'])(**config['model']) as net:
        try:
            for i in tqdm(range(data_length)):
                data = next(test_set)
                predictions = net.predict(data['image'], keys=keys)

                dump_data = {}
                dump_data['dump_data'] = (
                    predictions['descriptors'],
                    predictions['keypoints'],
                )
                dump_data['image_path'] = data['image_path']
                dump_data['dump_path'] = data['dump_path']
                dump_data['dump_name'] = dump_name
                dataset.format_data(dump_data)
        except dataset.end_set:
            pass


