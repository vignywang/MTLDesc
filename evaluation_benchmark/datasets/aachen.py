#
# Created by ZhangYuyang on 2020/7/11
#
import os

import cv2 as cv


class AachenVisualizationDataset(object):

    def __init__(self, **config):
        self.data_list = self._format_file_list(config['dataset_root'], config['pair_list'])

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        data = self.data_list[idx]
        query = cv.imread(data['query'])[:, :, ::-1].copy()  # read as rgb
        db = cv.imread(data['db'])[:, :, ::-1].copy()

        return {
            'image1': query,
            'image2': db,
        }

    @staticmethod
    def _format_file_list(dataset_root, pair_list):
        with open(pair_list, 'r') as pf:
            pair_list = pf.readlines()
            data_list = []
            for pair in pair_list:
                pair = pair.strip('\n')
                query, db = pair.split(' ')
                data_list.append(
                    {
                        'query': os.path.join(dataset_root, query),
                        'db': os.path.join(dataset_root, db),
                    }
                )

        return data_list



