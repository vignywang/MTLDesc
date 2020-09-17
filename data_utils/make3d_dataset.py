#
# Created by ZhangYuyang on 2020/9/14
#
import os
import numpy as np
import cv2


class Make3dDataset(object):

    def __init__(self, **config):
        self.config = config
        if 'output_type' not in self.config:
            self.config['output_type'] = 0
        # load file list
        self.format_file_list()

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image_file = self.image_list[idx]
        depth_file = self.depth_list[idx]

        color_image = cv2.imread(image_file)
        image = color_image[:, :, ::-1].copy()
        depth = np.load(depth_file)

        # debug use
        # inv_depth = 1. / np.clip(depth, 1e-5, np.inf)
        # inv_depth_max = np.max(inv_depth[mask.astype(np.bool)])
        # inv_depth /= inv_depth_max
        # inv_depth = np.clip(inv_depth * 255., 0, 255)
        # inv_depth = np.where(depth < 1e-4, np.zeros_like(depth), inv_depth)
        # image_depth = np.concatenate((image[:, :, ::-1], np.tile(inv_depth[:, :, np.newaxis], [1, 1, 3])), axis=0)
        # cv2.imwrite('/home/yuyang/tmp/make3d_tmp/make3d_imageDepth_{}.jpg'.format(idx), image_depth)

        if self.config['output_type'] == 0:
            image = image.astype(np.float32) * 2. / 255. - 1.
        else:
            image = image.astype(np.float32) / 255.
        return {
            'color_image': color_image,
            'image': image,
            'depth': depth,
        }

    def format_file_list(self):
        with open(os.path.join(self.config['root'], 'make3d.txt'), 'r') as f:
            lines = f.readlines()
        self.image_list = [os.path.join(self.config['root'], line[:-1] + '.jpg') for line in lines]
        self.depth_list = [os.path.join(self.config['root'], line[:-1] + '.npy') for line in lines]


if __name__ == '__main__':
    config = {
        'root' : '/home/yuyang/data/make3d/processed_dataset',
    }
    dataset = Make3dDataset(**config)

    for i, data in enumerate(dataset):
        a = data
        if i == 3:
            break



