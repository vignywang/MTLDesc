#
# Created by ZhangYuyang on 2020/9/14
#
import os
import random
from glob import glob

import torch
import h5py
import numpy as np
import cv2
from tqdm import tqdm

from torch.utils.data import Dataset
from .dataset_tools import ImgAugTransform


class MegaDepthNoOrd(Dataset):

    def __init__(self, **config):
        self.config = {
            'height': 224,
            'width': 224,
            'scales': [0.5, 0.75, 1.0],
            'small': 0,
        }
        self.config.update(config)
        self.photometric = ImgAugTransform()
        self._format_file_list()

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image_dir = os.path.join(self.config['dataset_root'], self.image_list[idx])
        depth_dir = os.path.join(self.config['dataset_root'], self.depth_list[idx])

        image = cv2.imread(image_dir)[:, :, ::-1]
        depth = np.load(depth_dir)

        # preprocess
        image, depth = self.preprocess(image, depth)
        mask = np.where(depth > 0, np.ones_like(depth), np.zeros_like(depth))

        # augmentation
        if self.config['do_augmentation']:
            if torch.rand([]).item() < 0.5:
                image = self.photometric(image)

        # debug use
        # inv_depth = 1. / np.clip(depth, 1e-5, np.inf)
        # inv_depth_max = np.max(inv_depth[mask.astype(np.bool)])
        # inv_depth /= inv_depth_max
        # inv_depth = np.clip(inv_depth * 255., 0, 255)
        # inv_depth = np.where(depth < 1e-4, np.zeros_like(depth), inv_depth)
        # image_depth = np.concatenate((image[:, :, ::-1], np.tile(inv_depth[:, :, np.newaxis], [1, 1, 3])), axis=0)
        # cv2.imwrite('/home/yuyang/tmp/mega_tmp/mega_imageDepth_{}.jpg'.format(idx), image_depth)

        # rescale to [-1, 1]
        image = image.astype(np.float32) * 2. / 255. - 1
        return {
            'image': torch.from_numpy(image.transpose((2, 0, 1))),
            'depth': torch.from_numpy(depth),
            'mask': torch.from_numpy(mask),
        }

    def preprocess(self, image, depth):
        # Scaling
        if self.config['do_scale']:
            h, w = depth.shape
            scale_factor = random.choice(self.config['scales'])
            h, w = (int(h * scale_factor), int(w * scale_factor))
            image = cv2.resize(image, (w, h), interpolation=cv2.INTER_LINEAR)
            depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_NEAREST)

        # Padding to fit for crop_size
        h, w = depth.shape
        pad_h = max(self.config['height'] - h, 0)
        pad_w = max(self.config['width'] - w, 0)
        pad_kwargs = {
            "top": int(pad_h/2),
            "bottom": pad_h-int(pad_h/2),
            "left": int(pad_w/2),
            "right": pad_w-int(pad_w/2),
            "borderType": cv2.BORDER_CONSTANT,
        }
        if pad_h > 0 or pad_w > 0:
            image = cv2.copyMakeBorder(image, value=0, **pad_kwargs)
            depth = cv2.copyMakeBorder(depth, value=0, **pad_kwargs)

        # Cropping
        h, w = depth.shape
        start_h = random.randint(0, h - self.config['height'])
        start_w = random.randint(0, w - self.config['width'])
        end_h = start_h + self.config['height']
        end_w = start_w + self.config['width']
        image = image[start_h:end_h, start_w:end_w]
        depth = depth[start_h:end_h, start_w:end_w]

        # Random flipping
        if random.random() < 0.5:
            image = np.fliplr(image).copy()  # HWC
            depth = np.fliplr(depth).copy()  # HW

        return image, depth

    def _format_file_list(self):
        root = self.config['dataset_root']
        image_list = np.array(sorted(glob(os.path.join(root, '*.jpg')), key=lambda x: int(x.split('/')[-1].split('.')[0])))
        depth_list = np.array(sorted(glob(os.path.join(root, '*.npy')), key=lambda x: int(x.split('/')[-1].split('.')[0])))

        if self.config['small'] > 0:
            np.random.seed(self.config['seed'])
            choices = np.random.choice(range(image_list.shape[0]), size=int(image_list.shape[0]*self.config['small']), replace=False)
            image_list = image_list[choices]
            depth_list = depth_list[choices]
        self.image_list = list(image_list)
        self.depth_list = list(depth_list)


class MegaDepthNoOrdTest(MegaDepthNoOrd):

    def __init__(self, **config):
        super(MegaDepthNoOrdTest, self).__init__(**config)
        if 'output_type' not in self.config:
            self.config['output_type'] = 0

    def __getitem__(self, idx):
        image_dir = os.path.join(self.config['dataset_root'], self.image_list[idx])
        depth_dir = os.path.join(self.config['dataset_root'], self.depth_list[idx])

        image = cv2.imread(image_dir)
        depth = np.load(depth_dir)

        # preprocess
        image, depth = self.preprocess(image, depth)
        color_image = image.copy()
        image = image[:, :, ::-1].copy()
        mask = np.where(depth > 0, np.ones_like(depth), np.zeros_like(depth))

        if self.config['output_type'] == 0:
            image = image.astype(np.float32) * 2. / 255. - 1.
        else:
            image = image.astype(np.float32) / 255.
        return {
            'color_image': color_image,
            'image': image,
            'depth': depth,
            'mask': mask,
        }

    def preprocess(self, image, depth):
        # Padding to fit for crop_size
        h, w = depth.shape
        pad_h = max(self.config['height'] - h, 0)
        pad_w = max(self.config['width'] - w, 0)
        pad_kwargs = {
            "top": int(pad_h/2),
            "bottom": pad_h-int(pad_h/2),
            "left": int(pad_w/2),
            "right": pad_w-int(pad_w/2),
            "borderType": cv2.BORDER_CONSTANT,
        }
        if pad_h > 0 or pad_w > 0:
            image = cv2.copyMakeBorder(image, value=0, **pad_kwargs)
            depth = cv2.copyMakeBorder(depth, value=0, **pad_kwargs)

        # Center Crop
        h, w = depth.shape
        start_h = int((h - self.config['height'])/2)
        start_w = int((w - self.config['width'])/2)
        end_h = start_h + self.config['height']
        end_w = start_w + self.config['width']
        image = image[start_h:end_h, start_w:end_w]
        depth = depth[start_h:end_h, start_w:end_w]

        return image, depth


class MegaDepthRaw(object):

    def __init__(self, **config):
        self.config = config

    def run(self, *args, **kwargs):
        self._select()
        self._scale()

    def _select(self):
        print('select...')
        select_image_dir = os.path.join(self.config['output_root'], 'image_list_no_ord')
        select_depth_dir = os.path.join(self.config['output_root'], 'depth_list_no_ord')
        if os.path.isfile(select_image_dir + '.npy') and os.path.isfile(select_depth_dir + '.npy'):
            self.select_image_list = np.load(select_image_dir + '.npy')
            self.select_depth_list = np.load(select_depth_dir + '.npy')
            return

        if not os.path.exists(self.config['output_root']):
            os.mkdir(self.config['output_root'])
        self.select_image_list = []
        self.select_depth_list = []
        image_list, depth_list = self._format_file_list()
        for idx in tqdm(range(len(image_list))):
            depth_dir = os.path.join(self.config['dataset_root'], depth_list[idx])
            depth_file = h5py.File(depth_dir, 'r')
            depth = np.array(depth_file['depth'])
            depth_file.close()
            if depth.min() == -1:
                continue

            self.select_image_list.append(image_list[idx])
            self.select_depth_list.append(depth_list[idx])

        output_root = self.config['output_root']
        if not os.path.exists(output_root):
            os.mkdir(output_root)

        np.save(os.path.join(output_root, 'image_list_no_ord'), np.array(self.select_image_list))
        np.save(os.path.join(output_root, 'depth_list_no_ord'), np.array(self.select_depth_list))

    def _scale(self):
        print('scale...')
        for i, dirs in tqdm(
                enumerate(zip(self.select_image_list, self.select_depth_list)), total=len(self.select_image_list)):
            image_dir = os.path.join(self.config['dataset_root'], dirs[0])
            depth_dir = os.path.join(self.config['dataset_root'], dirs[1])

            image = cv2.imread(image_dir)
            depth_file = h5py.File(depth_dir, 'r')
            depth = np.array(depth_file['depth'])
            depth_file.close()

            h, w = depth.shape
            min_size = min(h, w)
            scale_factor = self.config['min_size'] / float(min_size)
            h, w = (int(h * scale_factor), int(w * scale_factor))
            image = cv2.resize(image, (w, h), interpolation=cv2.INTER_LINEAR)
            depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_NEAREST)

            if not os.path.exists(self.config['processed_output_root']):
                os.mkdir(self.config['processed_output_root'])
            cv2.imwrite(os.path.join(self.config['processed_output_root'], '%06d.jpg' % i), image)
            np.save(os.path.join(self.config['processed_output_root'], '%06d' % i), depth)

    def _format_file_list(self):
        list_root = self.config['list_root']

        def _format_image_target_list(anotation):
            image_list = np.load(os.path.join(list_root, anotation, 'imgs_MD.npy'), allow_pickle=True)
            depth_list = np.load(os.path.join(list_root, anotation, 'targets_MD.npy'), allow_pickle=True)
            return image_list, depth_list

        image_list = []
        depth_list = []
        for anotation in self.config['anotations']:
            res = _format_image_target_list(anotation)
            image_list += res[0]
            depth_list += res[1]

        return image_list, depth_list


if __name__ == '__main__':
    pass
    # config = {
    #     'list_root': '/data/localization/MegaDepthOrder/list/train_list',
    #     'dataset_root': '/data/localization/MegaDepthOrder/preprocessed_no_ord',
    # }

    # dataset = MegaDepthNoOrd(**config)
    # for i, data in enumerate(dataset):
    #     a = data
    #     if i == 20:
    #         break



