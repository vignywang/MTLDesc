# Original Author:   Kazuto Nakashima
# URL:      http://kazuto1011.github.io
# Created:  2017-10-30
#
# Created by ZhangYuyang on 2020/8/31
#
import os.path as osp
import random

import cv2
import scipy.io as sio
import numpy as np
from PIL import Image
from torch.utils.data import Dataset


class _BaseDataset(Dataset):
    """
    Base dataset class
    """

    def __init__(self, **config):
        self.config = {
            'root': None,
            'split': 'train',
            'ignore_label': 255,
            'mean_rgb': [122.675, 116.669, 104.008],
            'augment': True,
            'flip': True,
            'height': 240,
            'width': 320,
            'scales': [1.0],

        }
        self.config.update(config)
        self.files = []
        self._set_files()

        cv2.setNumThreads(0)

    def _set_files(self):
        """
        Create a file path/image id list.
        """
        raise NotImplementedError()

    def _load_data(self, image_id):
        """
        Load the image and label in numpy.ndarray
        """
        raise NotImplementedError()

    def _augmentation(self, image, label):
        # Scaling
        h, w = label.shape

        # if self.base_size:
        #     if h > w:
        #         h, w = (self.base_size, int(self.base_size * w / h))
        #     else:
        #         h, w = (int(self.base_size * h / w), self.base_size)

        scale_factor = random.choice(self.config['scales'])
        h, w = (int(h * scale_factor), int(w * scale_factor))
        image = cv2.resize(image, (w, h), interpolation=cv2.INTER_LINEAR)
        label = Image.fromarray(label).resize((w, h), resample=Image.NEAREST)
        label = np.asarray(label, dtype=np.int64)

        # Padding to fit for crop_size
        h, w = label.shape
        pad_h = max(self.config['height'] - h, 0)
        pad_w = max(self.config['width'] - w, 0)
        pad_kwargs = {
            "top": 0,
            "bottom": pad_h,
            "left": 0,
            "right": pad_w,
            "borderType": cv2.BORDER_CONSTANT,
        }
        if pad_h > 0 or pad_w > 0:
            image = cv2.copyMakeBorder(image, value=self.config['mean_rgb'], **pad_kwargs)
            label = cv2.copyMakeBorder(label, value=self.config['ignore_label'], **pad_kwargs)

        # Cropping
        h, w = label.shape
        start_h = random.randint(0, h - self.config['height'])
        start_w = random.randint(0, w - self.config['width'])
        end_h = start_h + self.config['height']
        end_w = start_w + self.config['width']
        image = image[start_h:end_h, start_w:end_w]
        label = label[start_h:end_h, start_w:end_w]

        if self.config['flip']:
            # Random flipping
            if random.random() < 0.5:
                image = np.fliplr(image).copy()  # HWC
                label = np.fliplr(label).copy()  # HW
        return image, label

    def __getitem__(self, index):
        image_id, image, label = self._load_data(index)
        if self.config['augment']:
            image, label = self._augmentation(image, label)
        # Mean subtraction
        image -= self.config['mean_rgb']
        # HWC -> CHW
        image = image.transpose(2, 0, 1)
        return {
            'image_id': image_id,
            'image': image.astype(np.float32),
            'label': label.astype(np.int64),
        }

    def __len__(self):
        return len(self.files)

    def __repr__(self):
        fmt_str = "Dataset: " + self.__class__.__name__ + "\n"
        fmt_str += "    # data: {}\n".format(self.__len__())
        fmt_str += "    Split: {}\n".format(self.config['split'])
        fmt_str += "    Root: {}".format(self.config['root'])
        return fmt_str


class CocoStuff10k(_BaseDataset):
    """COCO-Stuff 10k dataset"""

    def __init__(self,  **config):
        # self.warp_image = warp_image
        super(CocoStuff10k, self).__init__(**config)

    def _set_files(self):
        # Create data list via {train, test, all}.txt
        if self.config['split'] in ["train", "test", "all"]:
            file_list = osp.join(self.config['root'], "imageLists", self.config['split'] + ".txt")
            file_list = tuple(open(file_list, "r"))
            file_list = [id_.rstrip() for id_ in file_list]
            self.files = file_list
        else:
            raise ValueError("Invalid split name: {}".format(self.config['split']))

    def _load_data(self, index):
        # Set paths
        image_id = self.files[index]
        image_path = osp.join(self.config['root'], "images", image_id + ".jpg")
        label_path = osp.join(self.config['root'], "annotations", image_id + ".mat")

        # Load an image and label
        # change bgt to rgb
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)[:, :, ::-1].astype(np.float32)
        label = sio.loadmat(label_path)["S"]
        label -= 1  # unlabeled (0 -> -1)
        label[label == -1] = 255

        # Warping: this is just for reproducing the official scores on GitHub
        # if self.warp_image:
        #     image = cv2.resize(image, (513, 513), interpolation=cv2.INTER_LINEAR)
        #     label = Image.fromarray(label).resize((513, 513), resample=Image.NEAREST)
        #     label = np.asarray(label)

        return image_id, image, label


