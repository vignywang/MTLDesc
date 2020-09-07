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
import torch
from PIL import Image
from torch.utils.data import Dataset

from .dataset_tools import HomographyAugmentation
from .dataset_tools import ImgAugTransform
from .dataset_tools import draw_image_keypoints


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
        image_id, image, label, org_image = self._load_data(index)
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
            'org_image': org_image
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
        org_image = image.copy()
        label = sio.loadmat(label_path)["S"]
        label -= 1  # unlabeled (0 -> -1)
        label[label == -1] = 255

        # Warping: this is just for reproducing the official scores on GitHub
        # if self.warp_image:
        #     image = cv2.resize(image, (513, 513), interpolation=cv2.INTER_LINEAR)
        #     label = Image.fromarray(label).resize((513, 513), resample=Image.NEAREST)
        #     label = np.asarray(label)

        return image_id, image, label, org_image


class CocoStuff10kTest(Dataset):

    def __init__(self, **config):
        self.config = config
        self._format_file_list()

    def __len__(self):
        return len(self.files)

    def _format_file_list(self):
        file_list = osp.join(self.config['dataset_root'], "imageLists", 'test.txt')
        file_list = tuple(open(file_list, "r"))
        file_list = [id_.rstrip() for id_ in file_list]
        self.files = file_list

    def _load_data(self, index):
        # Set paths
        image_id = self.files[index]
        image_path = osp.join(self.config['dataset_root'], "images", image_id + ".jpg")
        label_path = osp.join(self.config['dataset_root'], "annotations", image_id + ".mat")

        # Load an image and label
        # change bgt to rgb
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)[:, :, ::-1].astype(np.float32)
        label = sio.loadmat(label_path)["S"]

        return image, label

    def __getitem__(self, index):
        image, label = self._load_data(index)

        shape = image.shape
        org_h, org_w = shape[0], shape[1]

        # rescale to 16*
        if org_h % 16 != 0:
            scale_h = int(np.round(org_h / 16.) * 16.)
        else:
            scale_h = org_h

        if org_w % 16 != 0:
            scale_w = int(np.round(org_w / 16.) * 16.)
        else:
            scale_w = org_w

        image = cv2.resize(image, dsize=(scale_w, scale_h), interpolation=cv2.INTER_LINEAR)
        label = cv2.resize(label, dsize=(scale_w, scale_h), interpolation=cv2.INTER_NEAREST)

        image2 = image - np.array([122.675, 116.669, 104.008])  # - mean_rgb
        image2 = image2.transpose(2, 0, 1)

        image = image * 2. / 255. - 1
        image = image.transpose(2, 0, 1)

        label -= 1
        label[label == -1] = 255
        return {
            'image2': image2.astype(np.float32),
            'image': image.astype(np.float32),
            'label': label.astype(np.int64),
        }


class CocoStuff10kTrainDescriptor(Dataset):

    def __init__(self, **config):
        self.config = config
        self.homography = HomographyAugmentation()
        self.photometric = ImgAugTransform()
        self.fix_grid = self._generate_fixed_grid()
        self._format_file_list()

    def __len__(self):
        return len(self.files)

    def _format_file_list(self):
        file_list = osp.join(self.config['dataset_root'], 'imageLists', 'train.txt')
        file_list = tuple(open(file_list, "r"))
        file_list = [id_.rstrip() for id_ in file_list]
        self.files = file_list

    def _generate_fixed_grid(self, option=None):
        """
        预先采样固定间隔的225个图像格子
        """
        if option == None:
            y_num = 20
            x_num = 20
        else:
            y_num = option[0]
            x_num = option[1]

        grid_y = np.linspace(0, self.config['height']-1, y_num+1, dtype=np.int)
        grid_x = np.linspace(0, self.config['width']-1, x_num+1, dtype=np.int)

        grid_y_start = grid_y[:y_num].copy()
        grid_y_end = grid_y[1:y_num+1].copy()
        grid_x_start = grid_x[:x_num].copy()
        grid_x_end = grid_x[1:x_num+1].copy()

        grid_start = np.stack((np.tile(grid_y_start[:, np.newaxis], (1, x_num)),
                               np.tile(grid_x_start[np.newaxis, :], (y_num, 1))), axis=2).reshape((-1, 2))
        grid_end = np.stack((np.tile(grid_y_end[:, np.newaxis], (1, x_num)),
                             np.tile(grid_x_end[np.newaxis, :], (y_num, 1))), axis=2).reshape((-1, 2))
        grid = np.concatenate((grid_start, grid_end), axis=1)

        return grid

    def _load_data(self, index):
        # Set paths
        image_id = self.files[index]
        image_path = osp.join(self.config['dataset_root'], "images", image_id + ".jpg")
        label_path = osp.join(self.config['dataset_root'], "annotations", image_id + ".mat")

        # Load an image and label
        # change bgt to rgb
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)[:, :, ::-1].astype(np.float32)
        label = sio.loadmat(label_path)["S"]

        return image, label

    def __getitem__(self, idx):
        image, label = self._load_data(idx)
        image, label = self._preprocess(image, label)

        # sample homography
        if torch.rand([]).item() < 0.5:
            warped_image, warped_label, homography = image.copy(), label.copy(), np.eye(3)
        else:
            homography = self.homography.sample(self.config['height'], self.config['width'])
            warped_image = cv2.warpPerspective(image, homography, None, None, cv2.INTER_AREA, borderValue=0)
            warped_label = cv2.warpPerspective(label, homography, None, None, cv2.INTER_NEAREST, borderValue=0).astype(np.int64)

        if torch.rand([]).item() < 0.5:
            image = self.photometric(image)
            warped_image = self.photometric(warped_image)

        # sample descriptor points for descriptor learning
        desp_point = self._random_sample_point()
        shape = image.shape

        warped_desp_point, valid_mask, not_search_mask = self._generate_warped_point(
            desp_point, homography, shape[0], shape[1])

        image = image.astype(np.float32) * 2. / 255. - 1.
        warped_image = warped_image.astype(np.float32) * 2. / 255. - 1.

        label -= 1
        label[label == -1] = 255
        label = torch.from_numpy(label.astype(np.int64))
        warped_label -= 1
        warped_label[warped_label == - 1] = 255
        warped_label = torch.from_numpy(warped_label.astype(np.int64))

        image = torch.from_numpy(image).permute((2, 0, 1))
        warped_image = torch.from_numpy(warped_image).permute((2, 0, 1))

        desp_point = torch.from_numpy(self._scale_point_for_sample(desp_point))
        warped_desp_point = torch.from_numpy(self._scale_point_for_sample(warped_desp_point))

        valid_mask = torch.from_numpy(valid_mask)
        not_search_mask = torch.from_numpy(not_search_mask)

        return {
            'image': image,  # [1,h,w]
            'label': label,
            'warped_image': warped_image,  # [1,h,w]
            'warped_label': warped_label,
            'desp_point': desp_point,  # [n,1,2]
            'warped_desp_point': warped_desp_point,  # [n,1,2]
            'valid_mask': valid_mask,  # [n]
            'not_search_mask': not_search_mask,  # [n,n]
        }

    def _preprocess(self, image, label):
        # Scaling
        h, w = label.shape

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
            image = cv2.copyMakeBorder(image, value=0, **pad_kwargs)
            label = cv2.copyMakeBorder(label, value=0, **pad_kwargs)

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

    def _random_sample_point(self):
        """
        根据预设的输入图像大小，随机均匀采样坐标点
        """
        grid = self.fix_grid.copy()
        # 随机选择指定数目个格子

        point_list = []
        for i in range(grid.shape[0]):
            y_start, x_start, y_end, x_end = grid[i]
            rand_y = np.random.randint(y_start, y_end)
            rand_x = np.random.randint(x_start, x_end)
            point_list.append(np.array((rand_y, rand_x), dtype=np.float32))
        point = np.stack(point_list, axis=0)

        return point

    @ staticmethod
    def _generate_warped_point(point, homography, height, width, threshold=16):
        """
        根据投影变换得到变换后的坐标点，有效关系及不参与负样本搜索的矩阵
        Args:
            point: [n,2] 与warped_point一一对应
            homography: 点对之间的变换关系

        Returns:
            not_search_mask: [n,n] type为float32的mask,不搜索的位置为1
        """
        # 得到投影点的坐标
        point = np.concatenate((point[:, ::-1], np.ones((point.shape[0], 1))), axis=1)[:, :, np.newaxis]  # [n,3,1]
        project_point = np.matmul(homography, point)[:, :, 0]
        project_point = project_point[:, :2] / project_point[:, 2:3]
        project_point = project_point[:, ::-1]  # 调换为y,x的顺序

        # 投影点在图像范围内的点为有效点，反之则为无效点
        boarder_0 = np.array((0, 0), dtype=np.float32)
        boarder_1 = np.array((height-1, width-1), dtype=np.float32)
        valid_mask = (project_point >= boarder_0) & (project_point <= boarder_1)
        valid_mask = np.all(valid_mask, axis=1)
        invalid_mask = ~valid_mask

        # 根据无效点及投影点之间的距离关系确定不搜索的负样本矩阵

        dist = np.linalg.norm(project_point[:, np.newaxis, :] - project_point[np.newaxis, :, :], axis=2)
        not_search_mask = ((dist <= threshold) | invalid_mask[np.newaxis, :]).astype(np.float32)

        return project_point.astype(np.float32), valid_mask.astype(np.float32), not_search_mask

    def _scale_point_for_sample(self, point):
        """
        将点归一化到[-1,1]的区间范围内，并调换顺序为x,y，方便采样
        Args:
            point: [n,2] y,x的顺序，原始范围为[0,height-1], [0,width-1]
        Returns:
            point: [n,1,2] x,y的顺序，范围为[-1,1]
        """
        org_size = np.array((self.config['height']-1, self.config['width']-1), dtype=np.float32)
        point = ((point * 2. / org_size - 1.)[:, ::-1])[:, np.newaxis, :].copy()
        return point


class CocoStuff10kRaw(_BaseDataset):
    """COCO-Stuff 10k dataset"""

    def __init__(self,  **config):
        # self.warp_image = warp_image
        super(CocoStuff10kRaw, self).__init__(**config)

    def __getitem__(self, index):
        image_id, image, image_gray,label = self._load_data(index)

        return {
            'image_id': image_id,
            'image_color': image,
            'image_gray': image_gray,
            'label': label,
        }

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

        # load grayscale
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        image_gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        label = sio.loadmat(label_path)['S']

        return image_id, image, image_gray, label



