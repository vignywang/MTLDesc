#
# Created by ZhangYuyang on 2019/8/9
#

import os
import glob
import cv2 as cv
import numpy as np

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from data_utils.dataset_tools import HomographyAugmentation
from data_utils.dataset_tools import PhotometricAugmentation
from data_utils.dataset_tools import draw_image_keypoints
from data_utils.dataset_tools import space_to_depth


class SyntheticTrainDataset(Dataset):

    def __init__(self, params):
        self.params = params
        self.height = params.height
        self.width = params.width
        self.dataset_dir = params.synthetic_dataset_dir
        self.image_list, self.point_list = self._format_file_list()
        self.homography_augmentation = HomographyAugmentation()
        self.photometric_augmentation = PhotometricAugmentation()

    def __len__(self):
        assert len(self.image_list) == len(self.point_list)
        return len(self.image_list)

    def __getitem__(self, idx):

        image = cv.imread(self.image_list[idx], flags=cv.IMREAD_GRAYSCALE)
        point = np.load(self.point_list[idx])

        org_mask = np.ones_like(image)
        # cv_image_keypoint = draw_image_keypoints(image, point)
        if self.params.do_augmentation:
            if np.random.rand() >= 0.1:
                image, org_mask, point = self.homography_augmentation(image, point)
                image = self.photometric_augmentation(image)
                # cv_image_keypoint = draw_image_keypoints(image, point)

        # 将亚像素精度处的点的位置去小数到整数
        point = np.abs(np.floor(point)).astype(np.int)

        # 将它们转换成tensor
        image = torch.from_numpy(image).to(torch.float).unsqueeze(dim=0)
        image = image*2./255. - 1.  # scale到[-1,1]之间
        org_mask = torch.from_numpy(org_mask)
        point = torch.from_numpy(point)

        # 由点的位置生成训练所需label
        label = self.convert_points_to_label(point).to(torch.long)
        # 由原始的掩膜生成对应label形状的掩膜
        mask = space_to_depth(org_mask).to(torch.uint8)
        mask = torch.all(mask, dim=0).to(torch.float)

        # sample = {"cv_image": cv_image_keypoint}
        sample = {"image": image, "label": label, "mask": mask}
        return sample

    def convert_points_to_label(self, points):

        height = self.height
        width = self.width
        n_height = int(height / 8)
        n_width = int(width / 8)
        assert n_height * 8 == height and n_width * 8 == width

        num_pt = points.shape[0]
        label = torch.zeros((height*width))
        if num_pt > 0:
            points_h, points_w = torch.split(points, 1, dim=1)
            points_idx = points_w + points_h * width
            label = label.scatter_(dim=0, index=points_idx[:, 0], value=1.0).reshape((height, width))
        else:
            label = label.reshape((height, width))

        dense_label = space_to_depth(label)
        dense_label = torch.cat((dense_label, 0.5 * torch.ones((1, n_height, n_width))), dim=0)  # [65, 30, 40]
        sparse_label = torch.argmax(dense_label, dim=0)  # [30,40]

        return sparse_label

    def _format_file_list(self):
        dataset_dir = self.dataset_dir
        subfloder_dir = os.path.join(dataset_dir, "*")
        subfloder_list = glob.glob(subfloder_dir)
        subfloder_list = sorted(subfloder_list, key=lambda x: x.split('/')[-1])
        image_list = []
        point_list = []

        for subfloder in subfloder_list:
            subimage_dir = os.path.join(subfloder, "images/training/*.png")
            subpoint_dir = os.path.join(subfloder, "points/training/*.npy")
            subimage_list = glob.glob(subimage_dir)
            subpoint_list = glob.glob(subpoint_dir)
            subimage_list = sorted(subimage_list, key=lambda x: int(x.split('/')[-1].split('.')[0]))
            subpoint_list = sorted(subpoint_list, key=lambda x: int(x.split('/')[-1].split('.')[0]))
            image_list += subimage_list
            point_list += subpoint_list

        return image_list, point_list


class SyntheticValTestDataset(Dataset):

    def __init__(self, params, dataset_type='validation', add_noise=False):
        self.params = params
        self.height = params.height
        self.width = params.width
        self.dataset_dir = params.synthetic_dataset_dir
        if add_noise:
            self.add_noise = True
            self.photometric_noise = PhotometricAugmentation()
        else:
            self.add_noise = False
            self.photometric_noise = None

        if dataset_type not in {'validation', 'test'}:
            print("The dataset type must be validation or test, please check!")
            assert False
        self.dataset_type = dataset_type
        self.image_list, self.point_list = self._format_file_list()

    def __len__(self):
        assert len(self.image_list) == len(self.point_list)
        return len(self.image_list)

    def __getitem__(self, idx):

        image = cv.imread(self.image_list[idx], flags=cv.IMREAD_GRAYSCALE)
        point = np.load(self.point_list[idx])
        # debug_show_image_keypoints(image, point)
        if self.add_noise:
            image = self.photometric_noise(image)

        # 将亚像素精度处的点的位置四舍五入到整数
        # point = np.round(point).astype(np.int)
        point = np.floor(point).astype(np.int)
        image = torch.from_numpy(image).to(torch.float).unsqueeze(dim=0)
        image = image*2./255. - 1.  # 归一化到[-1,1]之间
        point = torch.from_numpy(point)

        sample = {"image": image, "gt_point": point}
        return sample

    def _format_file_list(self):
        dataset_dir = self.dataset_dir
        # 只读取有图案的数据来进行验证
        subfloder_dir = os.path.join(dataset_dir, "draw_*")
        subfloder_list = glob.glob(subfloder_dir)
        subfloder_list = sorted(subfloder_list, key=lambda x: x.split('/')[-1])
        image_list = []
        point_list = []

        for subfloder in subfloder_list:
            # 注意此处读取的是验证数据集的列表
            subimage_dir = os.path.join(subfloder, "images", self.dataset_type, "*.png")
            subpoint_dir = os.path.join(subfloder, "points", self.dataset_type, "*.npy")
            subimage_list = glob.glob(subimage_dir)
            subpoint_list = glob.glob(subpoint_dir)
            subimage_list = sorted(subimage_list, key=lambda x: int(x.split('/')[-1].split('.')[0]))
            subpoint_list = sorted(subpoint_list, key=lambda x: int(x.split('/')[-1].split('.')[0]))
            image_list += subimage_list
            point_list += subpoint_list

        return image_list, point_list


if __name__ == "__main__":

    np.random.seed(1234)
    torch.manual_seed(2312)

    class Parameters:
        synthetic_dataset_dir = "/data/MegPoint/dataset/synthetic"

        height = 240
        width = 320
        do_augmentation = True

    params = Parameters()
    synthetic_dataset = SyntheticTrainDataset(params=params)
    # synthetic_dataset = SyntheticValTestDataset(params=params)
    dataloader = DataLoader(synthetic_dataset, batch_size=1, shuffle=True, num_workers=4)
    for i, data in enumerate(dataloader):
        image = data['image'][0, 0].numpy().astype(np.uint8)
        label = data['label'][0].numpy()
        mask = data['mask'][0].numpy()
        cv.imshow('image', image)
        a = 3


