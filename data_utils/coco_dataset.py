#
# Created by ZhangYuyang on 2019/8/19
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


class COCOAdaptionDataset(Dataset):

    def __init__(self, params, dataset_type):
        assert dataset_type in ['train2014', 'val2014']
        self.params = params
        self.height = params.height
        self.width = params.width
        self.dataset_dir = os.path.join(params.coco_dataset_root, dataset_type, 'images')
        if dataset_type == 'train2014':
            num_limits = False
        else:
            num_limits = True
        self.image_list, self.image_name_list = self._format_file_list(num_limits)

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image = cv.imread(self.image_list[idx], flags=cv.IMREAD_GRAYSCALE)
        image = cv.resize(image, (self.width, self.height), interpolation=cv.INTER_LINEAR)
        name = self.image_name_list[idx]
        sample = {'image': image, 'name': name}
        return sample

    def _format_file_list(self, num_limits=False):
        image_list = glob.glob(os.path.join(self.dataset_dir, "*.jpg"))
        image_list = sorted(image_list)
        if num_limits:
            length = 1000
        else:
            length = len(image_list)
        image_name_list = []
        for image in image_list:
            image_name = (image.split('/')[-1]).split('.')[0]
            image_name_list.append(image_name)
        image_list = image_list[:length]
        image_name_list = image_name_list[:length]
        return image_list, image_name_list


class COCOAdaptionTrainDataset(Dataset):

    def __init__(self, params):
        self.params = params
        self.height = params.height
        self.width = params.width
        self.dataset_dir = os.path.join(params.coco_dataset_dir, 'train2014/pseudo_image_points')
        self.image_list, self.point_list = self._format_file_list()
        self.homography_augmentation = HomographyAugmentation(**params.homography_params)
        self.photometric_augmentation = PhotometricAugmentation(**params.photometric_params)

    def __len__(self):
        assert len(self.image_list) == len(self.point_list)
        return len(self.image_list)

    def __getitem__(self, idx):

        image = cv.imread(self.image_list[idx], flags=cv.IMREAD_GRAYSCALE)
        point = np.load(self.point_list[idx])

        org_mask = np.ones_like(image)
        if self.params.do_augmentation:
            if np.random.rand() >= 0.5:
                image, org_mask, point = self.homography_augmentation(image, point)
                image = self.photometric_augmentation(image)

        # cv_image_keypoint = draw_image_keypoints(image, point)
        # 将亚像素精度处的点的位置去小数到整数
        point = np.abs(np.floor(point)).astype(np.int)

        # 将它们转换成tensor
        image = torch.from_numpy(image).to(torch.float).unsqueeze(dim=0)
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
        label = torch.zeros((height * width))
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
        org_image_list = glob.glob(os.path.join(dataset_dir, "*.jpg"))
        org_image_list = sorted(org_image_list)
        image_list = []
        point_list = []
        for org_image_dir in org_image_list:
            name = (org_image_dir.split('/')[-1]).split('.')[0]
            point_dir = os.path.join(dataset_dir, name + '.npy')
            image_list.append(org_image_dir)
            point_list.append(point_dir)

        return image_list, point_list


class COCOAdaptionValDataset(Dataset):

    def __init__(self, params, add_noise=False):
        self.params = params
        self.height = params.height
        self.width = params.width
        self.dataset_dir = os.path.join(params.coco_dataset_dir, 'val2014/pseudo_image_points')
        if add_noise:
            self.add_noise = True
            self.photometric_noise = PhotometricAugmentation(**params.photometric_params)
        else:
            self.add_noise = False
            self.photometric_noise = None

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
        point = torch.from_numpy(point)

        sample = {"image": image, "gt_point": point}
        return sample

    def _format_file_list(self):
        dataset_dir = self.dataset_dir
        org_image_list = glob.glob(os.path.join(dataset_dir, "*.jpg"))
        org_image_list = sorted(org_image_list)
        image_list = []
        point_list = []
        for org_image_dir in org_image_list:
            name = (org_image_dir.split('/')[-1]).split('.')[0]
            point_dir = os.path.join(dataset_dir, name + '.npy')
            image_list.append(org_image_dir)
            point_list.append(point_dir)

        return image_list, point_list


class COCOSuperPointTrainDataset(Dataset):

    def __init__(self, params):
        self.params = params
        self.height = params.height
        self.width = params.width
        self.dataset_dir = os.path.join(params.coco_dataset_dir, 'train2014/pseudo_image_points1')
        self.image_list, self.point_list = self._format_file_list()
        self.homography = HomographyAugmentation(**params.homography_params)
        self.photometric = PhotometricAugmentation(**params.photometric_params)
        self.center_grid = self._generate_center_grid()

    def __len__(self):
        assert len(self.image_list) == len(self.point_list)
        return len(self.image_list)

    def __getitem__(self, idx):

        image = cv.imread(self.image_list[idx], flags=cv.IMREAD_GRAYSCALE)
        point = np.load(self.point_list[idx])
        org_mask = np.ones_like(image)
        # cv_image_keypoint = draw_image_keypoints(image, point)

        # random sample homography
        warped_image, warped_org_mask, warped_point, homography = self.homography(image, point, return_homo=True)
        # cv_image_keypoint = draw_image_keypoints(warped_image, warped_point)
        descriptor_mask = self._generate_descriptor_mask(homography)

        # augmentation
        if np.random.rand() < 0.5:
            image = self.photometric(image)
        if np.random.rand() < 0.5:
            warped_image = self.photometric(warped_image)

        point = np.abs(np.floor(point)).astype(np.int)
        warped_point = np.abs(np.floor(warped_point)).astype(np.int)

        # original image related
        image = torch.from_numpy(image).to(torch.float).unsqueeze(dim=0)
        org_mask = torch.from_numpy(org_mask)
        point = torch.from_numpy(point)
        homography = torch.from_numpy(homography)

        # warped image related
        warped_image = torch.from_numpy(warped_image).to(torch.float).unsqueeze(dim=0)
        warped_org_mask = torch.from_numpy(warped_org_mask)
        warped_point = torch.from_numpy(warped_point)

        descriptor_mask = torch.from_numpy(descriptor_mask)

        # original image related
        label = self._convert_points_to_label(point).to(torch.long)
        mask = space_to_depth(org_mask).to(torch.uint8)
        mask = torch.all(mask, dim=0).to(torch.float)

        # warped image related
        warped_label = self._convert_points_to_label(warped_point).to(torch.long)
        warped_mask = space_to_depth(warped_org_mask).to(torch.uint8)
        warped_mask = torch.all(warped_mask, dim=0).to(torch.float)

        sample = {'image': image, 'mask': mask, 'label': label,
                  'warped_image': warped_image, 'warped_mask': warped_mask, 'warped_label': warped_label,
                  'homography': homography, 'descriptor_mask': descriptor_mask}

        return sample

    def _convert_points_to_label(self, points):

        height = self.height
        width = self.width
        n_height = int(height / 8)
        n_width = int(width / 8)
        assert n_height * 8 == height and n_width * 8 == width

        num_pt = points.shape[0]
        label = torch.zeros((height * width))
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
        org_image_list = glob.glob(os.path.join(dataset_dir, "*.jpg"))
        org_image_list = sorted(org_image_list)
        image_list = []
        point_list = []
        for org_image_dir in org_image_list:
            name = (org_image_dir.split('/')[-1]).split('.')[0]
            point_dir = os.path.join(dataset_dir, name + '.npy')
            image_list.append(org_image_dir)
            point_list.append(point_dir)

        return image_list, point_list

    def _generate_center_grid(self, patch_height=8, patch_width=8):
        n_height = int(self.height/patch_height)
        n_width = int(self.width/patch_width)
        center_grid = []
        for i in range(n_height):
            for j in range(n_width):
                h = (patch_height-1.)/2. + i*patch_height
                w = (patch_width-1.)/2. + j*patch_width
                center_grid.append((h, w))
        center_grid = np.stack(center_grid, axis=0)
        return center_grid

    def _generate_descriptor_mask(self, homography):
        # homography = np.eye(3)
        center_grid = self.center_grid.copy()  # [n,2]
        num = center_grid.shape[0]
        ones = np.ones((num, 1), dtype=np.float)
        homo_center_grid = np.concatenate((center_grid, ones), axis=1)[:, :, np.newaxis]  # [n,3,1]
        warped_homo_center_grid = np.matmul(homography, homo_center_grid)
        warped_center_grid = warped_homo_center_grid[:, :2, 0] / warped_homo_center_grid[:, 2:, 0]  # [n,2]

        center_grid = np.expand_dims(center_grid, axis=1)  # [n,1,2]
        warped_center_grid = np.expand_dims(warped_center_grid, axis=0)  # [1,n,2]

        dist = np.linalg.norm((warped_center_grid-center_grid), axis=2)  # [n,n]
        mask = (dist < 8.).astype(np.float32)

        return mask


if __name__ == "__main__":

    np.random.seed(2343)

    class Parameters:
        coco_dataset_dir = '/data/MegPoint/dataset/coco'
        height = 240
        width = 320
        do_augmentation = True

        homography_params = {
            'patch_ratio': 0.9,  # 0.8,
            'perspective_amplitude_x': 0.1,  # 0.2,
            'perspective_amplitude_y': 0.1,  # 0.2,
            'scaling_sample_num': 5,
            'scaling_amplitude': 0.2,
            'translation_overflow': 0.05,
            'rotation_sample_num': 25,
            'rotation_max_angle': np.pi / 3,  # np.pi / 2,
            'do_perspective': True,
            'do_scaling': True,
            'do_rotation': True,
            'do_translation': True,
            'allow_artifacts': True
        }

        photometric_params = {
            'gaussian_noise_mean': 0,  # 10,
            'gaussian_noise_std': 5,
            'speckle_noise_min_prob': 0,
            'speckle_noise_max_prob': 0.0035,
            'brightness_max_abs_change': 15,  # 25,
            'contrast_min': 0.7,  # 0.3,
            'contrast_max': 1.3,  # 1.5,
            'shade_transparency_range': (-0.5, 0.5),  # (-0.5, 0.8),
            'shade_kernel_size_range': (50, 100),
            'shade_nb_ellipese': 20,
            'motion_blur_max_kernel_size': 7,
            'do_gaussian_noise': True,
            'do_speckle_noise': True,
            'do_random_brightness': True,
            'do_random_contrast': True,
            'do_shade': True,
            'do_motion_blur': True
        }


    params = Parameters()
    superpoint_train_dataset = COCOSuperPointTrainDataset(params)
    for i, data in enumerate(superpoint_train_dataset):
        image = data['image']
        label = data['label']
        mask = data['mask']











