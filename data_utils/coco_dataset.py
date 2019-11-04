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


class COCOMegPointAdaptionDataset(Dataset):

    def __init__(self, params):
        self.height = params.height
        self.width = params.width
        self.coco_pseudo_idx = params.coco_pseudo_idx
        # self.dataset_dir = os.path.join(params.coco_dataset_dir, 'train2014', 'resized_images')
        self.dataset_dir = os.path.join(params.coco_dataset_dir, 'train2014/pseudo_image_points_'+self.coco_pseudo_idx)
        self.image_file_list = self._format_file_list(False)

    def __len__(self):
        return len(self.image_file_list)

    def __getitem__(self, idx):
        image = cv.imread(self.image_file_list[idx], flags=cv.IMREAD_GRAYSCALE)
        image = torch.from_numpy(image).to(torch.float).unsqueeze(dim=0)
        image = image*2./255. - 1.
        # image = self.image_list[idx]
        sample = {"image": image}
        return sample

    def _format_file_list(self, num_limits=False):
        image_list = glob.glob(os.path.join(self.dataset_dir, "*.jpg"))
        image_list = sorted(image_list)
        if num_limits:
            length = 20000  # 1000
        else:
            length = len(image_list)
        image_list = image_list[:length]
        return image_list

    def _read_from_list(self):
        print("Begin loading all the images to the memory")
        image_list = []
        for i in range(len(self.image_file_list)):
            image = cv.imread(self.image_file_list[i], flags=cv.IMREAD_GRAYSCALE)
            image = cv.resize(image, (self.width, self.height), interpolation=cv.INTER_LINEAR)
            image = torch.from_numpy(image).to(torch.float).unsqueeze(dim=0)
            image_list.append(image)
            if i % 1000 == 0:
                print("Having read %d images" % i)
        print("Loading finished!")
        return image_list


class COCOAdaptionDataset(Dataset):

    def __init__(self, params, dataset_type):
        assert dataset_type in ['train2014', 'val2014']
        self.params = params
        self.height = params.height
        self.width = params.width
        self.dataset_dir = os.path.join(params.coco_dataset_dir, dataset_type, 'images')
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
        self.coco_pseudo_idx = params.coco_pseudo_idx
        self.dataset_dir = os.path.join(params.coco_dataset_dir, 'train2014/pseudo_image_points_'+self.coco_pseudo_idx)
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
            if np.random.rand() >= 0.1:
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
        self.coco_pseudo_idx = params.coco_pseudo_idx
        self.dataset_dir = os.path.join(params.coco_dataset_dir, 'val2014/pseudo_image_points_'+self.coco_pseudo_idx)
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
        self.n_height = int(self.height/8)
        self.n_width = int(self.width/8)
        self.coco_pseudo_idx = params.coco_pseudo_idx
        self.dataset_dir = os.path.join(params.coco_dataset_dir, 'train2014/pseudo_image_points_'+self.coco_pseudo_idx)
        # self.params.logger.info("Initialize SuperPoint Train Dataset: %s" % self.dataset_dir)
        self.image_list, self.point_list = self._format_file_list()
        self.homography = HomographyAugmentation(**params.homography_params)
        self.photometric = PhotometricAugmentation(**params.photometric_params)
        self.center_grid = self._generate_center_grid()

        self.loss_type = params.loss_type.split('_')[0]
        assert self.loss_type in ['triplet', 'pairwise']

    def __len__(self):
        assert len(self.image_list) == len(self.point_list)
        return len(self.image_list)

    def __getitem__(self, idx):

        image = cv.imread(self.image_list[idx], flags=cv.IMREAD_GRAYSCALE)
        point = np.load(self.point_list[idx])
        org_mask = np.ones_like(image)
        # cv_image_keypoint = draw_image_keypoints(image, point)

        # 由随机采样的单应变换得到第二副图像及其对应的关键点位置、原始掩膜和该单应变换
        if torch.rand([]).item() < 0.5:
             warped_image, warped_org_mask, warped_point, homography = \
             image.copy(), org_mask.copy(), point.copy(), np.eye(3)
        else:
            warped_image, warped_org_mask, warped_point, homography = self.homography(image, point, return_homo=True)
        # warped_image, warped_org_mask, warped_point, homography = self.homography(image, point, return_homo=True)
        # cv_image_keypoint = draw_image_keypoints(warped_image, warped_point)

        # 1、对图像的相关处理
        if torch.rand([]).item() < 0.5:
            image = self.photometric(image)
        if torch.rand([]).item() < 0.5:
            warped_image = self.photometric(warped_image)

        image = torch.from_numpy(image).to(torch.float).unsqueeze(dim=0)
        warped_image = torch.from_numpy(warped_image).to(torch.float).unsqueeze(dim=0)
        image = image*2./255. - 1.
        warped_image = warped_image*2./255. - 1.

        # 2、对点和标签的相关处理
        # 2.1 输入的点标签和掩膜的预处理
        point = np.abs(np.floor(point)).astype(np.int)
        warped_point = np.abs(np.floor(warped_point)).astype(np.int)
        point = torch.from_numpy(point)
        warped_point = torch.from_numpy(warped_point)
        org_mask = torch.from_numpy(org_mask)
        warped_org_mask = torch.from_numpy(warped_org_mask)

        # 2.2 得到第一副图点和标签的最终输出
        label = self._convert_points_to_label(point).to(torch.long)
        mask = space_to_depth(org_mask).to(torch.uint8)
        mask = torch.all(mask, dim=0).to(torch.float)
        # point_mask = torch.where(
        #     label == 64,
        #     torch.zeros_like(label),
        #     torch.ones_like(label)
        # ).to(torch.float).reshape((-1,))  # 只有关键点区域是1

        # 2.3 得到第二副图点和标签的最终输出
        warped_label = self._convert_points_to_label(warped_point).to(torch.long)
        warped_mask = space_to_depth(warped_org_mask).to(torch.uint8)
        warped_mask = torch.all(warped_mask, dim=0).to(torch.float)

        # 3、对构造描述子loss有关关系的计算
        # 3.1 得到第二副图中有效描述子的掩膜
        warped_valid_mask = warped_mask.reshape((-1,))

        # 3.2 根据指定的loss类型计算不同的关系，
        #     pairwise要计算两两之间的对应关系，
        #     triplet要计算匹配对应关系，匹配有效掩膜，匹配点与其他点的近邻关系
        descriptor_mask = None
        matched_idx = None
        matched_valid = None
        not_search_mask = None
        if self.loss_type == 'pairwise':
            descriptor_mask = self._generate_descriptor_mask(homography)
            descriptor_mask = torch.from_numpy(descriptor_mask)
        else:
            matched_idx, matched_valid, not_search_mask = self.generate_corresponding_relationship(
                homography, warped_valid_mask)

            matched_idx = torch.from_numpy(matched_idx)
            matched_valid = torch.from_numpy(matched_valid).to(torch.float)
            not_search_mask = torch.from_numpy(not_search_mask)

        # 4、返回样本
        if self.loss_type == 'pairwise':
            return {
                'image': image,
                'mask': mask,
                'label': label,
                'warped_image': warped_image,
                'warped_mask': warped_mask,
                'warped_label': warped_label,
                'descriptor_mask': descriptor_mask,
                'warped_valid_mask': warped_valid_mask
            }
        else:
            return {
                'image': image,
                'mask': mask,
                'label': label,
                'warped_image': warped_image,
                'warped_mask': warped_mask,
                'warped_label': warped_label,
                'matched_idx': matched_idx,
                'matched_valid': matched_valid,
                'not_search_mask': not_search_mask,
            }

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
                center_grid.append((w, h))
        center_grid = np.stack(center_grid, axis=0)
        return center_grid

    def _generate_descriptor_mask(self, homography):

        center_grid, warped_center_grid = self.__compute_warped_center_grid(homography)

        center_grid = np.expand_dims(center_grid, axis=0)  # [1,n,2]
        warped_center_grid = np.expand_dims(warped_center_grid, axis=1)  # [n,1,2]

        dist = np.linalg.norm((warped_center_grid-center_grid), axis=2)  # [n,n]
        mask = (dist < 8.).astype(np.float32)

        return mask

    def generate_corresponding_relationship(self, homography, valid_mask, point_mask=None):

        # 1、得到当前所有描述子的中心点，以及它们经过单应变换后的中心点位置
        center_grid, warped_center_grid = self.__compute_warped_center_grid(homography)

        # 2、计算所有投影点与固定点的距离，从中找出匹配点，匹配点满足两者距离小于8
        dist = np.linalg.norm(warped_center_grid[:, np.newaxis, :]-center_grid[np.newaxis, :, :], axis=2)
        nearest_idx = np.argmin(dist, axis=1)
        nearest_dist = np.min(dist, axis=1)
        matched_valid = nearest_dist < 8.

        # 3、得到匹配点的坐标，并计算匹配点与匹配点间的距离，太近的非匹配点不会作为负样本出现在loss中
        matched_grid = center_grid[nearest_idx, :]
        diff = np.linalg.norm(matched_grid[:, np.newaxis, :] - matched_grid[np.newaxis, :, :], axis=2)
        nearest = diff < 16.

        # 4、根据当前匹配的idx得到无效点的mask
        valid_mask = valid_mask.numpy().astype(np.bool)
        valid_mask = valid_mask[nearest_idx]
        invalid = ~valid_mask[np.newaxis, :]

        # 5、得到不用搜索的区域mask，被mask的点要么太近，要么无效
        not_search_mask = (nearest | invalid).astype(np.float32)

        # 6、得到有效匹配的mask，满足（1）有匹配点（2）匹配点有效（3）其本身是关键点
        point_mask = point_mask.numpy().astype(np.bool)
        if point_mask is not None:
            matched_valid = matched_valid & valid_mask & point_mask
        else:
            matched_valid = matched_valid & valid_mask

        return nearest_idx, matched_valid, not_search_mask

    def __compute_warped_center_grid(self, homography, return_org_center_grid=True):

        center_grid = self.center_grid.copy()  # [n,2]
        num = center_grid.shape[0]
        ones = np.ones((num, 1), dtype=np.float)
        homo_center_grid = np.concatenate((center_grid, ones), axis=1)[:, :, np.newaxis]  # [n,3,1]
        warped_homo_center_grid = np.matmul(homography, homo_center_grid)
        warped_center_grid = warped_homo_center_grid[:, :2, 0] / warped_homo_center_grid[:, 2:, 0]  # [n,2]

        if return_org_center_grid:
            return center_grid, warped_center_grid
        else:
            return warped_center_grid


class COCOMegPointHeatmapTrainDataset(Dataset):

    def __init__(self, params, height=240, width=320):
        self.height = params.height
        self.width = params.width

        self.downsample_scale = 1
        self.sigma = 2
        self.g_kernel_size = 11
        self.g_paddings = self.g_kernel_size // 2

        self.n_height = int(self.height/8)
        self.n_width = int(self.width/8)
        self.dataset_dir = params.dataset_dir
        # self.params.logger.info("Initialize SuperPoint Train Dataset: %s" % self.dataset_dir)
        self.image_list, self.point_list = self._format_file_list()

        self.homography = HomographyAugmentation(**params.homography_params)
        self.photometric = PhotometricAugmentation(**params.photometric_params)

        self.center_grid = self._generate_center_grid()
        self.localmap = self._generate_local_gaussian_map()

    def __len__(self):
        assert len(self.image_list) == len(self.point_list)
        return len(self.image_list)

    def __getitem__(self, idx):

        image = cv.imread(self.image_list[idx], flags=cv.IMREAD_GRAYSCALE)
        point = np.load(self.point_list[idx])
        point_mask = np.ones_like(image).astype(np.float32)

        # 由随机采样的单应变换得到第二副图像及其对应的关键点位置、原始掩膜和该单应变换
        if torch.rand([]).item() < 0.5:
             warped_image, warped_point_mask, warped_point, homography = \
             image.copy(), point_mask.copy(), point.copy(), np.eye(3)
        else:
            warped_image, warped_point_mask, warped_point, homography = self.homography(
                image, point, mask=point_mask, return_homo=True)

        # 1、对图像增加哎噪声
        if torch.rand([]).item() < 0.5:
            image = self.photometric(image)
        if torch.rand([]).item() < 0.5:
            warped_image = self.photometric(warped_image)

        image = torch.from_numpy(image).to(torch.float).unsqueeze(dim=0)
        point_mask = torch.from_numpy(point_mask)

        warped_image = torch.from_numpy(warped_image).to(torch.float).unsqueeze(dim=0)
        warped_point_mask = torch.from_numpy(warped_point_mask)

        image = image*2./255. - 1.
        warped_image = warped_image*2./255. - 1.

        # 2.1 得到第一副图点构成的热图
        heatmap = self._convert_points_to_heatmap(point)
        # self._debug_show(heatmap, image)

        # 2.2 得到第二副图点构成的热图
        warped_heatmap = self._convert_points_to_heatmap(warped_point)

        # 3、对构造描述子loss有关关系的计算
        # 3.1 由变换有效点的掩膜得到有效描述子的掩膜
        warped_valid_mask = space_to_depth(warped_point_mask).clamp(0, 1).to(torch.uint8)
        warped_valid_mask = torch.all(warped_valid_mask, dim=0).to(torch.float)
        warped_valid_mask = warped_valid_mask.reshape((-1,))

        matched_idx, matched_valid, not_search_mask = self.generate_corresponding_relationship(
            homography, warped_valid_mask)

        matched_idx = torch.from_numpy(matched_idx)
        matched_valid = torch.from_numpy(matched_valid).to(torch.float)
        not_search_mask = torch.from_numpy(not_search_mask)

        return {
            'image': image,
            'point_mask': point_mask,
            'heatmap': heatmap,
            'warped_image': warped_image,
            'warped_point_mask': warped_point_mask,
            'warped_heatmap': warped_heatmap,
            'matched_idx': matched_idx,
            'matched_valid': matched_valid,
            'not_search_mask': not_search_mask,
        }

    def _debug_show(self, heatmap, image):
        heatmap = heatmap.numpy() * 150
        # heatmap = cv.resize(heatmap, dsize=(self.width, self.height), interpolation=cv.INTER_LINEAR)
        heatmap = cv.applyColorMap(heatmap.astype(np.uint8), colormap=cv.COLORMAP_BONE).astype(np.float)
        image = (image.squeeze().numpy() + 1) * 255 / 2
        hyper_image = np.clip(heatmap + image[:, :, np.newaxis], 0, 255).astype(np.uint8)
        cv.imshow("heat&image", hyper_image)
        cv.waitKey()

    def _convert_points_to_heatmap(self, points):
        height = int(self.height / self.downsample_scale)
        width = int(self.width / self.downsample_scale)
        assert height * self.downsample_scale == self.height and width * self.downsample_scale == self.width

        localmap = self.localmap.clone()
        padded_heatmap = torch.zeros(
            (height+self.g_paddings*2, width+self.g_paddings*2), dtype=torch.float)

        num_pt = points.shape[0]
        if num_pt > 0:
            for i in range(num_pt):
                pt = points[i]
                pt_y, pt_x = pt
                pt_y = int(pt_y // self.downsample_scale)  # 对真值点位置进行下采样,这里有量化误差
                pt_x = int(pt_x // self.downsample_scale)

                pt_y = np.clip(pt_y, 0, height)
                pt_x = np.clip(pt_x, 0, width)

                cur_localmap = padded_heatmap[pt_y: pt_y+self.g_kernel_size, pt_x: pt_x + self.g_kernel_size]
                cur_localmap = torch.where(localmap >= cur_localmap, localmap, cur_localmap)
                padded_heatmap[pt_y: pt_y+self.g_kernel_size, pt_x: pt_x + self.g_kernel_size] = cur_localmap

            heatmap = padded_heatmap[self.g_paddings: self.g_paddings+height, self.g_paddings: self.g_paddings+width]
        else:
            heatmap = torch.zeros((height, width), dtype=torch.float)

        return heatmap

    def _generate_local_gaussian_map(self):
        g_width = self.g_kernel_size
        g_height = self.g_kernel_size

        center_x = int(g_width / 2)
        center_y = int(g_height / 2)
        center = np.array((center_x, center_y))  # [2]

        coords_x = np.linspace(0, g_width-1, g_width)
        coords_y = np.linspace(0, g_height-1, g_height)

        coords = np.stack((
            np.tile(coords_x[np.newaxis, :], (g_height, 1)),
            np.tile(coords_y[:, np.newaxis], (1, g_width))),
            axis=2
        )  # [g_kernel_size,g_kernel_size,2]

        exponent = np.sum(np.square(coords - center), axis=2) / (2. * self.sigma * self.sigma)  # [13,13]
        localmap = np.exp(-exponent).astype(np.float32)
        localmap = torch.from_numpy(localmap)

        return localmap

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
                center_grid.append((w, h))
        center_grid = np.stack(center_grid, axis=0)
        return center_grid

    def _generate_descriptor_mask(self, homography):

        center_grid, warped_center_grid = self.__compute_warped_center_grid(homography)

        center_grid = np.expand_dims(center_grid, axis=0)  # [1,n,2]
        warped_center_grid = np.expand_dims(warped_center_grid, axis=1)  # [n,1,2]

        dist = np.linalg.norm((warped_center_grid-center_grid), axis=2)  # [n,n]
        mask = (dist < 8.).astype(np.float32)

        return mask

    def generate_corresponding_relationship(self, homography, valid_mask):

        # 1、得到当前所有描述子的中心点，以及它们经过单应变换后的中心点位置
        center_grid, warped_center_grid = self.__compute_warped_center_grid(homography)

        # 2、计算所有投影点与固定点的距离，从中找出匹配点，匹配点满足两者距离小于8
        dist = np.linalg.norm(warped_center_grid[:, np.newaxis, :]-center_grid[np.newaxis, :, :], axis=2)
        nearest_idx = np.argmin(dist, axis=1)
        nearest_dist = np.min(dist, axis=1)
        matched_valid = nearest_dist < 8.

        # 3、得到匹配点的坐标，并计算匹配点与匹配点间的距离，太近的非匹配点不会作为负样本出现在loss中
        matched_grid = center_grid[nearest_idx, :]
        diff = np.linalg.norm(matched_grid[:, np.newaxis, :] - matched_grid[np.newaxis, :, :], axis=2)
        nearest = diff < 16.

        # 4、根据当前匹配的idx得到无效点的mask
        valid_mask = valid_mask.numpy().astype(np.bool)
        valid_mask = valid_mask[nearest_idx]
        invalid = ~valid_mask[np.newaxis, :]

        # 5、得到不用搜索的区域mask，被mask的点要么太近，要么无效
        not_search_mask = (nearest | invalid).astype(np.float32)

        matched_valid = matched_valid & valid_mask

        return nearest_idx, matched_valid, not_search_mask

    def __compute_warped_center_grid(self, homography, return_org_center_grid=True):

        center_grid = self.center_grid.copy()  # [n,2]
        num = center_grid.shape[0]
        ones = np.ones((num, 1), dtype=np.float)
        homo_center_grid = np.concatenate((center_grid, ones), axis=1)[:, :, np.newaxis]  # [n,3,1]
        warped_homo_center_grid = np.matmul(homography, homo_center_grid)
        warped_center_grid = warped_homo_center_grid[:, :2, 0] / warped_homo_center_grid[:, 2:, 0]  # [n,2]

        if return_org_center_grid:
            return center_grid, warped_center_grid
        else:
            return warped_center_grid


class COCOSuperPointStatisticDataset(Dataset):

    def __init__(self, params):
        self.params = params
        self.height = params.height
        self.width = params.width
        self.n_height = int(self.height / 8)
        self.n_width = int(self.width / 8)
        self.dataset_dir = os.path.join(params.coco_dataset_dir,
                                        'train2014/pseudo_image_points_' + params.coco_pseudo_idx)
        # self.params.logger.info("Initialize SuperPoint Train Dataset: %s" % self.dataset_dir)
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

        # 由随机采样的单应变换得到第二副图像及其对应的关键点位置、原始掩膜和该单应变换
        warped_image, warped_org_mask, warped_point, homography = self.homography(image, point, return_homo=True)
        # warped_image, warped_org_mask, warped_point, homography = image.copy(), org_mask.copy(), point.copy(), np.eye(3)

        # 1、对图像的相关处理
        # if torch.rand([]).item() < 0.5:
        #     image = self.photometric(image)
        # if torch.rand([]).item() < 0.5:
        #     warped_image = self.photometric(warped_image)

        image = torch.from_numpy(image).to(torch.float).unsqueeze(dim=0)
        warped_image = torch.from_numpy(warped_image).to(torch.float).unsqueeze(dim=0)
        image = image*2./255. - 1.
        warped_image = warped_image*2./255. - 1.

        # 2、对点和标签的相关处理
        # 2.1 输入的点标签和掩膜的预处理
        point = np.abs(np.floor(point)).astype(np.int)
        warped_point = np.abs(np.floor(warped_point)).astype(np.int)
        point = torch.from_numpy(point)
        warped_point = torch.from_numpy(warped_point)
        org_mask = torch.from_numpy(org_mask)
        warped_org_mask = torch.from_numpy(warped_org_mask)

        # 2.2 得到第一副图点和标签的最终输出
        label = self._convert_points_to_label(point).to(torch.long)
        mask = space_to_depth(org_mask).to(torch.uint8)
        mask = torch.all(mask, dim=0).to(torch.float)
        point_mask = torch.where(
            label == 64,
            torch.zeros_like(label),
            torch.ones_like(label)
        ).reshape((-1,)).to(torch.float)

        # 2.3 得到第二副图点和标签的最终输出
        warped_label = self._convert_points_to_label(warped_point).to(torch.long)
        warped_mask = space_to_depth(warped_org_mask).to(torch.uint8)
        warped_mask = torch.all(warped_mask, dim=0).to(torch.float)

        # 3、对构造描述子loss有关关系的计算
        # 3.1 得到第二副图中有效描述子的掩膜
        warped_valid_mask = warped_mask.reshape((-1,))

        # 3.2 根据指定的loss类型计算不同的关系，
        matched_idx, matched_valid, not_search_mask_except_matched, not_search_mask = \
            self.generate_corresponding_relationship(
            homography, warped_valid_mask)
        matched_idx = torch.from_numpy(matched_idx)
        matched_valid = torch.from_numpy(matched_valid).to(torch.float)
        not_search_mask_except_matched = torch.from_numpy(not_search_mask_except_matched)
        not_search_mask = torch.from_numpy(not_search_mask)

        # 4、返回样本
        return {
            "image": image,
            "point_mask": point_mask,
            "warped_image": warped_image,
            "matched_idx": matched_idx,
            "matched_valid": matched_valid,
            "not_search_mask_except_matched": not_search_mask_except_matched,
            "not_search_mask": not_search_mask
        }

    def generate_corresponding_relationship(self, homography, valid_mask):

        center_grid, warped_center_grid = self.__compute_warped_center_grid(homography)

        dist = np.linalg.norm(warped_center_grid[:, np.newaxis, :]-center_grid[np.newaxis, :, :], axis=2)
        nearest_idx = np.argmin(dist, axis=1)
        nearest_dist = np.min(dist, axis=1)
        matched_valid = nearest_dist < 8.

        matched_grid = center_grid[nearest_idx, :]
        diff = np.linalg.norm(matched_grid[:, np.newaxis, :] - matched_grid[np.newaxis, :, :], axis=2)
        diff_except_matched = np.eye(diff.shape[0]) * 17 + diff

        valid_mask = valid_mask.numpy().astype(np.bool)
        valid_mask = valid_mask[nearest_idx]
        invalid = ~valid_mask[np.newaxis, :]

        nearest_except_matched = diff_except_matched < 16
        not_search_mask_except_matched = (nearest_except_matched | invalid).astype(np.float32)

        nearest = diff < 16.
        not_search_mask = (nearest | invalid).astype(np.float32)

        matched_valid = matched_valid & valid_mask

        return nearest_idx, matched_valid, not_search_mask_except_matched, not_search_mask

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
                center_grid.append((w, h))
        center_grid = np.stack(center_grid, axis=0)
        return center_grid

    def _generate_descriptor_mask(self, homography):

        center_grid, warped_center_grid = self.__compute_warped_center_grid(homography)

        center_grid = np.expand_dims(center_grid, axis=0)  # [1,n,2]
        warped_center_grid = np.expand_dims(warped_center_grid, axis=1)  # [n,1,2]

        dist = np.linalg.norm((warped_center_grid-center_grid), axis=2)  # [n,n]
        mask = (dist < 8.).astype(np.float32)

        return mask

    def __compute_warped_center_grid(self, homography, return_org_center_grid=True):

        center_grid = self.center_grid.copy()  # [n,2]
        num = center_grid.shape[0]
        ones = np.ones((num, 1), dtype=np.float)
        homo_center_grid = np.concatenate((center_grid, ones), axis=1)[:, :, np.newaxis]  # [n,3,1]
        warped_homo_center_grid = np.matmul(homography, homo_center_grid)
        warped_center_grid = warped_homo_center_grid[:, :2, 0] / warped_homo_center_grid[:, 2:, 0]  # [n,2]

        if return_org_center_grid:
            return center_grid, warped_center_grid
        else:
            return warped_center_grid


class COCOMegPointStatisticDataset(COCOSuperPointStatisticDataset):

    def __init__(self, params):
        super(COCOMegPointStatisticDataset, self).__init__(params)
        self.sample_num = params.sample_num

    def __getitem__(self, idx):

        image = cv.imread(self.image_list[idx], flags=cv.IMREAD_GRAYSCALE)

        warped_images = []
        for i in range(self.sample_num):
            warped_image, *_ = self.homography.warp(image)
            warped_image = torch.from_numpy(warped_image).to(torch.float).unsqueeze(dim=0)
            warped_image = warped_image * 2. / 255. - 1.
            warped_images.append(warped_image)

        warped_images = torch.cat(warped_images, dim=0)

        image = torch.from_numpy(image).to(torch.float).unsqueeze(dim=0)
        image = image*2./255. - 1.

        # if torch.rand([]).item() < 0.5:
        #     image = self.photometric(image)
        # if torch.rand([]).item() < 0.5:
        #     warped_image = self.photometric(warped_image)

        # 4、返回样本
        return {
            "image": image,
            "warped_images": warped_images,
        }


class COCOMegPointSelfTrainingDataset(Dataset):

    def __init__(self, params):
        self.params = params
        self.height = params.height
        self.width = params.width
        self.n_height = int(self.height/8)
        self.n_width = int(self.width/8)
        self.dataset_dir = os.path.join(params.coco_dataset_dir, 'train2014', 'resized_images')
        self.image_list = self._format_file_list()
        self.point_memory_list = []
        self.mask_memory_list = []
        self.image_memory_list = []
        self.homography = HomographyAugmentation(**params.homography_params)
        self.photometric = PhotometricAugmentation(**params.photometric_params)
        self.center_grid = self._generate_center_grid()

    def append(self, image, point, mask):
        # 输入的图像应该是uint8类型的
        self.image_memory_list.append(image)
        self.point_memory_list.append(point)
        self.mask_memory_list.append(mask)

    def reset(self):
        self.image_memory_list = []
        self.point_memory_list = []
        self.mask_memory_list = []

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        assert (len(self.point_memory_list) != 0) and (len(self.image_memory_list) != 0)

        image = self.image_memory_list[idx]
        point = self.point_memory_list[idx]
        org_mask = self.mask_memory_list[idx]

        # 由随机采样的单应变换得到第二副图像及其对应的关键点位置、原始掩膜和该单应变换
        if torch.rand([]).item() < 0.5:
            warped_image, warped_org_mask, warped_point, homography = \
            image.copy(), org_mask.copy(), point.copy(), np.eye(3)
        else:
            warped_image, warped_org_mask, warped_point, homography = self.homography(
                image, point, mask=org_mask, return_homo=True)

        # 1、对图像的相关处理
        if torch.rand([]).item() < 0.5:
            image = self.photometric(image)
        if torch.rand([]).item() < 0.5:
            warped_image = self.photometric(warped_image)

        image = torch.from_numpy(image).to(torch.float).unsqueeze(dim=0)
        warped_image = torch.from_numpy(warped_image).to(torch.float).unsqueeze(dim=0)
        image = image*2./255. - 1.
        warped_image = warped_image*2./255. - 1.

        # 2、对点和标签的相关处理
        # 2.1 输入的点标签和掩膜的预处理
        point = np.abs(np.floor(point)).astype(np.int)
        warped_point = np.abs(np.floor(warped_point)).astype(np.int)
        point = torch.from_numpy(point)
        warped_point = torch.from_numpy(warped_point)
        org_mask = torch.from_numpy(org_mask)
        warped_org_mask = torch.from_numpy(warped_org_mask)

        # 2.2 得到第一副图点和标签的最终输出
        label = self._convert_points_to_label(point).to(torch.long)
        mask = org_mask.to(torch.float)

        # 2.3 得到第二副图点和标签的最终输出
        warped_label = self._convert_points_to_label(warped_point).to(torch.long)
        warped_mask = warped_org_mask.to(torch.float)

        # 2.4 将第二副图的mask转换成描述子的mask，这个mask会被用来计算下面描述子间有效的对应关系
        warped_valid_mask = space_to_depth(warped_org_mask).to(torch.uint8)
        warped_valid_mask = torch.all(warped_valid_mask, dim=0).to(torch.float)

        # # 3、对构造描述子loss有关关系的计算
        # # 3.1 得到第二副图中有效描述子的掩膜
        warped_valid_mask = warped_valid_mask.reshape((-1,))

        matched_idx, matched_valid, not_search_mask = self.generate_corresponding_relationship(
            homography, warped_valid_mask)
        matched_idx = torch.from_numpy(matched_idx)
        matched_valid = torch.from_numpy(matched_valid).to(torch.float)
        not_search_mask = torch.from_numpy(not_search_mask)

        # 4、返回样本
        return {
            'image': image,
            'mask': mask,
            'label': label,
            'warped_image': warped_image,
            'warped_mask': warped_mask,
            'warped_label': warped_label,
            'matched_idx': matched_idx,
            'matched_valid': matched_valid,
            'not_search_mask': not_search_mask,
        }

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
        for org_image_dir in org_image_list:
            image_list.append(org_image_dir)

        return image_list

    def _generate_center_grid(self, patch_height=8, patch_width=8):
        n_height = int(self.height/patch_height)
        n_width = int(self.width/patch_width)
        center_grid = []
        for i in range(n_height):
            for j in range(n_width):
                h = (patch_height-1.)/2. + i*patch_height
                w = (patch_width-1.)/2. + j*patch_width
                center_grid.append((w, h))
        center_grid = np.stack(center_grid, axis=0)
        return center_grid

    def _generate_descriptor_mask(self, homography):

        center_grid, warped_center_grid = self.__compute_warped_center_grid(homography)

        center_grid = np.expand_dims(center_grid, axis=0)  # [1,n,2]
        warped_center_grid = np.expand_dims(warped_center_grid, axis=1)  # [n,1,2]

        dist = np.linalg.norm((warped_center_grid-center_grid), axis=2)  # [n,n]
        mask = (dist < 8.).astype(np.float32)

        return mask

    def generate_corresponding_relationship(self, homography, valid_mask):

        center_grid, warped_center_grid = self.__compute_warped_center_grid(homography)

        # 1、找到距离最近的点当作匹配点，其中距离小于8的点为有效匹配
        dist = np.linalg.norm(warped_center_grid[:, np.newaxis, :]-center_grid[np.newaxis, :, :], axis=2)
        nearest_idx = np.argmin(dist, axis=1)
        nearest_dist = np.min(dist, axis=1)
        matched_valid = nearest_dist < 8.

        # 2、因为可能存在1对多的情况，因此被匹配点间的距离如果太小，则可能代表同一个区域，这在后续构造负样本时要排除掉
        # 这里得到需要排除掉的mask
        matched_grid = center_grid[nearest_idx, :]
        diff = np.linalg.norm(matched_grid[:, np.newaxis, :] - matched_grid[np.newaxis, :, :], axis=2)
        # nearest = diff < 8.
        nearest = diff < 16.

        # 3、得到当前无效点的mask，这些点在构造负样本时也要排除掉
        valid_mask = valid_mask.numpy().astype(np.bool)
        valid_mask = valid_mask[nearest_idx]
        invalid = ~valid_mask[np.newaxis, :]

        # 4、综合相近的被匹配点以及无效匹配点，得到负样本搜索时要排除掉的点的mask，1代表要排除，0代表保留
        not_search_mask = (nearest | invalid).astype(np.float32)

        # 5、根据投影点是否出界以及其与对应匹配点间的距离是否小于8得到最终有效匹配的mask
        matched_valid = matched_valid & valid_mask

        return nearest_idx, matched_valid, not_search_mask

    def __compute_warped_center_grid(self, homography, return_org_center_grid=True):
        # 计算8x8区域的中心点的投影点位置并返回

        center_grid = self.center_grid.copy()  # [n,2]
        num = center_grid.shape[0]
        ones = np.ones((num, 1), dtype=np.float)
        homo_center_grid = np.concatenate((center_grid, ones), axis=1)[:, :, np.newaxis]  # [n,3,1]
        warped_homo_center_grid = np.matmul(homography, homo_center_grid)
        warped_center_grid = warped_homo_center_grid[:, :2, 0] / warped_homo_center_grid[:, 2:, 0]  # [n,2]

        if return_org_center_grid:
            return center_grid, warped_center_grid
        else:
            return warped_center_grid


class COCOMegPointSelfTrainingOnlyImageDataset(COCOMegPointSelfTrainingDataset):

    def __init__(self, params):
        super(COCOMegPointSelfTrainingOnlyImageDataset, self).__init__(params)

    def __getitem__(self, idx):
        assert (len(self.point_memory_list) != 0) and (len(self.image_memory_list) != 0)

        image = self.image_memory_list[idx]
        point = self.point_memory_list[idx]
        org_mask = self.mask_memory_list[idx]

        # 由随机采样的单应变换得到第二副图像及其对应的关键点位置、原始掩膜和该单应变换
        if torch.rand([]).item() > 0.1:
            image, org_mask, point, homography = self.homography(image, point, mask=org_mask, return_homo=True)

        # 1、对图像的相关处理
        if torch.rand([]).item() < 0.5:
            image = self.photometric(image)

        image = torch.from_numpy(image).to(torch.float).unsqueeze(dim=0)
        image = image*2./255. - 1.

        # 2、对点和标签的相关处理
        # 2.1 输入的点标签和掩膜的预处理
        point = np.abs(np.floor(point)).astype(np.int)
        point = torch.from_numpy(point)
        org_mask = torch.from_numpy(org_mask)

        label = self._convert_points_to_label(point).to(torch.long)
        mask = space_to_depth(org_mask).to(torch.uint8)
        mask = torch.all(mask, dim=0).to(torch.float)

        new_label = torch.where(
            mask == 1, label, torch.ones_like(label) * 77
        )

        # 4、返回样本
        return {
            "image": image,
            "mask": mask,
            "label": label,
            "new_label": new_label
        }


class COCORawDataset(Dataset):

    def __init__(self, coco_dataset_dir, height=240, width=320, postfix="resized_images"):
        self.height = height
        self.width = width
        self.n_height = int(self.height/8)
        self.n_width = int(self.width/8)
        self.dataset_dir = os.path.join(coco_dataset_dir, 'train2014', postfix)
        self.image_list, self.image_name_list = self._format_file_list()

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):

        image = cv.imread(self.image_list[idx], flags=cv.IMREAD_GRAYSCALE)
        org_image = image
        image = torch.from_numpy(image).to(torch.float).unsqueeze(dim=0)
        image = (image * 2. / 255.) - 1.

        name = self.image_name_list[idx]
        return {
            "image": image,
            "org_image": org_image,
            "name": name,
        }

    def _format_file_list(self):
        image_list = glob.glob(os.path.join(self.dataset_dir, "*.jpg"))
        image_list = sorted(image_list)
        image_name_list = []
        for image in image_list:
            image_name = (image.split('/')[-1]).split('.')[0]
            image_name_list.append(image_name)
        return image_list, image_name_list


class COCODebugDataset(Dataset):

    def __init__(self, dataset_dir, height=240, width=320, read_mask=False):
        self.read_mask = read_mask
        self.height = height
        self.width = width
        self.n_height = int(self.height/8)
        self.n_width = int(self.width/8)
        self.dataset_dir = dataset_dir
        self.image_list, self.point_list, self.mask_list = self._format_file_list()

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):

        image = cv.imread(self.image_list[idx], flags=cv.IMREAD_GRAYSCALE)
        point = np.load(self.point_list[idx])

        mask = None
        if self.read_mask:
            mask = np.load(self.mask_list[idx])

        return {
            "image": image,
            "point": point,
            "mask": mask
        }

    def _format_file_list(self):
        dataset_dir = self.dataset_dir
        org_image_list = glob.glob(os.path.join(dataset_dir, "*.jpg"))
        org_image_list = sorted(org_image_list)
        image_list = []
        point_list = []
        mask_list = []
        for org_image_dir in org_image_list:
            name = (org_image_dir.split('/')[-1]).split('.')[0]
            point_dir = os.path.join(dataset_dir, name + '.npy')
            if self.read_mask:
                mask_dir = os.path.join(dataset_dir, name + "_mask.npy")
                mask_list.append(mask_dir)
            image_list.append(org_image_dir)
            point_list.append(point_dir)

        return image_list, point_list, mask_list


class COCOMegPointAdversarialDataset(Dataset):

    def __init__(self, params):
        self.params = params
        self.height = params.height
        self.width = params.width
        self.n_height = int(self.height/8)
        self.n_width = int(self.width/8)
        self.dataset_dir = os.path.join(params.coco_dataset_dir, 'train2014', 'resized_images')
        self.image_list = self._format_file_list()
        self.homography = HomographyAugmentation(**params.homography_params)
        self.photometric = PhotometricAugmentation(**params.photometric_params)
        self.center_grid = self._generate_center_grid()

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):

        image = cv.imread(self.image_list[idx], flags=cv.IMREAD_GRAYSCALE)
        org_mask = np.ones_like(image)

        if torch.rand([]).item() < 0.5:
            warped_image, warped_org_mask, homography = image.copy(), org_mask.copy(), np.eye(3)
        else:
            warped_image, warped_org_mask, homography = self.homography.warp(image)

        if torch.rand([]).item() < 0.5:
            image = self.photometric(image)
        if torch.rand([]).item() < 0.5:
            warped_image = self.photometric(warped_image)

        # 1、得到图像以及对应的有效掩膜
        image = torch.from_numpy(image).to(torch.float).unsqueeze(dim=0)
        warped_image = torch.from_numpy(warped_image).to(torch.float).unsqueeze(dim=0)
        image = image*2./255. - 1.
        warped_image = warped_image*2./255. - 1.
        warped_org_mask = torch.from_numpy(warped_org_mask)

        # 2、得到描述子匹配的有效mask，代表经投影后也在图像范围内的点
        warped_mask = space_to_depth(warped_org_mask).to(torch.uint8)
        warped_mask = torch.all(warped_mask, dim=0).to(torch.float)
        warped_valid_mask = warped_mask.reshape((-1,))

        # 3.2 根据当前采样的单应变换计算描述子的匹配关系
        # matched_idx: 距离最近的匹配点的id
        # matched_valid: 匹配的有效性
        # not_search_mask: 对每一对匹配点而言的不搜索的负样本的范围
        matched_idx, matched_valid, not_search_mask = \
            self.generate_corresponding_relationship(homography, warped_valid_mask)
        matched_idx = torch.from_numpy(matched_idx)
        matched_valid = torch.from_numpy(matched_valid).to(torch.float)
        not_search_mask = torch.from_numpy(not_search_mask)

        # 4、返回样本
        return {
            'image': image,
            'warped_image': warped_image,
            'matched_idx': matched_idx,
            'matched_valid': matched_valid,
            'not_search_mask': not_search_mask,
        }

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
        for org_image_dir in org_image_list:
            image_list.append(org_image_dir)

        return image_list

    def _generate_center_grid(self, patch_height=8, patch_width=8):
        n_height = int(self.height/patch_height)
        n_width = int(self.width/patch_width)
        center_grid = []
        for i in range(n_height):
            for j in range(n_width):
                h = (patch_height-1.)/2. + i*patch_height
                w = (patch_width-1.)/2. + j*patch_width
                center_grid.append((w, h))
        center_grid = np.stack(center_grid, axis=0)
        return center_grid

    def _generate_descriptor_mask(self, homography):

        center_grid, warped_center_grid = self.__compute_warped_center_grid(homography)

        center_grid = np.expand_dims(center_grid, axis=0)  # [1,n,2]
        warped_center_grid = np.expand_dims(warped_center_grid, axis=1)  # [n,1,2]

        dist = np.linalg.norm((warped_center_grid-center_grid), axis=2)  # [n,n]
        mask = (dist < 8.).astype(np.float32)

        return mask

    def generate_corresponding_relationship(self, homography, valid_mask):

        center_grid, warped_center_grid = self.__compute_warped_center_grid(homography)

        # 1、找到距离最近的点当作匹配点，其中距离小于8的点为有效匹配
        dist = np.linalg.norm(warped_center_grid[:, np.newaxis, :]-center_grid[np.newaxis, :, :], axis=2)
        nearest_idx = np.argmin(dist, axis=1)
        nearest_dist = np.min(dist, axis=1)
        matched_valid = nearest_dist < 8.

        # 2、因为可能存在1对多的情况，因此被匹配点间的距离如果太小，则可能代表同一个区域，这在后续构造负样本时要排除掉
        # 这里得到需要排除掉的mask
        matched_grid = center_grid[nearest_idx, :]
        diff = np.linalg.norm(matched_grid[:, np.newaxis, :] - matched_grid[np.newaxis, :, :], axis=2)
        # nearest = diff < 8.
        nearest = diff < 16.

        # 3、得到当前无效点的mask，这些点在构造负样本时也要排除掉
        valid_mask = valid_mask.numpy().astype(np.bool)
        valid_mask = valid_mask[nearest_idx]
        invalid = ~valid_mask[np.newaxis, :]

        # 4、综合相近的被匹配点以及无效匹配点，得到负样本搜索时要排除掉的点的mask，1代表要排除，0代表保留
        not_search_mask = (nearest | invalid).astype(np.float32)

        # 5、根据投影点是否出界以及其与对应匹配点间的距离是否小于8得到最终有效匹配的mask
        matched_valid = matched_valid & valid_mask

        return nearest_idx, matched_valid, not_search_mask

    def __compute_warped_center_grid(self, homography, return_org_center_grid=True):
        # 计算8x8区域的中心点的投影点位置并返回

        center_grid = self.center_grid.copy()  # [n,2]
        num = center_grid.shape[0]
        ones = np.ones((num, 1), dtype=np.float)
        homo_center_grid = np.concatenate((center_grid, ones), axis=1)[:, :, np.newaxis]  # [n,3,1]
        warped_homo_center_grid = np.matmul(homography, homo_center_grid)
        warped_center_grid = warped_homo_center_grid[:, :2, 0] / warped_homo_center_grid[:, 2:, 0]  # [n,2]

        if return_org_center_grid:
            return center_grid, warped_center_grid
        else:
            return warped_center_grid


if __name__ == "__main__":

    np.random.seed(2343)

    class Parameters:
        coco_dataset_dir = '/data/MegPoint/dataset/coco'
        height = 240
        width = 320
        do_augmentation = True
        coco_pseudo_idx = '0'
        loss_type = 'triplet'  # 'binary'

        homography_params = {
            'patch_ratio': 0.8,  # 0.8,
            'perspective_amplitude_x': 0.2,  # 0.2,
            'perspective_amplitude_y': 0.2,  # 0.2,
            'scaling_sample_num': 5,
            'scaling_amplitude': 0.2,
            'translation_overflow': 0.05,
            'rotation_sample_num': 25,
            'rotation_max_angle': np.pi / 2,  # np.pi / 2,
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
            'brightness_max_abs_change': 25,  # 25,
            'contrast_min': 0.5,  # 0.3,
            'contrast_max': 1.5,  # 1.5,
            'shade_transparency_range': (-0.5, 0.5),  # (-0.5, 0.8),
            'shade_kernel_size_range': (100, 150),  # (50, 100),
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
    # magicpoint_adaption_dataset = COCOAdaptionTrainDataset(params)
    for i, data in enumerate(superpoint_train_dataset):
    # for i, data in enumerate(magicpoint_adaption_dataset):
        image = data['image']
        label = data['label']
        mask = data['mask']











