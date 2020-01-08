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
        if point_mask is not None:
            point_mask = point_mask.numpy().astype(np.bool)
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

    def __init__(self, params):
        self.params = params
        self.height = params.height
        self.width = params.width

        self.sigma = 1  # 3
        self.g_kernel_size = 1  # 15
        self.g_paddings = self.g_kernel_size // 2
        self.params.logger.info("Heatmap Dataset, the sigma:%d, the kernel_size:%d " % (self.sigma, self.g_kernel_size))

        self.n_height = int(self.height/8)
        self.n_width = int(self.width/8)
        self.dataset_dir = params.dataset_dir
        self.params.logger.info("Initialize MegPoint Train Dataset: %s" % self.dataset_dir)
        self.image_list, self.point_list = self._format_file_list()

        self.homography = HomographyAugmentation(**params.homography_params)
        self.photometric = PhotometricAugmentation(**params.photometric_params)

        self.center_grid = self._generate_center_grid()
        # self.localmap = self._generate_local_gaussian_map()

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

        homography = torch.from_numpy(homography).to(torch.float)

        return {
            "image": image,
            "point_mask": point_mask,
            "heatmap": heatmap,
            "warped_image": warped_image,
            "warped_point_mask": warped_point_mask,
            "warped_heatmap": warped_heatmap,
            "matched_idx": matched_idx,
            "matched_valid": matched_valid,
            "not_search_mask": not_search_mask,
            "homography": homography,
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
        """
        将原始点位置经下采样后得到heatmap与incmap，heatmap上对应下采样整型点位置处的值为1，其余为0；incmap与heatmap一一对应，
        在关键点位置处存放整型点到亚像素角点的偏移量，以及训练时用来屏蔽非关键点inc量的incmap_valid
        Args:
            points: [n,2]

        Returns:
            heatmap: [h,w] 关键点位置为1，其余为0
            incmap: [2,h,w] 关键点位置存放实际偏移，其余非关键点处的偏移量为0
            incmap_valid: [h,w] 关键点位置为1，其余为0，用于训练时屏蔽对非关键点偏移量的训练，只关注关键点的偏移量

        """
        height = self.height
        width = self.width

        # localmap = self.localmap.clone()
        # padded_heatmap = torch.zeros(
        #     (height+self.g_paddings*2, width+self.g_paddings*2), dtype=torch.float)
        heatmap = torch.zeros((height, width), dtype=torch.float)

        num_pt = points.shape[0]
        if num_pt > 0:
            for i in range(num_pt):
                pt = points[i]
                pt_y_float, pt_x_float = pt

                pt_y_int = round(pt_y_float)
                pt_x_int = round(pt_x_float)

                pt_y = int(pt_y_int)  # 对真值点位置进行下采样,这里有量化误差
                pt_x = int(pt_x_int)

                # 排除掉经下采样后在边界外的点
                if pt_y < 0 or pt_y > height - 1:
                    continue
                if pt_x < 0 or pt_x > width - 1:
                    continue

                # 关键点位置在heatmap上置1，并在incmap上记录该点离亚像素点的偏移量
                heatmap[pt_y, pt_x] = 1.0

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

        center_grid, warped_center_grid = self._compute_warped_center_grid(homography)

        center_grid = np.expand_dims(center_grid, axis=0)  # [1,n,2]
        warped_center_grid = np.expand_dims(warped_center_grid, axis=1)  # [n,1,2]

        dist = np.linalg.norm((warped_center_grid-center_grid), axis=2)  # [n,n]
        mask = (dist < 8.).astype(np.float32)

        return mask

    def generate_corresponding_relationship(self, homography, valid_mask):

        # 1、得到当前所有描述子的中心点，以及它们经过单应变换后的中心点位置
        center_grid, warped_center_grid = self._compute_warped_center_grid(homography)

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

    def _compute_warped_center_grid(self, homography, return_org_center_grid=True):

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


class COCOMegPointHeatmapOnlyDataset(COCOMegPointHeatmapTrainDataset):
    """
    只用于训练heatmap的数据集,将父类中与描述子有关的部分砍掉了
    """
    def __init__(self, params):
        super(COCOMegPointHeatmapOnlyDataset, self).__init__(params)

    def __getitem__(self, idx):
        image = cv.imread(self.image_list[idx], flags=cv.IMREAD_GRAYSCALE)
        point = np.load(self.point_list[idx])
        point_mask = np.ones_like(image).astype(np.float32)

        # 由随机采样的单应变换得到第二副图像及其对应的关键点位置、原始掩膜和该单应变换
        warped_image, warped_point_mask, warped_point, homography = self.homography(
            image, point, mask=point_mask, return_homo=True)

        # 1、对图像增加噪声
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

        # 2.2 得到第二副图点构成的热图
        warped_heatmap = self._convert_points_to_heatmap(warped_point)

        homography = torch.from_numpy(homography).to(torch.float)

        return {
            "image": image,
            "point_mask": point_mask,
            "heatmap": heatmap,
            "warped_image": warped_image,
            "warped_point_mask": warped_point_mask,
            "warped_heatmap": warped_heatmap,
            "homography": homography,
        }


class COCOMegPointDescriptorOnlyDataset(Dataset):
    """
    这个类只输出有关描述子训练相关所需要的数据
    """
    def __init__(self, params):
        self.params = params
        self.height = params.height
        self.width = params.width
        self.sample_num = params.sample_num

        # self.params.logger.info("Initialize MegPoint Train Dataset: %s" % self.dataset_dir)
        self.dataset_dir = params.dataset_dir
        self.image_list = self._format_file_list()

        self.homography = HomographyAugmentation()
        self.photometric = PhotometricAugmentation()

        self.fix_grid = self._generate_fixed_grid()

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image = cv.imread(self.image_list[idx], flags=cv.IMREAD_GRAYSCALE)
        image = self._crop_and_resize(image)

        # 1、得到随机采样的图像点
        point = self._random_sample_point()
        height, width = image.shape

        # 2、由随机采样的单应变换得到第二副图像 todo
        if torch.rand([]).item() < 0.5:
            warped_image, homography = image.copy(), np.eye(3)
        else:
            warped_image, _, homography = self.homography.warp(image)

        warped_point, valid_mask, not_search_mask = self._generate_warped_point(point, homography, height, width)

        # 3、按0.5的概率对两张图像进行加噪声
        if torch.rand([]).item() < 0.5:
            image = self.photometric(image)
        if torch.rand([]).item() < 0.5:
            warped_image = self.photometric(warped_image)

        # debug use
        # image_point = draw_image_keypoints(image, point, show=False)
        # warped_image_point = draw_image_keypoints(warped_image, warped_point, show=False)
        # cat_all = np.concatenate((image_point, warped_image_point), axis=1)
        # cv.imshow("cat_all", cat_all)
        # cv.waitKey()

        # todo
        image = image.astype(np.float32) * 2. / 255. - 1.
        warped_image = warped_image.astype(np.float32) * 2. / 255. - 1.

        image = torch.from_numpy(image).unsqueeze(dim=0)
        point = torch.from_numpy(self._scale_point_for_sample(point))
        warped_image = torch.from_numpy(warped_image).unsqueeze(dim=0)
        warped_point = torch.from_numpy(self._scale_point_for_sample(warped_point))

        valid_mask = torch.from_numpy(valid_mask)
        not_search_mask = torch.from_numpy(not_search_mask)

        return {
            "image": image,
            "point": point,
            "warped_image": warped_image,
            "warped_point": warped_point,
            "valid_mask": valid_mask,
            "not_search_mask": not_search_mask,
        }

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

        dist = np.linalg.norm(project_point[:, np.newaxis] - project_point[np.newaxis, :], axis=2)
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
        org_size = np.array((self.height-1, self.width-1), dtype=np.float32)
        point = ((point * 2. / org_size - 1.)[:, ::-1])[:, np.newaxis, :].copy()
        return point

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

        # 取格子的中心当作采样点
        # point_list = []
        # for i in range(grid.shape[0]):
        #     y_start, x_start, y_end, x_end = grid[i]
        #     y = (y_end - y_start) / 2 + y_start
        #     x = (x_end - x_start) / 2 + x_start
        #     point_list.append(np.array((y, x), dtype=np.float32))
        # point = np.stack(point_list, axis=0)

        return point

    def _generate_fixed_grid(self, x_num=20, y_num=20):
        """
        预先采样固定间隔的225个图像格子
        """
        grid_y = np.linspace(0, self.height-1, y_num+1, dtype=np.int)
        grid_x = np.linspace(0, self.width-1, x_num+1, dtype=np.int)

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

    def _crop_and_resize(self, img):
        """
        裁减图像并缩放到指定大小
        """
        org_height, org_width = img.shape
        ratio = self.height / self.width
        if org_height >= int(org_width * ratio):
            cur_height = int(org_width * ratio)
            delta = org_height - cur_height
            h_min = delta // 2
            h_max = h_min + cur_height
            img = img[h_min: h_max, :]
        else:
            cur_width = int(org_height / ratio)
            delta = org_width - cur_width
            w_min = delta // 2
            w_max = w_min + cur_width
            img = img[:, w_min: w_max]
        img = cv.resize(img, (self.width, self.height), None, interpolation=cv.INTER_LINEAR)

        return img

    def _format_file_list(self):
        dataset_dir = self.dataset_dir
        image_list = glob.glob(os.path.join(dataset_dir, "*.jpg"))
        image_list = sorted(image_list)

        return image_list


class COCOMegPointHeatmapAllTrainDataset(Dataset):
    """
    根据新的网络模型ResNetAll及HalfResNetAll设计的coco数据集类
    """

    def __init__(self, params):
        self.params = params
        self.height = params.height
        self.width = params.width
        assert self.height == 240 and self.width == 320

        self.dataset_dir = params.dataset_dir
        # self.params.logger.info("Initialize MegPoint Train Dataset: %s" % self.dataset_dir)
        self.image_list, self.point_list = self._format_file_list()

        self.homography = HomographyAugmentation()
        self.photometric = PhotometricAugmentation()

        self.fix_grid = self._generate_fixed_grid()

    def __len__(self):
        assert len(self.image_list) == len(self.point_list)
        return len(self.image_list)

    def __getitem__(self, idx):
        """
        Returns:
            image: [1,h,w] 归一化到[-1,1]之间的图像
            point_mask: [h,w] 每个点代表此处是否有效
            heatmap: [h,w] 有关键点的位置为1，否则为0
            warped_image: [1,h,w]
            warped_point_mask: [h,w]
            warped_heatmap: [h,w]
            desp_point: [n,1,2] 用于训练描述子随机采样的坐标点
            warped_desp_point: [n,1,2] 与desp_point对应的在变换图像中的采样点
            valid_mask: [n] 表示这些坐标点的有效性
            not_search_mask: [n,n] 用于构造loss时负样本不搜索的区域
        """
        image = cv.imread(self.image_list[idx], flags=cv.IMREAD_GRAYSCALE)
        point = np.load(self.point_list[idx])
        point_mask = np.ones_like(image).astype(np.float32)

        # 1、由随机采样的单应变换得到第二副图像及其对应的关键点位置、原始掩膜和该单应变换
        if torch.rand([]).item() < 0.5:
            warped_image, warped_point_mask, warped_point, homography = \
                image.copy(), point_mask.copy(), point.copy(), np.eye(3)
        else:
            warped_image, warped_point_mask, warped_point, homography = self.homography(image, point, return_homo=True)

        if torch.rand([]).item() < 0.5:
            image = self.photometric(image)
            warped_image = self.photometric(warped_image)

        # 2.1 得到第一副图点构成的热图
        heatmap = self._convert_points_to_heatmap(point)

        # 2.2 得到第二副图点构成的热图
        warped_heatmap = self._convert_points_to_heatmap(warped_point)

        # 3、采样训练描述子要用的点
        desp_point = self._random_sample_point()
        height, width = image.shape

        warped_desp_point, valid_mask, not_search_mask = self._generate_warped_point(
            desp_point, homography, height, width)

        # debug use
        # image_point = draw_image_keypoints(image, desp_point, show=False)
        # warped_image_point = draw_image_keypoints(warped_image, warped_desp_point, show=False)
        # cat_all = np.concatenate((image_point, warped_image_point), axis=1)
        # cv.imshow("cat_all", cat_all)
        # cv.waitKey()

        image = image.astype(np.float32) * 2. / 255. - 1.
        warped_image = warped_image.astype(np.float32) * 2. / 255. - 1.

        image = torch.from_numpy(image).unsqueeze(dim=0)
        warped_image = torch.from_numpy(warped_image).unsqueeze(dim=0)

        point_mask = torch.from_numpy(point_mask)
        warped_point_mask = torch.from_numpy(warped_point_mask)

        desp_point = torch.from_numpy(self._scale_point_for_sample(desp_point))
        warped_desp_point = torch.from_numpy(self._scale_point_for_sample(warped_desp_point))

        valid_mask = torch.from_numpy(valid_mask)
        not_search_mask = torch.from_numpy(not_search_mask)

        return {
            "image": image,  # [1,h,w]
            "point_mask": point_mask,  # [h,w]
            "heatmap": heatmap,  # [h,w]
            "warped_image": warped_image,  # [1,h,w]
            "warped_point_mask": warped_point_mask,  # [h,w]
            "warped_heatmap": warped_heatmap,  # [h,w]
            "desp_point": desp_point,  # [n,1,2]
            "warped_desp_point": warped_desp_point,  # [n,1,2]
            "valid_mask": valid_mask,  # [n]
            "not_search_mask": not_search_mask,  # [n,n]
        }

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

        dist = np.linalg.norm(project_point[:, np.newaxis] - project_point[np.newaxis, :], axis=2)
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
        org_size = np.array((self.height-1, self.width-1), dtype=np.float32)
        point = ((point * 2. / org_size - 1.)[:, ::-1])[:, np.newaxis, :].copy()
        return point

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

        # 取格子的中心当作采样点
        # point_list = []
        # for i in range(grid.shape[0]):
        #     y_start, x_start, y_end, x_end = grid[i]
        #     y = (y_end - y_start) / 2 + y_start
        #     x = (x_end - x_start) / 2 + x_start
        #     point_list.append(np.array((y, x), dtype=np.float32))
        # point = np.stack(point_list, axis=0)

        return point

    def _generate_fixed_grid(self, x_num=20, y_num=20):
        """
        预先采样固定间隔的225个图像格子
        """
        grid_y = np.linspace(0, self.height-1, y_num+1, dtype=np.int)
        grid_x = np.linspace(0, self.width-1, x_num+1, dtype=np.int)

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

    def _convert_points_to_heatmap(self, points):
        """
        将原始点位置经下采样后得到heatmap与incmap，heatmap上对应下采样整型点位置处的值为1，其余为0；incmap与heatmap一一对应，
        在关键点位置处存放整型点到亚像素角点的偏移量，以及训练时用来屏蔽非关键点inc量的incmap_valid
        Args:
            points: [n,2]

        Returns:
            heatmap: [h,w] 关键点位置为1，其余为0
            incmap: [2,h,w] 关键点位置存放实际偏移，其余非关键点处的偏移量为0
            incmap_valid: [h,w] 关键点位置为1，其余为0，用于训练时屏蔽对非关键点偏移量的训练，只关注关键点的偏移量

        """
        height = self.height
        width = self.width

        # localmap = self.localmap.clone()
        # padded_heatmap = torch.zeros(
        #     (height+self.g_paddings*2, width+self.g_paddings*2), dtype=torch.float)
        heatmap = torch.zeros((height, width), dtype=torch.float)

        num_pt = points.shape[0]
        if num_pt > 0:
            for i in range(num_pt):
                pt = points[i]
                pt_y_float, pt_x_float = pt

                pt_y_int = round(pt_y_float)
                pt_x_int = round(pt_x_float)

                pt_y = int(pt_y_int)  # 对真值点位置进行下采样,这里有量化误差
                pt_x = int(pt_x_int)

                # 排除掉经下采样后在边界外的点
                if pt_y < 0 or pt_y > height - 1:
                    continue
                if pt_x < 0 or pt_x > width - 1:
                    continue

                # 关键点位置在heatmap上置1，并在incmap上记录该点离亚像素点的偏移量
                heatmap[pt_y, pt_x] = 1.0

        return heatmap

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


class COCOMegPointHeatmapOnlyIndexDataset(COCOMegPointHeatmapTrainDataset):
    """
    这个类除了常规输出heatmap监督真值外，还需要输出固定数量的以关键点为中心的区域index值，用于训练时的heatmap采样
    """
    def __init__(self, params):
        super(COCOMegPointHeatmapOnlyIndexDataset, self).__init__(params)
        self.region_num = 100  # 以特征点为中心，然后随机扰动
        self.half_region_size = params.half_region_size  # 区域的大小
        self.general_coords = self._generate_general_region_coords()

    def __getitem__(self, idx):
        image = cv.imread(self.image_list[idx], flags=cv.IMREAD_GRAYSCALE)
        point = np.load(self.point_list[idx])
        point_mask = np.ones_like(image).astype(np.float32)

        # 由随机采样的单应变换得到第二副图像及其对应的关键点位置、原始掩膜和该单应变换
        warped_image, warped_point_mask, warped_point, point, homography = self.homography(
            image, point, mask=point_mask, return_homo=True, required_point_num=True)

        # 1、对图像增加噪声
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

        # 2.2 得到第二副图点构成的热图
        warped_heatmap = self._convert_points_to_heatmap(warped_point)

        homography = torch.from_numpy(homography).to(torch.float)

        # 3、根据关键点位置得到包含关键点的区域index
        ridx_list = []
        center_list = []
        warped_ridx_list = []
        warped_center_list = []
        for i in range(100):
            ridx, center_point = self._generate_region_index(point[i])
            warped_ridx, warped_center_point = self._generate_region_index(warped_point[i])

            ridx_list.append(ridx)
            center_list.append(center_point)

            warped_ridx_list.append(warped_ridx)
            warped_center_list.append(warped_center_point)

        ridx = torch.from_numpy(np.stack(ridx_list, axis=0))  # [100,kxk]
        center = torch.from_numpy(np.stack(center_list, axis=0))  # [100,2]
        warped_ridx = torch.from_numpy(np.stack(warped_ridx_list, axis=0))
        warped_center = torch.from_numpy(np.stack(warped_center_list, axis=0))

        return {
            "image": image,
            "point_mask": point_mask,
            "heatmap": heatmap,
            "ridx": ridx,
            "center": center,
            "warped_image": warped_image,
            "warped_point_mask": warped_point_mask,
            "warped_heatmap": warped_heatmap,
            "warped_ridx": warped_ridx,
            "warped_center": warped_center,
            "homography": homography,
            "idx": idx,
        }

    def _generate_region_index(self, point):
        """输入一个点，增加扰动并输出包含该点的指定大小区域的各个点的index
        Args:
            point: (2,) 分别为y,x坐标值
        Returns：
            region_idx: (rsize, rsize)，各个位置处的值为其在图像中的索引值
            upleft_coord: 区域左上角的坐标
        """
        point = np.round(point).astype(np.long)
        pt_y, pt_x = point
        hh_size = int(0.5*self.half_region_size)

        dy = np.random.randint(low=-hh_size, high=hh_size+1)
        dx = np.random.randint(low=-hh_size, high=hh_size+1)

        pt_y += dy
        pt_x += dx

        # 保证所取区域在图像范围内,pt代表所取区域的中心点位置
        if pt_y - self.half_region_size < 0:
            pt_y = self.half_region_size
        elif pt_y + self.half_region_size > self.height - 1:
            pt_y = self.height - 1 - self.half_region_size

        if pt_x - self.half_region_size < 0:
            pt_x = self.half_region_size
        elif pt_x + self.half_region_size > self.width - 1:
            pt_x = self.width - 1 - self.half_region_size

        center_point = np.array((pt_y, pt_x))

        # 得到区域中各个点的坐标
        region_coords = self.general_coords.copy()
        region_coords += center_point

        # 将坐标转换为idx
        coords_y, coords_x = np.split(region_coords, 2, axis=2)
        region_idx = (coords_y * self.width + coords_x).astype(np.long).reshape((-1,))
        return region_idx, center_point.astype(np.float32)

    def _generate_general_region_coords(self):
        """生成一个一般性的区域坐标点，每个点的坐标顺序为(y,x)"""
        region_size = int(self.half_region_size * 2 + 1)
        coords_x = np.tile(
            np.arange(-self.half_region_size, self.half_region_size+1)[np.newaxis, :], (region_size, 1))  # [n,n]
        coords_y = np.tile(
            np.arange(-self.half_region_size, self.half_region_size+1)[:, np.newaxis], (1, region_size))  # [n,n]
        coords = np.stack((coords_y, coords_x), axis=2)  # [n,n,2]
        return coords

    def _format_file_list(self):
        dataset_dir = self.dataset_dir
        image_list = []
        point_list = []

        with open(os.path.join(dataset_dir, "file_list.txt"), "r") as rf:
            all_lines = rf.readlines()
            for l in all_lines:
                image_point_dir = l.strip("\n")
                image_dir, point_dir = image_point_dir.split(",")
                image_list.append(image_dir)
                point_list.append(point_dir)

        return image_list, point_list


class COCOMegPointHeatmapPreciseTrainDataset(COCOMegPointHeatmapTrainDataset):
    """
    用于返回基于heatmap的点监督信息，以及用于描述子训练的投影点坐标，该坐标会用来插值得到精确对应的图象块描述子
    """
    def __init__(self, params):
        super(COCOMegPointHeatmapPreciseTrainDataset, self).__init__(params)

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

        # 2.2 得到第二副图点构成的热图
        warped_heatmap = self._convert_points_to_heatmap(warped_point)

        # 3、对构造描述子loss有关关系的计算
        matched_coords, matched_valid, not_search_mask = self.generate_corresponding_relationship(homography)

        matched_coords = torch.from_numpy(matched_coords)
        matched_valid = torch.from_numpy(matched_valid)
        not_search_mask = torch.from_numpy(not_search_mask)

        return {
            "image": image,
            "point_mask": point_mask,
            "heatmap": heatmap,
            "warped_image": warped_image,
            "warped_point_mask": warped_point_mask,
            "warped_heatmap": warped_heatmap,
            "matched_idx": torch.ones((1,)),
            "matched_coords": matched_coords,
            "matched_valid": matched_valid,
            "not_search_mask": not_search_mask,
        }

    def generate_corresponding_relationship(self, homography, valid_mask=None):
        # 1、得到当前所有描述子的中心点，以及它们经过单应变换后的中心点位置
        center_grid, warped_center_grid = self._compute_warped_center_grid(homography)

        # 2、匹配点的坐标即为经单应变换后的坐标，计算匹配点与匹配点间的距离，太近的非匹配点不会作为负样本出现在loss中
        matched_coords = warped_center_grid[:, ::-1].copy().astype(np.float32)  # y,x的顺序
        diff = np.linalg.norm(matched_coords[:, np.newaxis, :] - matched_coords[np.newaxis, :, :], axis=2)
        nearest = diff < 16.

        # 3、默认valid_mask需要通过投影点是否出界来重新进行计算
        warped_center_x, warped_center_y = np.split(warped_center_grid, 2, axis=1)
        valid_x = (warped_center_x <= self.width - 1) & (warped_center_x >= 0)
        valid_y = (warped_center_y <= self.height - 1) & (warped_center_y >= 0)
        matched_valid = (valid_x & valid_y)[:, 0]
        invalid = ~matched_valid[np.newaxis, :]

        # 4、得到不用搜索的区域mask，被mask的点要么太近，要么无效
        not_search_mask = (nearest | invalid).astype(np.float32)

        matched_valid = matched_valid.astype(np.float32)

        return matched_coords, matched_valid, not_search_mask


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


if __name__ == "__main__":

    np.random.seed(2343)

    class Parameters:
        dataset_dir = "/data/MegPoint/dataset/coco/train2014/pseudo_image_points_0"
        height = 240
        width = 320
        sample_num = 100

    params = Parameters()
    train_dataset = COCOMegPointHeatmapAllTrainDataset(params)
    for i, data in enumerate(train_dataset):
        a = 3













