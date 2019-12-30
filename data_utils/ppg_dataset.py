# 
# Created by ZhangYuyang on 2019/12/29
#
import os
import glob

import cv2 as cv
import numpy as np
import torch
from torch.utils.data import Dataset

from data_utils.dataset_tools import HomographyAugmentation
from data_utils.dataset_tools import PhotometricAugmentation
from data_utils.dataset_tools import space_to_depth
from data_utils.dataset_tools import draw_image_keypoints


class PPGMegPointHeatmapTrainDataset(Dataset):

    def __init__(self, params):
        self.params = params
        self.height = params.height
        self.width = params.width

        self.n_height = int(self.height/8)
        self.n_width = int(self.width/8)
        self.dataset_dir = params.dataset_dir
        # self.params.logger.info("Initialize MegPoint Train Dataset: %s" % self.dataset_dir)
        self.image_list, self.point_list = self._format_file_list()

        self.homography = HomographyAugmentation(self.height, self.width, **params.homography_params)
        self.photometric = PhotometricAugmentation(**params.photometric_params)

        self.center_grid = self._generate_center_grid()

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

    def _crop_and_resize(self, img, point):
        org_height, org_width = img.shape
        ratio = self.height / self.width
        if org_height >= int(org_width * ratio):
            cur_height = int(org_width * ratio)
            delta = org_height - cur_height
            h_min = delta // 2
            h_max = h_min + cur_height
            img = img[h_min: h_max, :]
            point = self._filter_points(point, h_min, h_max, 0, org_width)
            x_scale = self.width / org_width
            y_scale = self.height / cur_height
        else:
            cur_width = int(org_height / ratio)
            delta = org_width - cur_width
            w_min = delta // 2
            w_max = w_min + cur_width
            img = img[:, w_min: w_max]
            point = self._filter_points(point, 0, org_height, w_min, w_max)
            x_scale = self.width / cur_width
            y_scale = self.height / org_height
        img = cv.resize(img, (self.width, self.height), None, interpolation=cv.INTER_LINEAR)
        point *= np.array((y_scale, x_scale))

        return img, point

    def _filter_points(self, point, h_min, h_max, w_min, w_max):
        pt_num = point.shape[0]
        filtered_point = []
        for i in range(pt_num):
            y, x = point[i]
            if y < h_min or y > h_max - 1 or x < w_min or x > w_max - 1:
                continue
            else:
                dis = np.array((h_min, w_min))
                filtered_point.append(point[i] - dis)
        point = np.stack(filtered_point, axis=0)
        return point


class PPGMegPointHeatmapOnlyDataset(PPGMegPointHeatmapTrainDataset):
    """
    只用于训练heatmap的数据集,将父类中与描述子有关的部分砍掉了
    """
    def __init__(self, params):
        super(PPGMegPointHeatmapOnlyDataset, self).__init__(params)

    def __getitem__(self, idx):
        image = cv.imread(self.image_list[idx], flags=cv.IMREAD_GRAYSCALE)
        point = np.load(self.point_list[idx])

        # 调整图像到指定大小
        image, point = self._crop_and_resize(image, point)
        # org_image_point = draw_image_keypoints(image, point, show=False)

        point_mask = np.ones_like(image).astype(np.float32)

        # 由随机采样的单应变换得到第二副图像及其对应的关键点位置、原始掩膜和该单应变换
        warped_image, warped_point_mask, warped_point, homography = self.homography(
            image, point, mask=point_mask, return_homo=True)
        warped_image_point = draw_image_keypoints(warped_image, warped_point, show=False)
        # cat_iamge_point = np.concatenate((org_image_point, warped_image_point), axis=1)
        # cv.imshow("cat", cat_iamge_point)
        # cv.waitKey()

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


if __name__ == "__main__":
    np.random.seed(2343)

    class Parameters:
        dataset_dir = "/data/MegPoint/dataset/indoorDist/train"
        height = 375
        width = 500

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
    dataset = PPGMegPointHeatmapOnlyDataset(params)
    for i, data in enumerate(dataset):
        a = 3






