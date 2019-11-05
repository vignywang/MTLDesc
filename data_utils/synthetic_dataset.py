#
# Created by ZhangYuyang on 2019/8/9
#

import os
import glob
import cv2 as cv
import numpy as np

import torch
import torch.nn.functional as f
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

        new_label = torch.where(
            mask == 1, label, torch.ones_like(label) * 77
        )

        # sample = {"cv_image": cv_image_keypoint}
        sample = {"image": image, "label": label, "mask": mask, "new_label": new_label}
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


class SyntheticHeatmapDataset(Dataset):

    def __init__(self, params):
        self.params = params
        self.height = params.height
        self.width = params.width
        self.downsample_scale = 1
        self.sigma = 1  # 3
        self.g_kernel_size = 1  # 15
        self.g_paddings = self.g_kernel_size // 2
        self.dataset_dir = params.synthetic_dataset_dir
        self.image_list, self.point_list = self._format_file_list()
        self.homography_augmentation = HomographyAugmentation()
        self.photometric_augmentation = PhotometricAugmentation()

        self.localmap = self._generate_local_gaussian_map()

    def __len__(self):
        assert len(self.image_list) == len(self.point_list)
        return len(self.image_list)

    def __getitem__(self, idx):
        image = cv.imread(self.image_list[idx], flags=cv.IMREAD_GRAYSCALE)
        point = np.load(self.point_list[idx])

        mask = np.ones_like(image)
        if np.random.rand() >= 0.1:
            image, mask, point = self.homography_augmentation(image, point)
            image = self.photometric_augmentation(image)

        image = torch.from_numpy(image).to(torch.float).unsqueeze(dim=0)
        image = image*2./255. - 1.  # scale到[-1,1]之间

        # todo:
        mask = torch.from_numpy(cv.resize(
            mask,
            dsize=(int(self.width/self.downsample_scale), int(self.height/self.downsample_scale)),
            interpolation=cv.INTER_LINEAR
        )).to(torch.float)
        heatmap = self._convert_points_to_heatmap(points=point)

        # debug show
        # self._debug_show(heatmap, image)

        sample = {"image": image, "mask": mask, "heatmap": heatmap}
        return sample

    def _debug_show(self, heatmap, image):
        heatmap = heatmap.numpy() * 100
        heatmap = cv.resize(heatmap, dsize=(self.width, self.height), interpolation=cv.INTER_LINEAR)
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


class SyntheticAdversarialDataset(SyntheticTrainDataset):

    def __init__(self, params):
        super(SyntheticAdversarialDataset, self).__init__(params=params)

    def __getitem__(self, idx):

        image = cv.imread(self.image_list[idx], flags=cv.IMREAD_GRAYSCALE)
        point = np.load(self.point_list[idx])

        org_mask = np.ones_like(image)
        if torch.rand([]).item() >= 0.1:
            image, org_mask, point = self.homography_augmentation(image, point)
            image = self.photometric_augmentation(image)

        # 将亚像素精度处的点的位置去小数到整数
        point = np.abs(np.floor(point)).astype(np.int)

        # 将它们转换成tensor
        image = torch.from_numpy(image).to(torch.float).unsqueeze(dim=0)
        image = image * 2. / 255. - 1.  # scale到[-1,1]之间
        org_mask = torch.from_numpy(org_mask)
        point = torch.from_numpy(point)

        # 由点的位置生成训练所需label
        label = self.convert_points_to_label(point).to(torch.long)
        # 由原始的掩膜生成对应label形状的掩膜
        mask = space_to_depth(org_mask).to(torch.uint8)
        mask = torch.all(mask, dim=0).to(torch.float)

        sample = {"image": image, "label": label, "mask": mask}
        return sample


class SyntheticSuperPointStatisticDataset(Dataset):

    def __init__(self, params):
        self.params = params
        self.height = params.height
        self.width = params.width
        self.n_height = int(self.height / 8)
        self.n_width = int(self.width / 8)
        self.dataset_dir = params.synthetic_dataset_dir
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


