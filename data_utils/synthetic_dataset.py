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


class SyntheticTrainDataset(Dataset):

    def __init__(self, params):
        self.params = params
        self.height = params.height
        self.width = params.width
        self.dataset_dir = params.synthetic_dataset_dir
        self.image_list, self.point_list = self._format_file_list()
        self.homography_augmentation = HomographyAugmentation(**params.homography_params)
        self.photometric_augmentation = PhotometricAugmentation()

    def __len__(self):
        assert len(self.image_list) == len(self.point_list)
        return len(self.image_list)

    def __getitem__(self, idx):

        image = cv.imread(self.image_list[idx], flags=cv.IMREAD_GRAYSCALE)
        point = np.load(self.point_list[idx])

        org_mask = np.ones_like(image)
        # debug_show_image_keypoints(image, point)
        if self.params.do_augmentation:
            image, org_mask, point = self.homography_augmentation(image, point)
            # debug_show_image_keypoints(image, point)
            image = self.photometric_augmentation(image)
            # debug_show_image_keypoints(image, point)

        # 将亚像素精度处的点的位置四舍五入到整数
        point = np.floor(point).astype(np.int)

        # 将它们转换成tensor
        image = torch.from_numpy(image).to(torch.float).unsqueeze(dim=0)
        org_mask = torch.from_numpy(org_mask)
        point = torch.from_numpy(point)

        # 由点的位置生成训练所需label
        label = self.convert_points_to_label(point).to(torch.long)
        # 由原始的掩膜生成对应label形状的掩膜
        mask = space_to_depth(org_mask).to(torch.uint8)
        mask = torch.all(mask, dim=0).to(torch.float)

        sample = {"image": image, "label": label, "mask": mask}
        return sample

    def convert_points_to_label(self, points):

        height = self.height
        width = self.width
        n_height = int(height / 8)
        n_width = int(width / 8)
        assert n_height * 8 == height and n_width * 8 == width

        points_h, points_w = torch.split(points, 1, dim=1)
        points_idx = points_w + points_h * width
        label = torch.zeros((height * width))
        label = label.scatter_(dim=0, index=points_idx[:, 0], value=1.0).reshape((height, width))

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
        point = np.round(point).astype(np.int)
        image = torch.from_numpy(image).to(torch.float).unsqueeze(dim=0)
        point = torch.from_numpy(point)

        sample = {"image": image, "gt_point": point}
        return sample

    def convert_points_to_label(self, points):
        """
        将关键点label从[n,2]的稀疏点表示转换为[h/8,w/8]的sparse label,其中多的一维额外表示8x8区域内有无特征点
        Args:
            points: [h,w]
        Returns:
            sparse_label: [h/8. w/8]
        """
        height = self.height
        width = self.width
        n_height = int(height / 8)
        n_width = int(width / 8)
        assert n_height * 8 == height and n_width * 8 == width

        points_h, points_w = torch.split(points, 1, dim=1)
        points_idx = points_w + points_h * width
        label = torch.zeros((height * width))
        label = label.scatter_(dim=0, index=points_idx[:, 0], value=1.0).reshape((height, width))

        dense_label = space_to_depth(label)
        dense_label = torch.cat((dense_label, 0.5 * torch.ones((1, n_height, n_width))), dim=0)  # [65, 30, 40]
        sparse_label = torch.argmax(dense_label, dim=0)  # [30,40]

        return sparse_label

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


class HomographyAugmentation(object):

    def __init__(self, height=240, width=320, do_perspective=True, do_scaling=True, do_rotation=True,
                 do_translation=True, scaling_sample_num=5, scaling_amplitude=0.1, perspective_amplitude_x=0.1,
                 perspective_amplitude_y=0.1, patch_ratio=0.5, rotation_sample_num=25, rotation_max_angle=np.pi / 2,
                 allow_artifacts=False):
        self.height = height
        self.width = width
        self.patch_ratio = patch_ratio
        self.perspective_amplitude_x = perspective_amplitude_x
        self.perspective_amplitude_y = perspective_amplitude_y
        self.scaling_sample_num = scaling_sample_num
        self.scaling_amplitude = scaling_amplitude
        self.rotation_sample_num = rotation_sample_num
        self.rotation_max_angle = rotation_max_angle
        self.do_perspective = do_perspective
        self.do_scaling = do_scaling
        self.do_rotation = do_rotation
        self.do_translation = do_translation
        self.allow_artifacts = allow_artifacts

    def __call__(self, image, points):
        homography = self.sample_homography()
        image, mask = self._compute_warped_image_and_mask(image, homography)
        points = self.warp_keypoints(points, homography)
        return image, mask, points

    def _compute_warped_image_and_mask(self, image, homography):
        dsize = (self.width, self.height)
        org_mask = np.ones_like(image, dtype=np.float)
        warped_image = cv.warpPerspective(image, homography, dsize=dsize, flags=cv.INTER_LINEAR)
        warped_mask = cv.warpPerspective(org_mask, homography, dsize=dsize, flags=cv.INTER_LINEAR)
        valid_mask = warped_mask.astype(np.uint8)

        return warped_image, valid_mask

    @staticmethod
    def warp_keypoints(points, homography):
        """
        通过单应变换将原始关键点集变换到新的关键点集下
        Args:
            points: 变换前的关键点集合，shape为(n, 2)
            homography: 单应变换矩阵，shape为(3,3)

        Returns:
            new_points: 变换后的关键点集合
        """
        n, _ = points.shape
        points = np.flip(points, axis=1)
        points = np.concatenate((points, np.ones((n, 1))), axis=1)[:, :, np.newaxis]
        new_points = np.matmul(homography, points)[:, :, 0]
        new_points = new_points[:, :2] / new_points[:, 2:]  # 进行归一化
        new_points = new_points[:, ::-1]
        return new_points

    def sample_homography(self):

        pts_1 = np.array(((0, 0), (0, 1), (1, 1), (1, 0)), dtype=np.float)  # 注意这里第一维是x，第二维是y
        margin = (1 - self.patch_ratio) / 2
        pts_2 = margin + np.array(((0, 0), (0, self.patch_ratio),
                                   (self.patch_ratio, self.patch_ratio), (self.patch_ratio, 0)),
                                  dtype=np.float)

        # 进行透视变换
        if self.do_perspective:
            if not self.allow_artifacts:
                perspective_amplitude_x = min(self.perspective_amplitude_x, margin)
                perspective_amplitude_y = min(self.perspective_amplitude_y, margin)
            y_displacement = np.random.uniform(-perspective_amplitude_y, perspective_amplitude_y)
            x_displacement_left = np.random.uniform(-perspective_amplitude_x, perspective_amplitude_x)
            x_displacement_right = np.random.uniform(-perspective_amplitude_x, perspective_amplitude_x)

            pts_2 += np.array(((x_displacement_left, y_displacement),
                               (x_displacement_left, -y_displacement),
                               (x_displacement_right, y_displacement),
                               (x_displacement_right, -y_displacement)))

        # 进行尺度变换
        if self.do_scaling:
            # 得到n+1个尺度参数，其中最后一个为1，即不进行尺度化
            scales = np.concatenate((np.random.normal(1, self.scaling_amplitude, (self.scaling_sample_num,)),
                                     np.ones((1,))), axis=0)
            # 中心点不变的尺度缩放
            center = np.mean(pts_2, axis=0, keepdims=True)
            scaled = np.expand_dims(pts_2 - center, axis=0) * np.expand_dims(np.expand_dims(scales, 1), 1) + center
            if self.allow_artifacts:
                valid = np.arange(self.scaling_sample_num)  # all scales are valid except scale=1
            else:
                valid = np.where(np.all((scaled >= 0.) & (scaled < 1.), axis=(1, 2)))[0]
            idx = valid[np.random.randint(0, valid.shape[0])]
            # 从n_scales个随机的缩放中随机地取一个出来
            pts_2 = scaled[idx]

        # 进行平移变换
        if self.do_translation:
            t_min, t_max = np.min(pts_2, axis=0), np.min(1 - pts_2, axis=0)
            pts_2 += np.expand_dims(np.stack((np.random.uniform(-t_min[0], t_max[0]),
                                              np.random.uniform(-t_min[1], t_max[1]))), axis=0)

        if self.do_rotation:
            angles = np.linspace(-self.rotation_max_angle, self.rotation_max_angle, self.rotation_sample_num)
            angles = np.concatenate((angles, np.zeros((1,))), axis=0)  # in case no rotation is valid
            center = np.mean(pts_2, axis=0, keepdims=True)
            rot_mat = np.reshape(np.stack((np.cos(angles), -np.sin(angles),
                                           np.sin(angles), np.cos(angles)), axis=1), newshape=(-1, 2, 2))
            # [x, y] * | cos -sin|
            #          | sin  cos|
            rotated = np.matmul(
                np.tile(np.expand_dims(pts_2 - center, axis=0), reps=(self.rotation_sample_num + 1, 1, 1)), rot_mat
            ) + center
            if self.allow_artifacts:
                valid = np.arange(self.rotation_sample_num)  # all angles are valid, except angle=0
            else:
                # 得到未超边界值的idx
                valid = np.where(np.all((rotated >= 0.) & (rotated < 1.), axis=(1, 2)))[0]
            idx = valid[np.random.randint(0, valid.shape[0])]
            pts_2 = rotated[idx]

        def mat(p, q):
            coefficient_mat = np.empty((8, 8), dtype=np.float)
            for i in range(4):
                coefficient_mat[i, :] = [p[i][0], p[i][1], 1, 0, 0, 0, -p[i][0] * q[i][0], -p[i][1] * q[i][0]]
            for i in range(4):
                coefficient_mat[i + 4, :] = [0, 0, 0, p[i][0], p[i][1], 1, -p[i][0] * q[i][1], -p[i][1] * q[i][1]]
            return coefficient_mat

        # 将矩形以及变换后的四边形坐标还原到实际尺度上，并计算他们之间对应的单应变换
        size = np.array((self.width - 1, self.height - 1), dtype=np.float)
        pts_1 *= size
        pts_2 *= size

        a_mat = mat(pts_1, pts_2)
        b_mat = np.concatenate(np.split(pts_2, 2, axis=1), axis=0)
        homography = np.linalg.lstsq(a_mat, b_mat, None)[0].reshape((8,))
        homography = np.concatenate((homography, np.ones((1,))), axis=0).reshape((3, 3))

        return homography


class PhotometricAugmentation(object):

    def __init__(self):
        self.gaussian_noise_mean = 10
        self.gaussian_noise_std = 5
        self.speckle_noise_min_prob = 0
        self.speckle_noise_max_prob = 0.0035
        self.brightness_max_abs_change = 50
        self.contrast_min = 0.3
        self.contrast_max = 1.5
        self.shade_transparency_range = (-0.5, 0.8)
        self.shade_kernel_size_range = (50, 100)
        self.shade_nb_ellipses = 20
        self.motion_blur_max_kernel_size = 7
        self.do_gaussian_noise = True  # False
        self.do_speckle_noise = True  # False
        self.do_random_brightness = True  # False
        self.do_random_contrast = True  # False
        self.do_shade = True  # False
        self.do_motion_blur = True  # False

    def __call__(self, image):
        if self.do_gaussian_noise:
            image = self.apply_gaussian_noise(image)
        if self.do_speckle_noise:
            image = self.apply_speckle_noise(image)
        if self.do_random_brightness:
            image = self.apply_random_brightness(image)
        if self.do_random_contrast:
            image = self.apply_random_contrast(image)
        if self.do_shade:
            image = self.apply_shade(image)
        if self.do_motion_blur:
            image = self.apply_motion_blur(image)
        return image.astype(np.uint8)

    def apply_gaussian_noise(self, image):
        noise = np.random.normal(self.gaussian_noise_mean, self.gaussian_noise_std, size=np.shape(image))
        noise_image = image+noise
        noise_image = np.clip(noise_image, 0, 255)
        return noise_image

    def apply_speckle_noise(self, image):
        prob = np.random.uniform(self.speckle_noise_max_prob, self.speckle_noise_max_prob)
        sample = np.random.uniform(0, 1, size=np.shape(image))
        noisy_image = np.where(sample < prob, np.zeros_like(image), image)
        noisy_image = np.where(sample >= (1-prob), 255*np.ones_like(image), noisy_image)
        return noisy_image

    def apply_random_brightness(self, image):
        delta = np.random.uniform(-self.brightness_max_abs_change, self.brightness_max_abs_change)
        image = np.clip(image+delta, 0, 255)
        return image

    def apply_random_contrast(self, image):
        ratio = np.random.uniform(self.contrast_min, self.contrast_max)
        image = np.clip(image*ratio, 0, 255)
        return image

    def apply_shade(self, image) :
        min_dim = min(image.shape[:2]) / 4
        mask = np.zeros(image.shape[:2], np.uint8)
        for i in range(self.shade_nb_ellipses):
            ax = int(max(np.random.rand() * min_dim, min_dim / 5))
            ay = int(max(np.random.rand() * min_dim, min_dim / 5))
            max_rad = max(ax, ay)
            x = np.random.randint(max_rad, image.shape[1] - max_rad)  # center
            y = np.random.randint(max_rad, image.shape[0] - max_rad)
            angle = np.random.rand() * 90
            cv.ellipse(mask, (x, y), (ax, ay), angle, 0, 360, 255, -1)

        transparency = np.random.uniform(*self.shade_transparency_range)
        kernel_size = np.random.randint(*self.shade_kernel_size_range)
        if (kernel_size % 2) == 0:  # kernel_size has to be odd
            kernel_size += 1
        mask = cv.GaussianBlur(mask.astype(np.float32), (kernel_size, kernel_size), 0)
        shaded = image * (1 - transparency * mask / 255.)
        return np.clip(shaded, 0, 255)

    def apply_motion_blur(self, image):
        # Either vertial, hozirontal or diagonal blur
        mode = np.random.choice(['h', 'v', 'diag_down', 'diag_up'])
        ksize = np.random.randint(0, int((self.motion_blur_max_kernel_size + 1) / 2)) * 2 + 1  # make sure is odd
        center = int((ksize - 1) / 2)
        kernel = np.zeros((ksize, ksize))
        if mode == 'h':
            kernel[center, :] = 1.
        elif mode == 'v':
            kernel[:, center] = 1.
        elif mode == 'diag_down':
            kernel = np.eye(ksize)
        elif mode == 'diag_up':
            kernel = np.flip(np.eye(ksize), 0)
        var = ksize * ksize / 16.
        grid = np.repeat(np.arange(ksize)[:, np.newaxis], ksize, axis=-1)
        gaussian = np.exp(-(np.square(grid - center) + np.square(grid.T - center)) / (2. * var))
        kernel *= gaussian
        kernel /= np.sum(kernel)
        image = cv.filter2D(image, -1, kernel)
        return image


def space_to_depth(org_tensor, patch_height=8, patch_width=8):
    org_height, org_width = org_tensor.shape
    n_height = int(org_height/patch_height)
    n_width = int(org_width/patch_width)
    depth_dim = patch_height*patch_width
    assert n_height*patch_height == org_height and n_width*patch_width == org_width

    h_parts = org_tensor.split(patch_height, dim=0)
    all_parts = []
    for h_part in h_parts:
        w_parts = h_part.split(patch_width, dim=1)
        for w_part in w_parts:
            all_parts.append(w_part.reshape(depth_dim,))
    new_tensor = torch.stack(all_parts, dim=1).reshape((depth_dim, n_height, n_width))

    return new_tensor


def debug_show_image_keypoints(image, points):
    """
    将输入的关键点画到图像上并显示出来
    Args:
        image: 待画点的原始图像
        points: 图像对应的关键点组合，输入为np.array，shape为（n，2）, 点的第一维代表y轴，第二维代表x轴
    Returns:
        None
    """
    n, _ = points.shape
    cv_keypoints = []
    for i in range(n):
        keypt = cv.KeyPoint()
        keypt.pt = (points[i, 1], points[i, 0])
        cv_keypoints.append(keypt)
    image = cv.drawKeypoints(image.astype(np.uint8), cv_keypoints, None)
    cv.imshow("image&keypoints", image)
    cv.waitKey()


if __name__ == "__main__":

    class Parameters:
        synthetic_dataset_dir = "/data/MegPoint/dataset/synthetic"

        height = 240
        width = 320
        do_augmentation = True
        homography_params = {
            'do_translation': True,
            'do_rotation': True,
            'do_scaling': True,
            'do_perspective': True,
            'scaling_amplitude': 0.2,
            'perspective_amplitude_x': 0.2,
            'perspective_amplitude_y': 0.2,
            'patch_ratio': 0.8,
            'rotation_max_angle': 1.57,  # 3.14
            'allow_artifacts': False,
        }

    params = Parameters()
    # synthetic_dataset = SyntheticTrainDataset(params=params)
    synthetic_dataset = SyntheticValTestDataset(params=params)
    dataloader = DataLoader(synthetic_dataset, batch_size=16, shuffle=False, num_workers=4)
    for i, data in enumerate(dataloader):
        if i == 3:
            break





















