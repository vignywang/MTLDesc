#
# Created by ZhangYuyang on 2019/8/19
#
import cv2 as cv
import numpy as np

import torch


class HomographyAugmentation(object):

    def __init__(self):
        self.height = 240
        self.width = 320
        self.patch_ratio = 0.8
        self.perspective_amplitude_x = 0.2
        self.perspective_amplitude_y = 0.2
        self.scaling_sample_num = 5
        self.scaling_amplitude = 0.2
        self.translation_overflow = 0.05
        self.rotation_sample_num = 25
        self.rotation_max_angle = np.pi/2
        self.do_perspective = True
        self.do_scaling = True
        self.do_rotation = True
        self.do_translation = True
        self.allow_artifacts = True

    def __call__(self, image, points):
        homography = self.sample_homography()
        image, mask = self._compute_warped_image_and_mask(image, homography)
        points = self._warp_keypoints(points, homography)
        return image, mask, points

    def _compute_warped_image_and_mask(self, image, homography):
        dsize = (self.width, self.height)
        org_mask = np.ones_like(image, dtype=np.float)
        warped_image = cv.warpPerspective(image, homography, dsize=dsize, flags=cv.INTER_LINEAR)
        warped_mask = cv.warpPerspective(org_mask, homography, dsize=dsize, flags=cv.INTER_LINEAR)
        valid_mask = warped_mask.astype(np.uint8)

        return warped_image, valid_mask

    def _warp_keypoints(self, points, homography):
        """
        通过单应变换将原始关键点集变换到新的关键点集下
        Args:
            points: 变换前的关键点集合，shape为(n, 2)
            homography: 单应变换矩阵，shape为(3,3)

        Returns:
            new_points: 变换后的关键点集合
        """
        n, _ = points.shape
        if n == 0:
            return points
        points = np.flip(points, axis=1)
        points = np.concatenate((points, np.ones((n, 1))), axis=1)[:, :, np.newaxis]
        new_points = np.matmul(homography, points)[:, :, 0]
        new_points = new_points[:, :2] / new_points[:, 2:]  # 进行归一化
        new_points = new_points[:, ::-1]
        new_points = self._filter_keypoints(new_points)
        return new_points

    def _filter_keypoints(self, points):
        boarder = np.array((self.height-1, self.width-1), dtype=np.float32)
        mask = (points >= 0)&(points <= boarder)
        in_idx = np.nonzero(np.all(mask, axis=1, keepdims=False))[0]
        in_points = []
        for idx in in_idx:
            in_points.append(points[idx])
        if len(in_points) != 0:
            in_points = np.stack(in_points, axis=0)
        else:
            in_points = np.empty((0, 2), np.float32)

        return in_points

    def sample_homography(self):

        pts_1 = np.array(((0, 0), (0, 1), (1, 1), (1, 0)), dtype=np.float)  # 注意这里第一维是x，第二维是y
        margin = (1 - self.patch_ratio) / 2
        pts_2 = margin + np.array(((0, 0), (0, self.patch_ratio),
                                   (self.patch_ratio, self.patch_ratio), (self.patch_ratio, 0)),
                                  dtype=np.float)

        # 进行透视变换
        if self.do_perspective:
            perspective_amplitude_x = self.perspective_amplitude_x
            perspective_amplitude_y = self.perspective_amplitude_y
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
                valid = np.arange(self.scaling_sample_num + 1)
            else:
                valid = np.where(np.all((scaled >= 0.) & (scaled < 1.), axis=(1, 2)))[0]
            idx = valid[np.random.randint(0, valid.shape[0])]
            # 从n_scales个随机的缩放中随机地取一个出来
            pts_2 = scaled[idx]

        # 进行平移变换
        if self.do_translation:
            t_min, t_max = np.min(pts_2, axis=0), np.min(1 - pts_2, axis=0)
            if self.allow_artifacts:
                t_min += self.translation_overflow
                t_max += self.translation_overflow
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
                valid = np.arange(self.rotation_sample_num)
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
        self.brightness_max_abs_change = 25  # 50
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


def draw_image_keypoints(image, points, color=(0, 255, 0)):
    """
    将输入的关键点画到图像上并显示出来
    Args:
        image: 待画点的原始图像
        points: 图像对应的关键点组合，输入为np.array，shape为（n，2）, 点的第一维代表y轴，第二维代表x轴
        color: 待描关键点的颜色
    Returns:
        None
    """
    n, _ = points.shape
    cv_keypoints = []
    for i in range(n):
        keypt = cv.KeyPoint()
        keypt.pt = (points[i, 1], points[i, 0])
        cv_keypoints.append(keypt)
    image = cv.drawKeypoints(image.astype(np.uint8), cv_keypoints, None, color=color)
    # cv.imshow("image&keypoints", image)
    # cv.waitKey()
    return image
