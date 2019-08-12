#
# Created by ZhangYuyang on 2019/8/9
#

import os
import glob
import cv2 as cv
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


class SyntheticTrainDataset(Dataset):

    def __init__(self, params):
        self.params = params
        self.height = params.height
        self.width = params.width
        self.homography_params = params.homography_params
        self.dataset_dir = params.synthetic_dataset_dir
        self.image_list, self.point_list = self._format_file_list()

    def __len__(self):
        assert len(self.image_list) == len(self.point_list)
        return len(self.image_list)

    def __getitem__(self, idx):

        image = cv.imread(self.image_list[idx], flags=cv.IMREAD_GRAYSCALE)
        point = np.load(self.point_list[idx])
        debug_show_image_keypoints(image, point)
        if self.params.do_augmentation:
            homography = sample_homography(**self.homography_params)
            image = cv.warpPerspective(image, homography, (self.width, self.height), flags=cv.INTER_LINEAR)
            point = warp_keypoints(point, homography)
        debug_show_image_keypoints(image, point)
        sample = {"image": image, "point": point}
        return sample

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
    new_points = new_points[:, :2]/new_points[:, 2:]  # 进行归一化
    new_points = new_points[:, ::-1]
    return new_points


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
    image = cv.drawKeypoints(image, cv_keypoints, None)
    cv.imshow("image&keypoints", image)
    cv.waitKey()


def sample_homography(height=240, width=320, perspective=True, scaling=True, rotation=True, translation=True,
                      n_scales=5, n_angles=25, scaling_amplitude=0.1, perspective_amplitude_x=0.1,
                      perspective_amplitude_y=0.1, patch_ratio=0.5, max_angle=np.pi / 2,
                      allow_artifacts=False):
    """
    该函数主要用来生成数据增强时所需的随机采样的透视、尺度、平移、旋转等组合的单应变换
    Args:
        height: int, 待采样图像的高
        width: int, 待采样图像的宽
        perspective: bool, 是否进行投影变换
        scaling: bool, 是否进行尺度变换
        rotation: bool, 是否进行旋转变换
        translation: bool, 是否进行平移变换
        n_scales: int, 表示随机生成的尺度数量，最终从其中再随机挑选一个
        n_angles: int, 表示随机生成的旋转角度数量，最终从其中再随机挑选一个
        scaling_amplitude: float, 表示尺度变换幅值
        perspective_amplitude_x: float, 表示投影变换x方向上的变换幅值
        perspective_amplitude_y: float, 表示投影变换y方向上的变换幅值
        patch_ratio: float, 表示用于仿射变换的图像块占原始图像块的比例
        max_angle: float, 表示旋转角度的最大采样值
        allow_artifacts: bool, 表示是否允许变换超出图像范围

    Returns:
        homography: 在指定范围内随机采样的单应变换

    """
    pts_1 = np.array(((0, 0), (0, 1), (1, 1), (1, 0)), dtype=np.float)  # 注意这里第一维是x，第二维是y
    margin = (1 - patch_ratio) / 2
    pts_2 = margin + np.array(((0, 0), (0, patch_ratio), (patch_ratio, patch_ratio), (patch_ratio, 0)), dtype=np.float)

    # 进行透视变换
    if perspective:
        if not allow_artifacts:
            perspective_amplitude_x = min(perspective_amplitude_x, margin)
            perspective_amplitude_y = min(perspective_amplitude_y, margin)
        y_displacement = np.random.uniform(-perspective_amplitude_y, perspective_amplitude_y)
        x_displacement_left = np.random.uniform(-perspective_amplitude_x, perspective_amplitude_x)
        x_displacement_right = np.random.uniform(-perspective_amplitude_x, perspective_amplitude_x)

        pts_2 += np.array(((x_displacement_left, y_displacement),
                           (x_displacement_left, -y_displacement),
                           (x_displacement_right, y_displacement),
                           (x_displacement_right, -y_displacement)))

        x_displacement = np.random.uniform(-perspective_amplitude_x, perspective_amplitude_x)
        y_displacement_up = np.random.uniform(-perspective_amplitude_y, perspective_amplitude_y)
        y_displacement_bottom = np.random.uniform(-perspective_amplitude_y, perspective_amplitude_y)

        pts_2 += np.array(((x_displacement, y_displacement_up),
                           (-x_displacement, y_displacement_bottom),
                           (x_displacement, y_displacement_bottom),
                           (-x_displacement, y_displacement_up)))

    # 进行尺度变换
    if scaling:
        # 得到n+1个尺度参数，其中最后一个为1，即不进行尺度化
        scales = np.concatenate((np.random.normal(1, scaling_amplitude, (n_scales,)), np.ones((1,))), axis=0)
        # 中心点不变的尺度缩放
        center = np.mean(pts_2, axis=0, keepdims=True)
        scaled = np.expand_dims(pts_2 - center, axis=0) * np.expand_dims(np.expand_dims(scales, 1), 1) + center
        if allow_artifacts:
            valid = np.arange(n_scales)  # all scales are valid except scale=1
        else:
            valid = np.where(np.all((scaled >= 0.) & (scaled < 1.), axis=(1, 2)))[0]
        idx = valid[np.random.randint(0, valid.shape[0])]
        # 从n_scales个随机的缩放中随机地取一个出来
        pts_2 = scaled[idx]

    # 进行平移变换
    if translation:
        t_min, t_max = np.min(pts_2, axis=0), np.min(1 - pts_2, axis=0)
        pts_2 += np.expand_dims(np.stack((np.random.uniform(-t_min[0], t_max[0]),
                                          np.random.uniform(-t_min[1], t_max[1]))), axis=0)

    if rotation:
        angles = np.linspace(-max_angle, max_angle, n_angles)
        angles = np.concatenate((angles, np.zeros((1,))), axis=0)  # in case no rotation is valid
        center = np.mean(pts_2, axis=0, keepdims=True)
        rot_mat = np.reshape(np.stack((np.cos(angles), -np.sin(angles),
                                       np.sin(angles), np.cos(angles)), axis=1), newshape=(-1, 2, 2))
        # [x, y] * | cos -sin|
        #          | sin  cos|
        rotated = np.matmul(
            np.tile(np.expand_dims(pts_2 - center, axis=0), reps=(n_angles+1, 1, 1)), rot_mat
        ) + center
        if allow_artifacts:
            valid = np.arange(n_angles)  # all angles are valid, except angle=0
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
            coefficient_mat[i+4, :] = [0, 0, 0, p[i][0], p[i][1], 1, -p[i][0] * q[i][1], -p[i][1] * q[i][1]]
        return coefficient_mat

    # 将矩形以及变换后的四边形坐标还原到实际尺度上，并计算他们之间对应的单应变换
    size = np.array((width-1, height-1), dtype=np.float)
    pts_1 *= size
    pts_2 *= size

    a_mat = mat(pts_1, pts_2)
    b_mat = np.concatenate(np.split(pts_2, 2, axis=1), axis=0)
    homography = np.linalg.lstsq(a_mat, b_mat, None)[0].reshape((8,))
    homography = np.concatenate((homography, np.ones((1,))), axis=0).reshape((3, 3))

    return homography


if __name__ == "__main__":

    class Parameters:
        synthetic_dataset_dir = "/data/MegPoint/dataset/synthetic"

        height = 240
        width = 320
        do_augmentation = True
        homography_params = {
            'translation': True,
            'rotation': True,
            'scaling': True,
            'perspective': True,
            'scaling_amplitude': 0.2,
            'perspective_amplitude_x': 0.2,
            'perspective_amplitude_y': 0.2,
            'patch_ratio': 0.8,
            'max_angle': 1.57,  # 3.14
            'allow_artifacts': False,
            'translation_overflow': 0.0,
        }
    params = Parameters()
    synthetic_dataset = SyntheticTrainDataset(params=params)
    for i, data in enumerate(synthetic_dataset):
        if i == 3:
            break





















