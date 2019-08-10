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
                      allow_artifacts=False, translation_overflow=0.):
    """
    该函数主要用来生成数据增强时所需的在一定参数范围内的单应变换
    Args:
        height:
        width:
        perspective:
        scaling:
        rotation:
        translation:
        n_scales:
        n_angles:
        scaling_amplitude:
        perspective_amplitude_x:
        perspective_amplitude_y:
        patch_ratio:
        max_angle:
        allow_artifacts:
        translation_overflow:

    Returns:
        homography: 在指定范围内采样的单应变换

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

        # debug use
        # y_displacement = 0.1
        # x_displacement_left = 0.1
        # x_displacement_right = 0.1
        pts_2 += np.array(((x_displacement_left, y_displacement),
                           (x_displacement_left, -y_displacement),
                           (x_displacement_right, y_displacement),
                           (x_displacement_right, -y_displacement)))

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





















