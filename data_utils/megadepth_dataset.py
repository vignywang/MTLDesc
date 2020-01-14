# 
# Created by ZhangYuyang on 2020/1/10
#
import os

import h5py
import cv2 as cv
from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as f
from torch.utils.data import Dataset

# from lib.utils import preprocess_image
from data_utils.dataset_tools import draw_image_keypoints


class MegaDepthDataset(Dataset):
    def __init__(
            self,
            base_path="/data/MegaDepthOrder",
            scene_info_path="/data/MegaDepthOrder/scene_info",
            train=True,
            min_overlap_ratio=.5,
            max_overlap_ratio=1,
            max_scale_ratio=np.inf,
            scenes_ratio=0.5,
            pairs_per_scene=0.5,
            image_size=256
    ):
        self.train = train

        if self.train:
            scene_list_path = os.path.join(base_path, "train_scenes.txt")
        else:
            scene_list_path = os.path.join(base_path, "valid_scenes.txt")

        self.scenes = []
        with open(scene_list_path, 'r') as f:
            lines = f.readlines()
            scenes_num = int(len(lines) * scenes_ratio)
            for i, line in enumerate(lines):
                self.scenes.append(line.strip('\n'))
                # todo
                if i == scenes_num:
                    break

        self.scene_info_path = scene_info_path
        self.base_path = base_path

        self.min_overlap_ratio = min_overlap_ratio
        self.max_overlap_ratio = max_overlap_ratio
        self.max_scale_ratio = max_scale_ratio

        self.pairs_per_scene = pairs_per_scene

        self.image_size = image_size

        self.dataset = []

        self.grid = self._generate_fixed_grid(height=image_size, width=image_size)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image1, image2, desp_point1, desp_point2, valid_mask, not_search_mask = self.recover_pair(self.dataset[idx])

        image1 = (torch.from_numpy(image1).to(torch.float32) * 2. / 255. - 1.).permute((2, 0, 1)).contiguous()
        image2 = (torch.from_numpy(image2).to(torch.float32) * 2. / 255. - 1.).permute((2, 0, 1)).contiguous()

        desp_point1 = torch.from_numpy(self._scale_point_for_sample(desp_point1, self.image_size, self.image_size))
        desp_point2 = torch.from_numpy(self._scale_point_for_sample(desp_point2, self.image_size, self.image_size))

        valid_mask = torch.from_numpy(valid_mask).to(torch.float)
        not_search_mask = torch.from_numpy(not_search_mask).to(torch.float)

        return {
            "image": image1,
            "warped_image": image2,
            "desp_point": desp_point1,
            "warped_desp_point": desp_point2,
            "valid_mask": valid_mask,
            "not_search_mask": not_search_mask,
        }

    def build_dataset(self):
        self.dataset = []
        # if not self.train:
        #     np_random_state = np.random.get_state()
        #     np.random.seed(42)
        #     print('Building the validation dataset...')
        # else:
        #     print('Building a new training dataset...')

        for scene in tqdm(self.scenes, total=len(self.scenes)):
            scene_info_path = os.path.join(
                self.scene_info_path, '%s.0.npz' % scene
            )
            if not os.path.exists(scene_info_path):
                continue
            scene_info = np.load(scene_info_path, allow_pickle=True)
            overlap_matrix = scene_info['overlap_matrix']
            scale_ratio_matrix = scene_info['scale_ratio_matrix']

            valid = np.logical_and(
                np.logical_and(
                    overlap_matrix >= self.min_overlap_ratio,
                    overlap_matrix <= self.max_overlap_ratio
                ),
                scale_ratio_matrix <= self.max_scale_ratio
            )

            pairs = np.vstack(np.where(valid))
            pairs_per_scene = int(self.pairs_per_scene * pairs.shape[1])
            try:
                selected_ids = np.random.choice(
                    pairs.shape[1], pairs_per_scene
                )
            except:
                continue

            image_paths = scene_info['image_paths']
            depth_paths = scene_info['depth_paths']
            points3D_id_to_2D = scene_info['points3D_id_to_2D']
            points3D_id_to_ndepth = scene_info['points3D_id_to_ndepth']
            intrinsics = scene_info['intrinsics']
            poses = scene_info['poses']

            for pair_idx in selected_ids:
                idx1 = pairs[0, pair_idx]
                idx2 = pairs[1, pair_idx]
                matches = np.array(list(
                    points3D_id_to_2D[idx1].keys() &
                    points3D_id_to_2D[idx2].keys()
                ))

                # Scale filtering
                matches_nd1 = np.array([points3D_id_to_ndepth[idx1][match] for match in matches])
                matches_nd2 = np.array([points3D_id_to_ndepth[idx2][match] for match in matches])
                scale_ratio = np.maximum(matches_nd1 / matches_nd2, matches_nd2 / matches_nd1)
                matches = matches[np.where(scale_ratio <= self.max_scale_ratio)[0]]

                point3D_id = np.random.choice(matches)
                point2D1 = points3D_id_to_2D[idx1][point3D_id]
                point2D2 = points3D_id_to_2D[idx2][point3D_id]
                nd1 = points3D_id_to_ndepth[idx1][point3D_id]
                nd2 = points3D_id_to_ndepth[idx2][point3D_id]
                central_match = np.array([
                    point2D1[1], point2D1[0],
                    point2D2[1], point2D2[0]
                ])
                self.dataset.append({
                    'image_path1': image_paths[idx1],
                    'depth_path1': depth_paths[idx1],
                    'intrinsics1': intrinsics[idx1],
                    'pose1': poses[idx1],
                    'image_path2': image_paths[idx2],
                    'depth_path2': depth_paths[idx2],
                    'intrinsics2': intrinsics[idx2],
                    'pose2': poses[idx2],
                    'central_match': central_match,
                    'scale_ratio': max(nd1 / nd2, nd2 / nd1)
                })
        np.random.shuffle(self.dataset)
        # if not self.train:
        #     np.random.set_state(np_random_state)

    def recover_pair(self, pair_metadata):
        depth_path1 = os.path.join(
            self.base_path, pair_metadata['depth_path1']
        )
        with h5py.File(depth_path1, 'r') as hdf5_file:
            depth1 = np.array(hdf5_file['/depth'])
        assert (np.min(depth1) >= 0)

        image_path1 = os.path.join(
            self.base_path, pair_metadata['image_path1']
        )
        image1 = cv.imread(image_path1)[:, :, ::-1].copy()
        assert (image1.shape[0] == depth1.shape[0] and image1.shape[1] == depth1.shape[1])
        intrinsics1 = pair_metadata['intrinsics1']
        pose1 = pair_metadata['pose1']

        depth_path2 = os.path.join(
            self.base_path, pair_metadata['depth_path2']
        )
        with h5py.File(depth_path2, 'r') as hdf5_file:
            depth2 = np.array(hdf5_file['/depth'])
        assert (np.min(depth2) >= 0)

        image_path2 = os.path.join(
            self.base_path, pair_metadata['image_path2']
        )
        image2 = cv.imread(image_path2)[:, :, ::-1].copy()
        assert (image2.shape[0] == depth2.shape[0] and image2.shape[1] == depth2.shape[1])
        intrinsics2 = pair_metadata['intrinsics2']
        pose2 = pair_metadata['pose2']

        central_match = pair_metadata['central_match']
        image1, bbox1, image2, bbox2 = self.crop(image1, image2, central_match)

        depth1 = depth1[
                 bbox1[0]: bbox1[0] + self.image_size,
                 bbox1[1]: bbox1[1] + self.image_size
                 ]
        depth2 = depth2[
                 bbox2[0]: bbox2[0] + self.image_size,
                 bbox2[1]: bbox2[1] + self.image_size
                 ]

        # 生成随机点用于训练时采样描述子
        desp_point1 = self._random_sample_point(self.grid)
        desp_point2, valid_mask, not_search_mask = self._generate_warped_point(
            desp_point1, depth1, bbox1, pose1, intrinsics1, depth2, bbox2, pose2, intrinsics2
        )

        # debug use
        # image_point1 = draw_image_keypoints(image1[:, :, ::-1], desp_point1[valid_mask][:, ::-1], show=False)
        # image_point2 = draw_image_keypoints(image2[:, :, ::-1], desp_point2[valid_mask][:, ::-1], show=False)
        # cat_all = np.concatenate((image_point, warped_image_point), axis=1)
        # warped_img1 = self._inverse_warp(image2, pose2, bbox2, intrinsics2, depth1, pose1, bbox1, intrinsics1)
        # warped_img2 = self._inverse_warp(image1, pose1, bbox1, intrinsics1, depth2, pose2, bbox2, intrinsics2)
        # cat_warped_img = np.concatenate((warped_img1, warped_img2), axis=1)
        # cat_image = np.concatenate((image1, image2), axis=1)
        # cat_all = np.concatenate((image_point1, image_point2), axis=0)
        # cv.imshow("cat_all", cat_all)
        # cv.waitKey()

        return (
            image1, image2, desp_point1, desp_point2, valid_mask, not_search_mask
        )
        # return (image1, intrinsics1, pose1, bbox1,
        #         image2, intrinsics2, pose2, bbox2)

    def crop(self, image1, image2, central_match):
        bbox1_i = max(int(central_match[0]) - self.image_size // 2, 0)
        if bbox1_i + self.image_size >= image1.shape[0]:
            bbox1_i = image1.shape[0] - self.image_size
        bbox1_j = max(int(central_match[1]) - self.image_size // 2, 0)
        if bbox1_j + self.image_size >= image1.shape[1]:
            bbox1_j = image1.shape[1] - self.image_size

        bbox2_i = max(int(central_match[2]) - self.image_size // 2, 0)
        if bbox2_i + self.image_size >= image2.shape[0]:
            bbox2_i = image2.shape[0] - self.image_size
        bbox2_j = max(int(central_match[3]) - self.image_size // 2, 0)
        if bbox2_j + self.image_size >= image2.shape[1]:
            bbox2_j = image2.shape[1] - self.image_size

        return (
            image1[
            bbox1_i: bbox1_i + self.image_size,
            bbox1_j: bbox1_j + self.image_size
            ],
            np.array([bbox1_i, bbox1_j]),
            image2[
            bbox2_i: bbox2_i + self.image_size,
            bbox2_j: bbox2_j + self.image_size
            ],
            np.array([bbox2_i, bbox2_j])
        )

    @staticmethod
    def _random_sample_point(grid):
        """
        根据预设的输入图像大小，随机均匀采样坐标点

        Args:
            grid: [n,4] 第1维的分量分别表示，grid_y_start, grid_x_start, grid_y_end, grid_x_end
        Returns:
            point: [n,2] 顺序为x,y
        """
        point_list = []
        for i in range(grid.shape[0]):
            y_start, x_start, y_end, x_end = grid[i]
            rand_y = np.random.randint(y_start, y_end)
            rand_x = np.random.randint(x_start, x_end)
            point_list.append(np.array((rand_x, rand_y), dtype=np.float32))
        point = np.stack(point_list, axis=0)

        return point

    @staticmethod
    def _generate_fixed_grid(height, width, x_num=20, y_num=20):
        """
        预先采样固定数目的图像格子

        Returns:
            grid: [n,4] 第1维的分量分别表示，grid_y_start, grid_x_start, grid_y_end, grid_x_end
        """
        grid_y = np.linspace(0, height-1, y_num+1, dtype=np.int)
        grid_x = np.linspace(0, width-1, x_num+1, dtype=np.int)

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

    def _generate_warped_point(
            self,
            src_point,
            src_depth,
            src_bbox,
            src_pose,
            src_intrinsics,
            tgt_depth,
            tgt_bbox,
            tgt_pose,
            tgt_intrinsics,
    ):
        """
        tgt_point = K*tgt_pose*src_pose^-1*K^-1*(src_point+src_bbox)-tgt_bbox
        有bbox是因为这些点来自预先裁减的图像块，而深度及姿态等定义在原图像上

        Args:
            src_point: [n,2] 顺序为x,y的点
            src_depth: [h,w] 表示每个整数点处的深度值
            src_bbox: [2,] 表示图像块的起始位置
            src_pose: 从世界坐标系到src坐标系的外参矩阵
            src_intrinsics: 图像块对于的内参矩阵
            tgt_depth: [h,w] 表示目标图像块的每个整数点处的深度值，用于和投影点深度对比以排除不准确的点
            tgt_bbox: 目标图像块的起始位置
            tgt_pose: 从世界坐标系到tgt坐标系的外参矩阵
            tgt_intrinsics: 目标图像的内参矩阵

        Returns:
            tgt_point:投影点
            valid_mask: 对于每一对投影点的有效性
            not_search_mask: 标记训练时不用于搜索的负样本
        """
        # 插值这些点的深度值
        src_point_depth, src_point_valid = self.interpolate_depth(src_point, src_depth)
        src_point_depth = np.where(src_point_depth == 0, np.ones_like(src_point_depth), src_point_depth)

        # 将点进行投影
        src_point = src_point + src_bbox[::-1]
        src_point = np.concatenate((src_point, np.ones((src_point.shape[0], 1), np.float32)), axis=1)[:, :, np.newaxis]  # [n,3,1]
        src_point = np.matmul(np.linalg.inv(src_intrinsics),  src_point)[:, :, 0]  # 得到归一化平面上的点
        src_point = src_point_depth[:, np.newaxis] * src_point  # 得到相机坐标系下的三维点位置
        src_point = np.concatenate((src_point, np.ones((src_point.shape[0], 1), np.float32)), axis=1)[:, :, np.newaxis]  # [n,4,1]

        tgt_point = np.matmul(tgt_pose, np.matmul(np.linalg.inv(src_pose), src_point))  # [n,4,1]
        tgt_point = tgt_point[:, :3, :] / tgt_point[:, 3:4, :]
        tgt_point_depth_estimate = tgt_point[:, 2, 0].astype(np.float32)  # [n,]
        tgt_point = np.matmul(tgt_intrinsics, tgt_point)
        tgt_point = tgt_point[:, :2, 0] / tgt_point[:, 2:3, 0]  # [n,2]
        tgt_point = (tgt_point - tgt_bbox[::-1]).astype(np.float32)

        # 投影点没有深度，或投影深度与标记深度相差过大的无效
        tgt_point_depth_annotate, tgt_point_valid = self.interpolate_depth(tgt_point, tgt_depth)
        inlier_mask = np.abs(tgt_point_depth_estimate - tgt_point_depth_annotate) < 0.5
        valid_mask = src_point_valid & tgt_point_valid & inlier_mask

        # 构造not_search_mask, 太近了以及无效点不会作为负样本的搜索点
        invalid_mask = ~valid_mask
        dist = np.linalg.norm((tgt_point[:, np.newaxis, :]-tgt_point[np.newaxis, :, :]), axis=2)
        not_search_mask = (dist <= 16) | (invalid_mask[np.newaxis, :])

        return tgt_point, valid_mask, not_search_mask

    @staticmethod
    def _scale_point_for_sample(point, height, width):
        """
        将点归一化到[-1,1]的区间范围内，方便采样
        Args:
            point: [n,2] x,y的顺序，原始范围为[0,width-1], [0,height-1]
        Returns:
            point: [n,1,2] x,y的顺序，范围为[-1,1]
        """
        org_size = np.array((width-1, height-1), dtype=np.float32)
        point = (point * 2. / org_size - 1.)[:, np.newaxis, :].copy()
        return point

    @staticmethod
    def _inverse_warp(src_image, src_pose, src_bbox, src_intrinscis, tgt_depth, tgt_pose, tgt_bbox, tgt_intrinsics):
        """
        通过将tgt上的像素点位置投影到src上，并用src的图像像素值合成该位置处的像素
        Args:
            src_pose: 源图像在世界坐标中的姿态
            tgt_pose: 目标图像在世界坐标中的姿态
            tgt_bbox: 图像块在原始图像上的起始位置
        """
        height, width = tgt_depth.shape
        # 记录下没有深度值的点
        tgt_mask = (tgt_depth == 0)

        # 将没有深度值的点赋值为1
        tgt_depth = np.where(tgt_depth == 0, np.ones_like(tgt_depth), tgt_depth).reshape((-1,))

        # 生成待投影的坐标值, u为图像上x方向坐标，v为图像y方向坐标
        tgt_u = np.tile(np.arange(0, width, dtype=np.float32)[np.newaxis, :], (height, 1)).reshape((-1)) + tgt_bbox[1]  # [h*w]
        tgt_v = np.tile(np.arange(0, height, dtype=np.float32)[:, np.newaxis], (1, width)).reshape((-1)) + tgt_bbox[0]

        tgt_x = (tgt_u - tgt_intrinsics[0, 2]) * (tgt_depth / tgt_intrinsics[0, 0])
        tgt_y = (tgt_v - tgt_intrinsics[1, 2]) * (tgt_depth / tgt_intrinsics[1, 1])
        tgt_z = tgt_depth

        # 通过姿态变换到src坐标系下
        tgt_coords = np.stack((tgt_x, tgt_y, tgt_z, np.ones_like(tgt_x)), axis=1)[:, :, np.newaxis]  # [h*w,4,1]齐次坐标
        src_coords = np.matmul(src_pose, np.matmul(np.linalg.inv(tgt_pose), tgt_coords))[:, :, 0]
        src_coords = src_coords[:, :3] / src_coords[:, 3:4]

        # 得到src图像坐标系下的x,y坐标
        src_x = src_intrinscis[0, 0] * src_coords[:, 0] / src_coords[:, 2] + src_intrinscis[0, 2] - src_bbox[1]
        src_y = src_intrinscis[1, 1] * src_coords[:, 1] / src_coords[:, 2] + src_intrinscis[1, 2] - src_bbox[0]
        src_pt = np.stack((src_x, src_y), axis=1).astype(np.float32)

        # 将待采样的点归一化到[-1,1]之间
        src_pt = src_pt * 2. / np.array((width-1, height-1), dtype=np.float32) - 1.
        src_pt = torch.from_numpy(src_pt).reshape((1, height, width, 2))

        # 用src图像块进行双线性插值
        img_torch = torch.from_numpy(src_image).to(torch.float).unsqueeze(dim=0).permute((0, 3, 1, 2)).contiguous()
        img_warped = f.grid_sample(img_torch, src_pt, mode="bilinear").floor().to(torch.uint8)
        img_warped = img_warped.permute((0, 2, 3, 1)).contiguous().numpy()[0]  # [h,w,3]
        img_warped = np.where(tgt_mask[:, :, np.newaxis], np.zeros_like(img_warped), img_warped)

        return img_warped

    @staticmethod
    def interpolate_depth(point, depth):
        """
        插值浮点型处的坐标点的深度值
        Args:
            point: [n,2] x,y的顺序
            depth: [h,w] 0表示没有深度值

        Returns:
            point_depth: [n,] 对应每个点的深度值
            valid_mask: [n,] 对应每个点深度值的有效性

        """
        point = torch.from_numpy(point)
        depth = torch.from_numpy(depth)

        h, w = depth.shape
        num, _ = point.shape

        i = point[:, 1]
        j = point[:, 0]

        # Valid point, 图像范围内的点为有效点
        i_top_left = torch.floor(i).long()
        j_top_left = torch.floor(j).long()
        valid_top_left = (i_top_left >= 0) & (j_top_left >= 0)

        i_top_right = torch.floor(i).long()
        j_top_right = torch.ceil(j).long()
        valid_top_right = (i_top_right >= 0) & (j_top_right <= w-1)

        i_bottom_left = torch.ceil(i).long()
        j_bottom_left = torch.floor(j).long()
        valid_bottom_left = (i_bottom_left <= h-1) & (j_bottom_left >= 0)

        i_bottom_right = torch.ceil(i).long()
        j_bottom_right = torch.ceil(j).long()
        valid_bottom_right = (i_bottom_right <= h-1) & (j_bottom_right <= w-1)

        valid_point = valid_top_left & valid_top_right & valid_bottom_left & valid_bottom_right

        # 保证周围4个点都在图像范围内
        i_top_left = i_top_left.clamp(0, h-1)
        i_top_right = i_top_right.clamp(0, h-1)
        i_bottom_left = i_bottom_left.clamp(0, h-1)
        i_bottom_right = i_bottom_right.clamp(0, h-1)

        j_top_left = j_top_left.clamp(0, w-1)
        j_top_right = j_top_right.clamp(0, w-1)
        j_bottom_left = j_bottom_left.clamp(0, w-1)
        j_bottom_right = j_bottom_right.clamp(0, w-1)

        # Valid depth 有效深度指其上下左右四个邻近整型点都存在有效深度
        valid_depth = (depth[i_top_left, j_top_left] > 0)
        valid_depth &= (depth[i_top_right, j_top_right] > 0)
        valid_depth &= (depth[i_bottom_left, j_bottom_left] > 0)
        valid_depth &= (depth[i_bottom_right, j_bottom_right] > 0)

        # 有效点和有效深度组成最终的有效掩膜
        valid_mask = valid_point & valid_depth

        # 将point归一化到[-1,1]之间用于grid_sample计算
        point = point * 2. / torch.tensor((w-1, h-1), dtype=torch.float) - 1.
        point = torch.reshape(point, (1, num, 1, 2))
        depth = torch.reshape(depth, (1, 1, h, w))
        point_depth = f.grid_sample(depth, point, mode="bilinear")[0, 0, :, 0].contiguous().numpy()  # [n]
        valid_mask = valid_mask.numpy()

        return point_depth, valid_mask


if __name__ == "__main__":
    np.random.seed(1234)
    dataset = MegaDepthDataset(
        base_path="/data/MegaDepthOrder",
        scene_info_path="/data/MegaDepthOrder/scene_info",
    )
    dataset.build_dataset()
    for i, data in enumerate(dataset):
        a = 3


