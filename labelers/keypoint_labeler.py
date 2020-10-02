import h5py
import os
import time
import cv2 as cv
import torch
import torch.nn.functional as f
from tqdm import tqdm
import numpy as np
from nets.superpoint_net import SuperPointNetFloat
from data_utils.dataset_tools import HomographyAugmentation
from utils.utils import spatial_nms
class TrainDatasetCreator(object):
    """构造数据集：图像对，关键点，描述子采样对"""
    def __init__(self,**config):
        self.config = config
        self.train = self.config['train']
        if torch.cuda.is_available():
            print('gpu is available, set device to cuda!')
            self.device = torch.device('cuda:0')
        else:
            print('gpu is not available, set device to cpu!')
            self.device = torch.device('cpu')
        scene_list_path = os.path.join(self.config['scenes_path'], "all_scenes.txt")
        self.output_keypoint = self.config['output_keypoint']
        if not os.path.exists(self.output_keypoint):
            os.mkdir(self.output_keypoint)
        self.output_despoint = self.config['output_despoint']
        if not os.path.exists(self.output_despoint):
            os.mkdir(self.output_despoint)
        self.output_image = self.config['output_image']
        if not os.path.exists(self.output_image):
            os.mkdir(self.output_image)

        self.scenes = []
        with open(scene_list_path, 'r') as f:
            lines = f.readlines()
            scenes_num = int(len(lines) *self.config['scenes_ratio'])
            for i, line in enumerate(lines):
                self.scenes.append(line.strip('\n'))
                # todo
                if i == scenes_num:
                    break
        self.max_scale_ratio = np.inf
        self.scene_info_path = self.config['scene_info_path']
        self.base_path = self.config['base_path']
        self.min_overlap_ratio = float(self.config['min_overlap_ratio'])
        self.max_overlap_ratio = self.config['max_overlap_ratio']
        self.pairs_per_scene = self.config['pairs_per_scene']
        self.image_height = self.config['image_height']
        self.image_width = self.config['image_width']
        self.homography = HomographyAugmentation()
        # 初始化模型
        model = SuperPointNetFloat()
        # 从预训练的模型中恢复参数
        model_dict = model.state_dict()
        pretrain_dict = torch.load(self.config['ckpt_path'], map_location=self.device)
        model_dict.update(pretrain_dict)
        model.load_state_dict(model_dict)
        model.to(self.device)
        self.model = model
        self.grid = self._generate_fixed_grid(height=240, width=320)
        self.all_points = self._sample_all_point(self.image_height,self.image_width)
    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def build_dataset(self):
        count = 0
       # cur_time = time.time()
        if self.train:
            np.random.seed(4212)
            print('Building a new training dataset...')
        else:
            np.random.seed(8931)
            print('Building the validation dataset...')
        for i, scene in enumerate(self.scenes):
            print("Processing %s scene..." % scene)

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
            # pairs_per_scene = int(self.pairs_per_scene * pairs.shape[1])
            try:
                selected_ids = np.random.choice(
                    pairs.shape[1], self.pairs_per_scene
                )
            except:
                continue
            image_paths = scene_info['image_paths']
            depth_paths = scene_info['depth_paths']
            points3D_id_to_2D = scene_info['points3D_id_to_2D']
            points3D_id_to_ndepth = scene_info['points3D_id_to_ndepth']
            intrinsics = scene_info['intrinsics']
            poses = scene_info['poses']
            for pair_idx in tqdm(selected_ids):
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
                central_match = np.array([
                    point2D1[1], point2D1[0],
                    point2D2[1], point2D2[0]
                ])

                image1, image2, desp_point1, desp_point2, valid_mask, not_search_mask,keypint1,keypint2,raw_desp_point1,raw_desp_point2,depth1,depth2,pose1,pose2,intrinsics1,intrinsics2,bbox1,bbox2=\
                self.get_pair(
                    image_path1=image_paths[idx1],
                    depth_path1=depth_paths[idx1],
                    intrinsics1=intrinsics[idx1],
                    pose1=poses[idx1],
                    image_path2=image_paths[idx2],
                    depth_path2=depth_paths[idx2],
                    intrinsics2=intrinsics[idx2],
                    pose2=poses[idx2],
                    central_match=central_match,
                )

                # 存储数据
                image12 = np.concatenate((image1, image2), axis=1)
                cur_img_path = os.path.join(self.output_image, "%07d.jpg" % count)
                cur_des_path = os.path.join(self.output_despoint, "%07d" % count)
                cur_key_path = os.path.join(self.output_keypoint, "%07d" % count)
                cv.imwrite(cur_img_path, image12)
                np.savez(cur_des_path, desp_point1=desp_point1, desp_point2=desp_point2, valid_mask=valid_mask,
                         not_search_mask=not_search_mask,raw_desp_point1=raw_desp_point1,raw_desp_point2=raw_desp_point2)
                np.savez(cur_key_path, points_0=keypint1, points_1=keypint2,depth1=depth1,depth2=depth2,pose1=pose1,pose2=pose2,intrinsics1=intrinsics1,intrinsics2=intrinsics2,bbox1=bbox1,bbox2=bbox2)

                count += 1

    def get_pair(
            self,
            image_path1,
            depth_path1,
            intrinsics1,
            pose1,
            image_path2,
            depth_path2,
            intrinsics2,
            pose2,
            central_match,
    ):
        depth_path1 = os.path.join(self.base_path, depth_path1)
        with h5py.File(depth_path1, 'r') as hdf5_file:
            depth1 = np.array(hdf5_file['/depth'])
        assert (np.min(depth1) >= 0)

        image_path1 = os.path.join(self.base_path, image_path1)
        image1 = cv.imread(image_path1).copy()
        assert (image1.shape[0] == depth1.shape[0] and image1.shape[1] == depth1.shape[1])

        depth_path2 = os.path.join(self.base_path, depth_path2)
        with h5py.File(depth_path2, 'r') as hdf5_file:
            depth2 = np.array(hdf5_file['/depth'])
        assert (np.min(depth2) >= 0)

        image_path2 = os.path.join(self.base_path, image_path2)
        image2 = cv.imread(image_path2).copy()
        assert (image2.shape[0] == depth2.shape[0] and image2.shape[1] == depth2.shape[1])

        image1, bbox1, image2, bbox2 = self.crop(image1, image2, central_match)

        depth1 = depth1[
                 bbox1[0]: bbox1[0] + self.image_height,
                 bbox1[1]: bbox1[1] + self.image_width
                 ]
        depth2 = depth2[
                 bbox2[0]: bbox2[0] + self.image_height,
                 bbox2[1]: bbox2[1] + self.image_width
                 ]

        #转化为灰度图像，并通过单映变换产生keypoint 和 heatmap
        gray_image1=cv.cvtColor(image1,cv.COLOR_BGR2GRAY)
        gray_image2=cv.cvtColor(image2,cv.COLOR_BGR2GRAY)
        tensor_heatmap1,keypint1=self._label(gray_image1)
        tensor_heatmap2,keypint2=self._label(gray_image2)
        try:
            tensor_heatmap2to1,point_num1= self._tensor_generate_warped_point(self.all_points,depth1, bbox1, pose1,intrinsics1, depth2, bbox2, pose2,intrinsics2,tensor_heatmap2)
            tensor_heatmap1to2,point_num2= self._tensor_generate_warped_point(self.all_points,depth2, bbox2, pose2,intrinsics2, depth1, bbox1, pose1,intrinsics1,tensor_heatmap1)
        except:
            desp_point1 = self._random_sample_point(self.grid)
            desp_point2, valid_mask, not_search_mask = self._generate_warped_point(desp_point1, depth1, bbox1,
                                                                                   pose1, intrinsics1, depth2,
                                                                                   bbox2, pose2, intrinsics2)
            scale_desp_point1 = self._scale_point_for_sample(desp_point1, height=self.image_height,
                                                             width=self.image_width)
            scale_desp_point2 = self._scale_point_for_sample(desp_point2, height=self.image_height,
                                                             width=self.image_width)

            return image1, image2, scale_desp_point1, scale_desp_point2, valid_mask, not_search_mask, keypint1, keypint2, desp_point1, desp_point2,depth1,depth2,pose1,pose2,intrinsics1,intrinsics2,bbox1,bbox2
        if point_num1>=point_num2:
            if self.config['despoint_type']=='random':
                # 生成随机点用于训练时采样描述子
                desp_point1 = self._random_sample_point(self.grid)
            elif self.config['despoint_type']=='heatmap':
                # 使用交叉验证heatmap采样描述子
                heatmap=tensor_heatmap1.cuda(self.device)+tensor_heatmap2to1*10
                desp_point1 = self._generate_grid_point(heatmap).astype(np.float32)
            desp_point2, valid_mask, not_search_mask = self._generate_warped_point(desp_point1, depth1, bbox1,
                                                                                   pose1, intrinsics1, depth2,
                                                                                   bbox2, pose2, intrinsics2)
            scale_desp_point1 = self._scale_point_for_sample(desp_point1, height=self.image_height,
                                                             width=self.image_width)
            scale_desp_point2 = self._scale_point_for_sample(desp_point2, height=self.image_height,
                                                             width=self.image_width)

            return image1, image2, scale_desp_point1, scale_desp_point2, valid_mask, not_search_mask, keypint1, keypint2, desp_point1, desp_point2,depth1,depth2,pose1,pose2,intrinsics1,intrinsics2,bbox1,bbox2
        else:
            if self.config['despoint_type']=='random':
                desp_point2 = self._random_sample_point(self.grid)
            elif self.config['despoint_type'] == 'heatmap':
                heatmap = tensor_heatmap2.cuda(self.device) + tensor_heatmap1to2*10
                desp_point2 = self._generate_grid_point(heatmap).astype(np.float32)
            desp_point1, valid_mask, not_search_mask = self._generate_warped_point(desp_point2, depth2, bbox2,
                                                                               pose2, intrinsics2, depth1,
                                                                               bbox1, pose1, intrinsics1)
            scale_desp_point1 = self._scale_point_for_sample(desp_point1, height=self.image_height,
                                                             width=self.image_width)
            scale_desp_point2 = self._scale_point_for_sample(desp_point2, height=self.image_height,
                                                             width=self.image_width)

            return image2, image1, scale_desp_point2, scale_desp_point1, valid_mask, not_search_mask, keypint2, keypint1, desp_point2, desp_point1,depth2,depth1,pose2,pose1,intrinsics2,intrinsics1,bbox2,bbox1




    def _label(self, image):
        h, w = image.shape
        images = [image]
        inv_homographies = [np.eye(3)]
        for j in range(self.config['adaption_num']):
            homography = self.homography.sample(height=h, width=w)
            inv_homography = np.linalg.inv(homography)
            transformed_image = cv.warpPerspective(image, homography, (w, h), flags=cv.INTER_LINEAR)
            images.append(transformed_image)
            inv_homographies.append(inv_homography)

        # 按32一个batch分块
        batched_image_list = []
        for i in range(int(len(images) / 32) + 1):
            batched_image_list.append(
                torch.from_numpy(np.stack(images[int(i * 32): int((i + 1) * 32)], axis=0)).unsqueeze(dim=1).to(
                    torch.float))

        prob_list = []
        for img in batched_image_list:
            img = img.to(self.device)
            _, _, prob, _ = self.model(img)
            prob_list.append(prob)

        # 将概率图展开为原始图像大小
        prob = torch.cat(prob_list, dim=0)
        prob = f.pixel_shuffle(prob, 8)
        prob = prob.detach().cpu().numpy()[:, 0, :, :]  # [n+1, h, w]

        # 分别得到每个预测的概率图反变换的概率图
        count = np.ones_like(image)
        counts = []
        probs = []
        counts.append(count)
        for j in range(self.config['adaption_num'] + 1):
            transformed_prob = cv.warpPerspective(prob[j], inv_homographies[j], (w, h),
                                                  flags=cv.INTER_LINEAR)
            transformed_count = cv.warpPerspective(count, inv_homographies[j], (w, h),
                                                   flags=cv.INTER_NEAREST)
            probs.append(transformed_prob)
            counts.append(transformed_count)

        probs = np.stack(probs, axis=2)  # [h,w,n+1]
        counts = np.stack(counts, axis=2)
        probs = np.sum(probs, axis=2)
        counts = np.sum(counts, axis=2)
        probs = probs / counts

        torch_probs = torch.from_numpy(probs).unsqueeze(dim=0).unsqueeze(dim=0)
        final_probs = spatial_nms(torch_probs).detach().cpu().numpy()[0, 0]
        satisfied_idx = np.where(final_probs > self.config['detection_threshold'])
        ordered_satisfied_idx = np.argsort(final_probs[satisfied_idx])[::-1]  # 降序
        if len(ordered_satisfied_idx) < self.config['top_k']:
            points = np.stack((satisfied_idx[0][ordered_satisfied_idx],
                               satisfied_idx[1][ordered_satisfied_idx]), axis=1)
        else:
            points = np.stack((satisfied_idx[0][:self.config['top_k']],
                               satisfied_idx[1][:self.config['top_k']]), axis=1)
        return torch_probs,points

    def crop(self, image1, image2, central_match):
        bbox1_i = max(int(central_match[0]) - self.image_height // 2, 0)
        if bbox1_i + self.image_height >= image1.shape[0]:
            bbox1_i = image1.shape[0] - self.image_height
        bbox1_j = max(int(central_match[1]) - self.image_width // 2, 0)
        if bbox1_j + self.image_width >= image1.shape[1]:
            bbox1_j = image1.shape[1] - self.image_width

        bbox2_i = max(int(central_match[2]) - self.image_height // 2, 0)
        if bbox2_i + self.image_height >= image2.shape[0]:
            bbox2_i = image2.shape[0] - self.image_height
        bbox2_j = max(int(central_match[3]) - self.image_width // 2, 0)
        if bbox2_j + self.image_width >= image2.shape[1]:
            bbox2_j = image2.shape[1] - self.image_width

        return (
            image1[
            bbox1_i: bbox1_i + self.image_height,
            bbox1_j: bbox1_j + self.image_width
            ],
            np.array([bbox1_i, bbox1_j]),
            image2[
            bbox2_i: bbox2_i + self.image_height,
            bbox2_j: bbox2_j + self.image_width
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
    def _sample_all_point(height,width):
        """
        根据预设的输入图像大小，采样全部坐标点

        Args:
            grid: [n,4] 第1维的分量分别表示，grid_y_start, grid_x_start, grid_y_end, grid_x_end
        Returns:
            point: [n,2] 顺序为x,y
        """
        point_list = []
        for i in range(height):
            for j in range(width):
                point_list.append(np.array((i, j), dtype=np.float32))
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

    def _generate_grid_point(self, heatmap):
        pooled = f.max_pool2d(heatmap,kernel_size=10, stride=10).repeat(1,100,1,1)
        pooled=f.pixel_shuffle(pooled,10)
        pointmap = torch.where(torch.eq(heatmap, pooled), heatmap, torch.zeros_like(heatmap))
        pointmap=pointmap.squeeze(0).squeeze(0).cpu().numpy()
        satisfied_idx = np.where(pointmap>0)
        ordered_satisfied_idx = np.argsort(pointmap[satisfied_idx])[::-1]  # 降序
        if len(ordered_satisfied_idx) < self.config['top_k']:
            points = np.stack((satisfied_idx[0][ordered_satisfied_idx],
                               satisfied_idx[1][ordered_satisfied_idx]), axis=1)
        else:
            points = np.stack((satisfied_idx[0][:self.config['top_k']],
                               satisfied_idx[1][:self.config['top_k']]), axis=1)
        return points

    def _tensor_NMStop(self, torch_probs):
        final_probs = spatial_nms(torch_probs, kernel_size=self.config['nms_kernel_size']).detach().cpu().numpy()[0, 0]
        satisfied_idx = np.where(final_probs > 0)
        ordered_satisfied_idx = np.argsort(final_probs[satisfied_idx])[::-1]  # 降序
        if len(ordered_satisfied_idx) < self.config['top_k']:
            points = np.stack((satisfied_idx[0][ordered_satisfied_idx],
                               satisfied_idx[1][ordered_satisfied_idx]), axis=1)
        else:
            points = np.stack((satisfied_idx[0][:self.config['top_k']],
                               satisfied_idx[1][:self.config['top_k']]), axis=1)
        return points.astype(np.float32)
    def _NMStop(self,final_probs):
        satisfied_idx = np.where(final_probs > 0)
        ordered_satisfied_idx = np.argsort(final_probs[satisfied_idx])[::-1]  # 降序
        if len(ordered_satisfied_idx) < self.config['top_k']:
            points = np.stack((satisfied_idx[0][ordered_satisfied_idx],
                               satisfied_idx[1][ordered_satisfied_idx]), axis=1)
        else:
            points = np.stack((satisfied_idx[0][:self.config['top_k']],
                               satisfied_idx[1][:self.config['top_k']]), axis=1)
        return points.astype(np.float32)

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

    def _tensor_generate_warped_point(
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
            tensor_heatmap2
    ):
        tensor_heatmap2=tensor_heatmap2.cuda(self.device).double()
        src_point= torch.from_numpy(src_point).cuda(self.device).float()
        depth1 = torch.from_numpy(src_depth).cuda(self.device).float()
        bbox1 = torch.from_numpy(src_bbox).cuda(self.device).float()
        pose1 = torch.from_numpy(src_pose).cuda(self.device).float()
        intrinsics1 = torch.from_numpy(src_intrinsics).cuda(self.device).float()
        depth2 = torch.from_numpy(tgt_depth).cuda(self.device).float()
        bbox2 = torch.from_numpy(tgt_bbox).cuda(self.device).float()
        pose2 = torch.from_numpy(tgt_pose).cuda(self.device).float()
        intrinsics2 = torch.from_numpy(tgt_intrinsics).cuda(self.device).float()
        pos1=src_point.permute(1,0)
        Z1, pos1, ids =  self.interpolate_depth_raw2(pos1, depth1)

        # COLMAP convention
        u1 = pos1[1, :] + bbox1[1] + .5
        v1 = pos1[0, :] + bbox1[0] + .5

        X1 = (u1 - intrinsics1[0, 2]) * (Z1 / intrinsics1[0, 0])
        Y1 = (v1 - intrinsics1[1, 2]) * (Z1 / intrinsics1[1, 1])

        XYZ1_hom = torch.cat([
            X1.view(1, -1),
            Y1.view(1, -1),
            Z1.view(1, -1),
            torch.ones(1, Z1.size(0), device= self.device)
        ], dim=0)
        XYZ2_hom = torch.chain_matmul(pose2, torch.inverse(pose1), XYZ1_hom)
        XYZ2 = XYZ2_hom[: -1, :] / XYZ2_hom[-1, :].view(1, -1)

        uv2_hom = torch.matmul(intrinsics2, XYZ2)
        uv2 = uv2_hom[: -1, :] / uv2_hom[-1, :].view(1, -1)

        u2 = uv2[0, :] - bbox2[1] - .5
        v2 = uv2[1, :] - bbox2[0] - .5
        uv2 = torch.cat([u2.view(1, -1), v2.view(1, -1)], dim=0)
        annotated_depth, pos2, new_ids = self.interpolate_depth_raw2(self.uv_to_pos(uv2), depth2)

        ids = ids[new_ids]
        pos1 = pos1[:, new_ids]
        estimated_depth = XYZ2[2, new_ids]

        inlier_mask = torch.abs(estimated_depth - annotated_depth) < 100

        ids = ids[inlier_mask]
        if ids.size(0) == 0:
            raise EmptyTensorError
        #vaild mask
        vaild_mask= torch.zeros(240*320)
        ones = torch.ones(ids.size(0))
        vaild_mask[ids] = ones
        vaild_mask=vaild_mask.view(240,320).cuda(self.device).double()
        # 采样heatmap
        pos2 = pos2[:, inlier_mask]
        pos1 = pos1[:, inlier_mask]
        pos_index=pos1.permute(1,0)
        pos_value = pos2.permute(1, 0)
        pos_index=pos_index.cpu().numpy().astype(np.int)
        pos_value = pos_value * 2. / torch.tensor((239, 319), dtype=torch.float, device=self.device) - 1
        pos_value=pos_value.cpu().numpy()[..., ::-1] # y,x to x,y
        map=np.zeros((240,320,2))
        map[tuple(np.transpose(pos_index))] = pos_value
        grid=torch.from_numpy(map).cuda(self.device).unsqueeze(0)
        heatmap2=f.grid_sample(tensor_heatmap2, grid, mode="bilinear", padding_mode="border")
        heatmap2=heatmap2*vaild_mask
        #heatmap2=heatmap2.squeeze(0).squeeze(0)*vaild_mask
        #heatmap2=heatmap2.cpu().numpy().astype(np.uint8)

        return heatmap2,len(ids)

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

    def interpolate_depth(self,point, depth):
        """
        插值浮点型处的坐标点的深度值
        Args:
            point: [n,2] x,y的顺序
            depth: [h,w] 0表示没有深度值

        Returns:
            point_depth: [n,] 对应每个点的深度值
            valid_mask: [n,] 对应每个点深度值的有效性

        """
        point = torch.from_numpy(point).cuda(self.device)
        depth = torch.from_numpy(depth).cuda(self.device)

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
        point = point * 2. / torch.tensor((w-1, h-1), dtype=torch.float).cuda(self.device) - 1.
        point = torch.reshape(point, (1, num, 1, 2))
        depth = torch.reshape(depth, (1, 1, h, w))
        point_depth = f.grid_sample(depth, point, mode="bilinear")[0, 0, :, 0].contiguous().cpu().numpy()  # [n]
        valid_mask = valid_mask.cpu().numpy()

        return point_depth, valid_mask

    def interpolate_depth2(self,point, depth):
        """
        插值浮点型处的坐标点的深度值
        Args:
            point: [n,2] x,y的顺序
            depth: [h,w] 0表示没有深度值

        Returns:
            point_depth: [n,] 对应每个点的深度值
            valid_mask: [n,] 对应每个点深度值的有效性

        """
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
        valid_top_right = (i_top_right >= 0) & (j_top_right <= w - 1)

        i_bottom_left = torch.ceil(i).long()
        j_bottom_left = torch.floor(j).long()
        valid_bottom_left = (i_bottom_left <= h - 1) & (j_bottom_left >= 0)

        i_bottom_right = torch.ceil(i).long()
        j_bottom_right = torch.ceil(j).long()
        valid_bottom_right = (i_bottom_right <= h - 1) & (j_bottom_right <= w - 1)

        valid_point = valid_top_left & valid_top_right & valid_bottom_left & valid_bottom_right

        # 保证周围4个点都在图像范围内
        i_top_left = i_top_left.clamp(0, h - 1)
        i_top_right = i_top_right.clamp(0, h - 1)
        i_bottom_left = i_bottom_left.clamp(0, h - 1)
        i_bottom_right = i_bottom_right.clamp(0, h - 1)

        j_top_left = j_top_left.clamp(0, w - 1)
        j_top_right = j_top_right.clamp(0, w - 1)
        j_bottom_left = j_bottom_left.clamp(0, w - 1)
        j_bottom_right = j_bottom_right.clamp(0, w - 1)

        # Valid depth 有效深度指其上下左右四个邻近整型点都存在有效深度
        valid_depth = (depth[i_top_left, j_top_left] > 0)
        valid_depth &= (depth[i_top_right, j_top_right] > 0)
        valid_depth &= (depth[i_bottom_left, j_bottom_left] > 0)
        valid_depth &= (depth[i_bottom_right, j_bottom_right] > 0)

        # 有效点和有效深度组成最终的有效掩膜
        valid_mask = valid_point & valid_depth

        # 将point归一化到[-1,1]之间用于grid_sample计算
        point = point * 2. / torch.tensor((w - 1, h - 1), dtype=torch.float).cuda(self.device) - 1.
        point = torch.reshape(point, (1, num, 1, 2))
        depth = torch.reshape(depth, (1, 1, h, w))
        point_depth = f.grid_sample(depth, point, mode="bilinear")[0, 0, :, 0].contiguous() # [n]
        valid_mask = valid_mask

        return point_depth, valid_mask

    def interpolate_depth_raw2(self, pos, depth):
        ids = torch.arange(0, pos.size(1), device=self.device)
        h, w = depth.size()

        i = pos[0, :]
        j = pos[1, :]

        # Valid corners
        i_top_left = torch.floor(i).long()
        j_top_left = torch.floor(j).long()
        valid_top_left = torch.min(i_top_left >= 0, j_top_left >= 0)

        i_top_right = torch.floor(i).long()
        j_top_right = torch.ceil(j).long()
        valid_top_right = torch.min(i_top_right >= 0, j_top_right < w)

        i_bottom_left = torch.ceil(i).long()
        j_bottom_left = torch.floor(j).long()
        valid_bottom_left = torch.min(i_bottom_left < h, j_bottom_left >= 0)

        i_bottom_right = torch.ceil(i).long()
        j_bottom_right = torch.ceil(j).long()
        valid_bottom_right = torch.min(i_bottom_right < h, j_bottom_right < w)

        valid_corners = torch.min(
            torch.min(valid_top_left, valid_top_right),
            torch.min(valid_bottom_left, valid_bottom_right)
        )

        i_top_left = i_top_left[valid_corners]
        j_top_left = j_top_left[valid_corners]

        i_top_right = i_top_right[valid_corners]
        j_top_right = j_top_right[valid_corners]

        i_bottom_left = i_bottom_left[valid_corners]
        j_bottom_left = j_bottom_left[valid_corners]

        i_bottom_right = i_bottom_right[valid_corners]
        j_bottom_right = j_bottom_right[valid_corners]

        ids = ids[valid_corners]

        # Valid depth
        valid_depth = torch.min(
            torch.min(
                depth[i_top_left, j_top_left] > 0,
                depth[i_top_right, j_top_right] > 0
            ),
            torch.min(
                depth[i_bottom_left, j_bottom_left] > 0,
                depth[i_bottom_right, j_bottom_right] > 0
            )
        )

        i_top_left = i_top_left[valid_depth]
        j_top_left = j_top_left[valid_depth]

        i_top_right = i_top_right[valid_depth]
        j_top_right = j_top_right[valid_depth]

        i_bottom_left = i_bottom_left[valid_depth]
        j_bottom_left = j_bottom_left[valid_depth]

        i_bottom_right = i_bottom_right[valid_depth]
        j_bottom_right = j_bottom_right[valid_depth]

        ids = ids[valid_depth]

        # Interpolation
        i = i[ids]
        j = j[ids]
        dist_i_top_left = i - i_top_left.float()
        dist_j_top_left = j - j_top_left.float()
        w_top_left = (1 - dist_i_top_left) * (1 - dist_j_top_left)
        w_top_right = (1 - dist_i_top_left) * dist_j_top_left
        w_bottom_left = dist_i_top_left * (1 - dist_j_top_left)
        w_bottom_right = dist_i_top_left * dist_j_top_left

        interpolated_depth = (
                w_top_left * depth[i_top_left, j_top_left] +
                w_top_right * depth[i_top_right, j_top_right] +
                w_bottom_left * depth[i_bottom_left, j_bottom_left] +
                w_bottom_right * depth[i_bottom_right, j_bottom_right]
        )

        pos = torch.cat([i.view(1, -1), j.view(1, -1)], dim=0)

        return [interpolated_depth, pos, ids]
    @staticmethod
    def uv_to_pos(uv):
        return torch.cat([uv[1, :].view(1, -1), uv[0, :].view(1, -1)], dim=0)

def grid_positions(h, w, device, matrix=False):
    lines = torch.arange(
        0, h, device=device
    ).view(-1, 1).float().repeat(1, w)
    columns = torch.arange(
        0, w, device=device
    ).view(1, -1).float().repeat(h, 1)
    if matrix:
        return torch.stack([lines, columns], dim=0)
    else:
        return torch.cat([lines.view(1, -1), columns.view(1, -1)], dim=0)
class EmptyTensorError(Exception):
    pass
