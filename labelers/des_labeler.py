import h5py
import os
import os.path as osp
from glob import glob
import cv2 as cv
import torch
import torch.nn.functional as f
from tqdm import tqdm
import numpy as np
from nets.scalenet import ScaleBackbone
from data_utils.dataset_tools import HomographyAugmentation
from utils.utils import spatial_nms
class TrainDatasetCreator(object):
    """Step2:根据描述子构造描述子采样对"""
    def __init__(self,**config):
        self.config = config
        self.train = self.config['train']
        self.image_height = self.config['image_height']
        self.image_width = self.config['image_width']
        if torch.cuda.is_available():
            print('gpu is available, set device to cuda!')
            self.device = torch.device('cuda:0')
        else:
            print('gpu is not available, set device to cpu!')
            self.device = torch.device('cpu')
        self.input_image=self.config['input_image']
        self.input_info = self.config['input_info']
        self.output_despoint = self.config['output_despoint']
        if not os.path.exists(self.output_despoint):
            os.mkdir(self.output_despoint)
        # 初始化模型
        model = ScaleBackbone()
        # 从预训练的模型中恢复参数
        model_dict = model.state_dict()
        pretrain_dict = torch.load(self.config['ckpt_path'], map_location=self.device)
        model_dict.update(pretrain_dict)
        model.load_state_dict(model_dict)
        model.to(self.device)
        self.model = model
        bord = self.config['border_remove']
        self.mask = torch.ones(1, 1, self.config['image_height'] - 2 * bord,self.config['image_width'] - 2 * bord).cuda()
        self.mask = f.pad(self.mask, (bord, bord, bord, bord), "constant", value=0).double()
        self.all_points = self._sample_all_point(self.image_height, self.image_width)
    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def build_dataset(self):
        if self.train:
            np.random.seed(4212)
            print('Building a new training dataset...')
        else:
            np.random.seed(8931)
            print('Building the validation dataset...')
        file_list = sorted(glob(osp.join(self.input_image, "*")))
        for img_path in tqdm(file_list):
            image12 = cv.imread(img_path)[:, :, ::-1].copy()  # 交换BGR为RGB
            img_name = (img_path.split('/')[-1]).split('.')[0]
            info_path=self.config['input_info']+"/"+img_name+".npz"
            info = np.load(info_path)
            depth1 = info["depth1"]
            depth2 = info["depth2"]
            pose1 = info["pose1"]
            pose2 = info["pose2"]
            intrinsics1 = info["intrinsics1"]
            intrinsics2 = info["intrinsics2"]
            bbox1 = info["bbox1"]
            bbox2 = info["bbox2"]
            image1, image2 = np.split(image12, 2, axis=1)
            self.img=image1
            h, w, _ = image1.shape
            image1 = torch.from_numpy(image1).to(torch.float).unsqueeze(dim=0).permute((0, 3, 1, 2)).to(self.device)
            image1 = (image1 / 255.) * 2. - 1.
            image2 = torch.from_numpy(image2).to(torch.float).unsqueeze(dim=0).permute((0, 3, 1, 2)).to(self.device)
            image2 = (image2 / 255.) * 2. - 1.
            # detector
            heatmap1,_,_ = self.model(image1)
            heatmap2,_,_ = self.model(image2)
            prob1 = torch.sigmoid(heatmap1).detach().double()
            prob2 = torch.sigmoid(heatmap2).detach()
            try:
                prob2, _ = self._tensor_generate_warped_point(self.all_points, depth1, bbox1, pose1,intrinsics1, depth2, bbox2, pose2,intrinsics2, prob2)
            except:
                prob2=prob1
            prob1 *= self.mask
            prob2 *= self.mask
            prob = prob1 + prob2
            desp_point1 = self._generate_grid_point(prob).astype(np.float32)
            desp_point2, valid_mask, not_search_mask = self._generate_warped_point(desp_point1, depth1, bbox1,
                                                                                   pose1, intrinsics1, depth2,
                                                                                   bbox2, pose2, intrinsics2)
            scale_desp_point1 = self._scale_point_for_sample(desp_point1, height=self.image_height,
                                                             width=self.image_width)
            scale_desp_point2 = self._scale_point_for_sample(desp_point2, height=self.image_height,
                                                             width=self.image_width)
            cur_des_path = os.path.join(self.output_despoint, img_name+'.npz')
            np.savez(cur_des_path, desp_point1=scale_desp_point1, desp_point2=scale_desp_point2, valid_mask=valid_mask,
                     not_search_mask=not_search_mask, raw_desp_point1=desp_point1, raw_desp_point2=desp_point2)
            # 得到对应的预测点
            #prob = prob.detach().cpu().numpy()
            #prob = prob[0, 0]
           # point, score = self._generate_predict_point(prob, height=scale_h, width=scale_w)  # [n,2]
    def _generate_predict_point(self, heatmap, height, width):
        xs, ys = np.where(heatmap >= self.config['detection_threshold'])
        pts = np.zeros((3, len(xs)))  # Populate point data sized 3xN.
        if len(xs) > 0:
            pts[0, :] = ys
            pts[1, :] = xs
            pts[2, :] = heatmap[xs, ys]

            if self.config['nms_radius']:
                pts, _ = self.nms_fast(
                    pts, height, width, dist_thresh=self.config['nms_radius'])
            inds = np.argsort(pts[2, :])
            pts = pts[:, inds[::-1]]  # Sort by confidence.

            # Remove points along border.
            bord = self.config['border_remove']
            toremoveW = np.logical_or(pts[0, :] < bord, pts[0, :] >= (width-bord))
            toremoveH = np.logical_or(pts[1, :] < bord, pts[1, :] >= (height-bord))
            toremove = np.logical_or(toremoveW, toremoveH)
            pts = pts[:, ~toremove]
            pts = pts.transpose()

        point = pts[:, :2][:, ::-1]
        score = pts[:, 2]

        return point, score
    def nms_fast(self, in_corners, H, W, dist_thresh):
        """
        Run a faster approximate Non-Max-Suppression on numpy corners shaped:
          3xN [x_i,y_i,conf_i]^T

        Algo summary: Create a grid sized HxW. Assign each corner location a 1, rest
        are zeros. Iterate through all the 1's and convert them either to -1 or 0.
        Suppress points by setting nearby values to 0.

        Grid Value Legend:
        -1 : Kept.
         0 : Empty or suppressed.
         1 : To be processed (converted to either kept or supressed).

        NOTE: The NMS first rounds points to integers, so NMS distance might not
        be exactly dist_thresh. It also assumes points are within image boundaries.

        Inputs
          in_corners - 3xN numpy array with corners [x_i, y_i, confidence_i]^T.
          H - Image height.
          W - Image width.
          dist_thresh - Distance to suppress, measured as an infinty norm distance.
        Returns
          nmsed_corners - 3xN numpy matrix with surviving corners.
          nmsed_inds - N length numpy vector with surviving corner indices.
        """
        grid = np.zeros((H, W)).astype(int)  # Track NMS data.
        inds = np.zeros((H, W)).astype(int)  # Store indices of points.
        # Sort by confidence and round to nearest int.
        inds1 = np.argsort(-in_corners[2, :])
        corners = in_corners[:, inds1]
        rcorners = corners[:2, :].round().astype(int)  # Rounded corners.
        # Check for edge case of 0 or 1 corners.
        if rcorners.shape[1] == 0:
            return np.zeros((3, 0)).astype(int), np.zeros(0).astype(int)
        if rcorners.shape[1] == 1:
            out = np.vstack((rcorners, in_corners[2])).reshape(3, 1)
            return out, np.zeros((1)).astype(int)
        # Initialize the grid.
        for i, rc in enumerate(rcorners.T):
            grid[rcorners[1, i], rcorners[0, i]] = 1
            inds[rcorners[1, i], rcorners[0, i]] = i
        # Pad the border of the grid, so that we can NMS points near the border.
        pad = dist_thresh
        grid = np.pad(grid, ((pad, pad), (pad, pad)), mode='constant')
        # Iterate through points, highest to lowest conf, suppress neighborhood.
        count = 0
        for i, rc in enumerate(rcorners.T):
            # Account for top and left padding.
            pt = (rc[0] + pad, rc[1] + pad)
            if grid[pt[1], pt[0]] == 1:  # If not yet suppressed.
                grid[pt[1] - pad:pt[1] + pad + 1, pt[0] - pad:pt[0] + pad + 1] = 0
                grid[pt[1], pt[0]] = -1
                count += 1
        # Get all surviving -1's and return sorted array of remaining corners.
        keepy, keepx = np.where(grid == -1)
        keepy, keepx = keepy - pad, keepx - pad
        inds_keep = inds[keepy, keepx]
        out = corners[:, inds_keep]
        values = out[-1, :]
        inds2 = np.argsort(-values)
        out = out[:, inds2]
        out_inds = inds1[inds_keep[inds2]]

        return out, out_inds


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
        xs, ys = np.where(pointmap > 0)
        pts = np.zeros((3, len(xs)))  # Populate point data sized 3xN.
        if len(xs) > 0:
            pts[0, :] = ys
            pts[1, :] = xs
            pts[2, :] = pointmap[xs, ys]
        pts_new, _ = self.nms_fast(pts, self.config['image_height'], self.config['image_width'], dist_thresh=self.config['nms_radius'])
        satisfied_idx=tuple(pts_new[:2][::-1].astype(np.int))
        ordered_satisfied_idx = np.argsort(pointmap[satisfied_idx])[::-1]  # 降序
        if len(ordered_satisfied_idx)<self.config['top_k']:
            pts_new, _ = self.nms_fast(pts, self.config['image_height'], self.config['image_width'],dist_thresh=3)
            satisfied_idx = tuple(pts_new[:2][::-1].astype(np.int))
            ordered_satisfied_idx = np.argsort(pointmap[satisfied_idx])[::-1]  # 降序
            if len(ordered_satisfied_idx) < self.config['top_k']:
               raise EmptyTensorError
        ordered_satisfied_idx = ordered_satisfied_idx[:self.config['top_k']]
        #x,y
        points = np.stack((satisfied_idx[1][ordered_satisfied_idx],
                           satisfied_idx[0][ordered_satisfied_idx]), axis=1)


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
        not_search_mask = (dist <= self.config['nms_radius']) | (invalid_mask[np.newaxis, :])

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

        inlier_mask = torch.abs(estimated_depth - annotated_depth) < 1

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