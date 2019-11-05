# 
# Created by ZhangYuyang on 2019/10/30
#
import os
import time

import torch
import numpy as np
import cv2 as cv
from torch.utils.data import DataLoader
import torch.nn.functional as f

from data_utils.coco_dataset import COCORawDataset
from data_utils.dataset_tools import HomographyAugmentation
from data_utils.dataset_tools import contrast_show_different_dataset
from nets.megpoint_net import MegPointShuffleHeatmap
from utils.utils import spatial_nms, interpolation


class Labeler(object):

    def __init__(self, params):
        self.params = params
        self.height = params.height
        self.width = params.width
        self.nms_threshold = params.nms_threshold
        self.detection_threshold = params.detection_threshold
        self.top_k = params.top_k
        self.homography = HomographyAugmentation()
        self.adaption_num = params.adaption_num

        # 初始化相关数据集
        self.raw_dataset = COCORawDataset(
            coco_dataset_dir=params.dataset_dir
        )
        self.data_loader = DataLoader(
            dataset=self.raw_dataset,
            batch_size=params.batch_size,
            shuffle=False,
            num_workers=8,
            drop_last=False
        )
        self.epoch_length = len(self.raw_dataset) // params.batch_size

        self.device = torch.device("cuda:0")

        self.model = None
        self._initialize_model()

        self.out_dir = params.out_dir
        if not os.path.exists(self.out_dir):
            os.mkdir(self.out_dir)

        self.mesh_grid = self._generate_mesh_grid()

    def _initialize_model(self):
        # 初始化模型相关
        print("Initialize Model with %s." % self.params.ckpt_file)
        self.model = MegPointShuffleHeatmap()

        model_state_dict = self.model.state_dict()
        pretrain_state_dict = torch.load(self.params.ckpt_file, map_location=self.device)
        model_state_dict.update(pretrain_state_dict)
        self.model.load_state_dict(model_state_dict)

        self.model = self.model.to(self.device)

    def label(self, total_count=-1):
        self.model.eval()

        if total_count < 0:
            total_count = 1e7

        start_time = time.time()
        dataset_length = len(self.raw_dataset)
        for i, data in enumerate(self.raw_dataset):
            name = data['name']
            org_image = data["org_image"]

            final_probs = self._label_from_homography(org_image)

            satisfied_idx = np.where(final_probs > self.detection_threshold)
            ordered_satisfied_idx = np.argsort(final_probs[satisfied_idx])[::-1]  # 降序
            if len(ordered_satisfied_idx) < self.top_k:
                points = np.stack((satisfied_idx[0][ordered_satisfied_idx],
                                   satisfied_idx[1][ordered_satisfied_idx]), axis=1)
            else:
                points = np.stack((satisfied_idx[0][:self.top_k],
                                   satisfied_idx[1][:self.top_k]), axis=1)

            # debug hyper parameters use
            # draw_image_keypoints(image, points)
            cv.imwrite(os.path.join(self.out_dir, name + '.jpg'), org_image)
            np.save(os.path.join(self.out_dir, name), points)

            if i >= total_count:
                break

            if i % 10 == 0:
                print("Having processed %dth/%d image, takes %.4f/step" % (
                    i, dataset_length, (time.time()-start_time)/10.))
                start_time = time.time()

    def _label_from_scale(self, image):
        down_scale_factors = np.linspace(0.5, 1, 5)
        up_scale_factors = np.linspace(1, 2, 5)
        scales = np.concatenate((down_scale_factors, up_scale_factors[1:]))
        # scales = [1.]
        org_size = image.shape

        image = torch.from_numpy(image).unsqueeze(dim=0).unsqueeze(dim=0).to(self.device).to(torch.float) / 255. * 2 - 1

        self.model.eval()
        heatmaps = []
        for s in scales:
            cur_size = (int(org_size[0]*s), int(org_size[1]*s))
            cur_image = f.interpolate(image, size=(cur_size[0], cur_size[1]), mode="bilinear", align_corners=True)

            cur_results = self.model(cur_image)
            cur_heatmap = cur_results[0]
            cur_heatmap = torch.sigmoid(cur_heatmap)
            cur_heatmap = f.interpolate(cur_heatmap, size=(org_size[0], org_size[1]), mode="bilinear", align_corners=True)

            heatmaps.append(cur_heatmap)

        heatmaps = torch.cat(heatmaps, dim=0)  # [11,1,240,320]
        heatmap = torch.mean(heatmaps, dim=0, keepdim=True)
        heatmap = spatial_nms(heatmap).squeeze().detach().cpu().numpy()
        return heatmap

    def _label_from_homography(self, image):
        self.model.eval()
        # 采样单应变换,对应的逆单应变换
        homographies = []
        inv_homographies = []
        homographies.append(np.eye(3))
        inv_homographies.append(np.eye(3))
        for j in range(self.adaption_num):
            homography = self.homography.sample()
            homographies.append(homography)
            inv_homographies.append(np.linalg.inv(homography))

        homographies = torch.from_numpy(np.stack(homographies, axis=0)).to(self.device).to(torch.float)
        inv_homographies = torch.from_numpy(np.stack(inv_homographies, axis=0)).to(self.device).to(torch.float)

        # batched homography sample
        images = torch.from_numpy(image).unsqueeze(dim=0).unsqueeze(dim=0).to(self.device).to(
            torch.float).repeat((self.adaption_num + 1, 1, 1, 1))  # [n,1,h,w]

        # warped_images = self._batched_homography_warping(images, homographies)
        warped_images = interpolation(images, homographies)
        warped_images = torch.round(warped_images).clamp(0, 255)

        # 送入网络做统一计算
        batched_image = warped_images * 2. / 255. - 1

        # todo
        # split_num = [20, 20, 20, 20, 21]
        split_num = [50, 51]
        batched_image_list = torch.split(batched_image, split_num, dim=0)

        heatmap_list = []
        for img in batched_image_list:
            results = self.model(img)
            heatmap = results[0]
            heatmap = torch.sigmoid(heatmap)
            heatmap = heatmap.detach().cpu().numpy()
            heatmap_list.append(heatmap)

        heatmaps = torch.from_numpy(np.concatenate(heatmap_list, axis=0)).to(self.device)  # [n,1,h,w]

        # 分别得到每个预测的概率图反变换的概率图
        counts = torch.ones_like(images)  # [n,1,h,w]

        heatmaps = interpolation(heatmaps, inv_homographies)  # [n,1,h,w]
        counts = interpolation(counts, inv_homographies)  # [n,1,h,w]
        heatmaps = torch.sum(heatmaps, dim=0, keepdim=True)
        counts = torch.sum(counts, dim=0, keepdim=True)

        torch_probs = heatmaps / counts

        final_probs = spatial_nms(torch_probs, int(self.nms_threshold * 2 + 1)).detach().cpu().numpy()[0, 0]

        return final_probs

    def _generate_mesh_grid(self):
        height = self.height
        width = self.width
        coords_x = torch.linspace(0, width-1, width).unsqueeze(dim=0).repeat((height, 1))
        coords_y = torch.linspace(0, height-1, height).unsqueeze(dim=1).repeat((1, width))
        ones = torch.ones_like(coords_x)
        grid = torch.stack((coords_x, coords_y, ones), dim=2)  # [h,w,3]
        return grid

    @staticmethod
    def generate_predict_point_by_threshold(prob, threshold):
        point_idx = np.where(prob > threshold)

        if point_idx[0].size == 0:
            point = np.empty((0, 2), dtype=np.int64)
            point_num = 0
        else:
            point = np.stack(point_idx, axis=1)  # [n,2]
            point_num = point.shape[0]

        return point, point_num


if __name__ == "__main__":

    class TestParameters(object):

        def __init__(self):
            self.dataset_dir = "/data/MegPoint/dataset/coco"
            self.out_dir = "/data/MegPoint/dataset/coco/train2014/heatmap_test_00"
            # self.ckpt_file = "/home/zhangyuyang/project/development/MegPoint/magicpoint_ckpt/good_results/synthetic_new_0.0010_64/model_59.pt"
            self.ckpt_file = "/home/zhangyuyang/project/development/MegPoint/magicpoint_ckpt/synthetic_heatmap_shuffle_weight_0.0010_64/model_59.pt"
            self.height = 240
            self.width = 320
            self.batch_size = 24
            self.adaption_num = 100
            self.detection_threshold = 0.4
            self.nms_threshold = 4
            self.top_k = 300
            self.point_threshold = None


    params = TestParameters()
    labeler = Labeler(params)
    labeler.label(50)

    heatmap_dataset_dir = "/data/MegPoint/dataset/coco/train2014/heatmap_test_00"
    superpoint_dataset_dir = "/data/MegPoint/dataset/coco/train2014/pseudo_image_points_0"
    contrast_show_different_dataset(heatmap_dataset_dir, superpoint_dataset_dir, 50)












