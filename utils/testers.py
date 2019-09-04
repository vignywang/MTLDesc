#
# Created by ZhangYuyang on 2019/8/14
#
import os
import time
import cv2 as cv
import numpy as np
import torch

from data_utils.hpatch_dataset import HPatchDataset
from utils.evaluation_tools import RepeatabilityCalculator, HomoAccuracyCalculator, MeanMatchingAccuracy
from utils.tranditional_algorithm import FAST
from utils.utils import spatial_nms, Matcher


class HPatchTester(object):

    def __init__(self, params):
        self.params = params
        self.logger = params.logger
        self.detection_threshold = params.detection_threshold
        self.correct_epsilon = params.correct_epsilon
        self.top_k = params.top_k
        self.nms_threshold = params.nms_threshold

        # 初始化测试数据集
        self.test_dataset = HPatchDataset(params)
        self.test_length = len(self.test_dataset)

        # 初始化模型
        self.model_fast = FAST(top_k=self.top_k)
        self.orb = cv.ORB_create(nfeatures=self.top_k)

        # 初始化测评计算子
        self.logger.info('Initialize the repeatability calculator, detection_threshold: %.4f, coorect_epsilon: %d'
                         % (self.detection_threshold, self.correct_epsilon))
        self.logger.info('Top k: %d' % self.top_k)
        self.illumination_repeatability = RepeatabilityCalculator(params.correct_epsilon)
        self.illumination_homo_accuracy = HomoAccuracyCalculator(params.correct_epsilon,
                                                                 params.hpatch_height, params.hpatch_width)
        self.illumination_mma = MeanMatchingAccuracy(params.correct_epsilon)
        self.viewpoint_repeatability = RepeatabilityCalculator(params.correct_epsilon)
        self.viewpoint_homo_accuracy = HomoAccuracyCalculator(params.correct_epsilon,
                                                              params.hpatch_height, params.hpatch_width)
        self.viewpoint_mma = MeanMatchingAccuracy(params.correct_epsilon)

        self.matcher = Matcher()
        self.orb_matcher = cv.BFMatcher_create(cv.NORM_HAMMING, crossCheck=True)

    def test_FAST_repeatability(self):

        start_time = time.time()
        count = 0

        self.illumination_repeatability.reset()
        self.viewpoint_repeatability.reset()

        self.logger.info("*****************************************************")
        self.logger.info("Testing FAST corner detection algorithm")

        for i, data in enumerate(self.test_dataset):
            first_image = data['first_image']
            second_image = data['second_image']
            gt_homography = data['gt_homography']
            image_type = data['image_type']

            first_point = self.model_fast.detect(first_image)
            second_point = self.model_fast.detect(second_image)
            if first_point.shape[0] == 0 or second_point.shape[0] == 0:
                continue

            # 对单样本进行测评
            if image_type == 'illumination':
                self.illumination_repeatability.update(first_point, second_point, gt_homography)
            elif image_type == 'viewpoint':
                self.viewpoint_repeatability.update(first_point, second_point, gt_homography)
            else:
                print("The image type must be one of illumination of viewpoint ! Please check !")
                assert False

            if i % 10 == 0:
                print("Having tested %d samples, which takes %.3fs" % (i, (time.time()-start_time)))
                start_time = time.time()
            count += 1

        # 计算各自的重复率以及总的重复率
        illum_repeat, illum_repeat_sum, illum_num_sum = self.illumination_repeatability.average()
        view_repeat, view_repeat_sum, view_num_sum = self.viewpoint_repeatability.average()
        total_repeat = (illum_repeat_sum + view_repeat_sum) / (illum_num_sum + view_num_sum)

        self.logger.info("FAST Repeatability: illumination: %.4f, viewpoint: %.4f, total: %.4f, %d/%d" %
                         (illum_repeat, view_repeat, total_repeat, count, len(self.test_dataset)))
        self.logger.info("Testing HPatch Repeatability done.")
        self.logger.info("*****************************************************")

    def test_orb_descriptors(self):

        # 重置测评算子参数
        self.illumination_repeatability.reset()
        self.illumination_homo_accuracy.reset()
        self.illumination_mma.reset()
        self.viewpoint_repeatability.reset()
        self.viewpoint_homo_accuracy.reset()
        self.viewpoint_mma.reset()

        self.logger.info("*****************************************************")
        self.logger.info("Testing ORB descriptors")

        start_time = time.time()
        count = 0

        for i, data in enumerate(self.test_dataset):
            first_image = data['first_image']
            second_image = data['second_image']
            gt_homography = data['gt_homography']
            image_type = data['image_type']

            first_kp = self.orb.detect(first_image, None)
            first_kp, first_desp = self.orb.compute(first_image, first_kp, None)

            second_kp = self.orb.detect(second_image, None)
            second_kp, second_desp = self.orb.compute(second_image, second_kp, None)

            # 得到匹配点
            matched = self.orb_matcher.match(first_desp, second_desp)
            src_pts = np.float32([first_kp[m.queryIdx].pt for m in matched]).reshape(-1, 2)
            dst_pts = np.float32([second_kp[m.trainIdx].pt for m in matched]).reshape(-1, 2)
            src_pts = src_pts[:, ::-1]
            dst_pts = dst_pts[:, ::-1]
            matched_point = (src_pts, dst_pts)

            # 计算得到单应变换
            pred_homography, _ = cv.findHomography(src_pts[:, np.newaxis, ::-1],
                                                   dst_pts[:, np.newaxis, ::-1], cv.RANSAC)

            # 对单样本进行测评
            if image_type == 'illumination':
                self.illumination_repeatability.update(src_pts, dst_pts, gt_homography)
                self.illumination_homo_accuracy.update(pred_homography, gt_homography)
                self.illumination_mma.update(gt_homography, matched_point)
            elif image_type == 'viewpoint':
                self.viewpoint_repeatability.update(src_pts, dst_pts, gt_homography)
                self.viewpoint_homo_accuracy.update(pred_homography, gt_homography)
                self.viewpoint_mma.update(gt_homography, matched_point)
            else:
                print("The image type magicpoint_tester.test(ckpt_file)must be one of illumination of viewpoint ! "
                      "Please check !")
                assert False

            if i % 10 == 0:
                print("Having tested %d samples, which takes %.3fs" % (i, (time.time()-start_time)))
                start_time = time.time()
            count += 1
            # if count % 1000 == 0:
            #     break

        # 计算各自的重复率以及总的重复率
        illum_repeat, illum_repeat_sum, illum_num_sum = self.illumination_repeatability.average()
        view_repeat, view_repeat_sum, view_num_sum = self.viewpoint_repeatability.average()
        total_repeat = (illum_repeat_sum + view_repeat_sum) / (illum_num_sum + view_num_sum)

        # 计算估计的单应变换准确度
        illum_homo_acc, illum_homo_sum, illum_homo_num = self.illumination_homo_accuracy.average()
        view_homo_acc, view_homo_sum, view_homo_num = self.viewpoint_homo_accuracy.average()
        total_homo_acc = (illum_homo_sum + view_homo_sum) / (illum_homo_num + view_homo_num)

        # 计算匹配的准确度
        illum_match_acc, illum_match_sum, illum_match_num = self.illumination_mma.average()
        view_match_acc, view_match_sum, view_match_num = self.viewpoint_mma.average()
        total_match_acc = (illum_match_sum + view_match_sum) / (illum_match_num + view_match_num)

        self.logger.info("Homography accuracy: illumination: %.4f, viewpoint: %.4f, total: %.4f" %
                         (illum_homo_acc, view_homo_acc, total_homo_acc))
        self.logger.info("Mean Matching Accuracy: illumination: %.4f, viewpoint: %.4f, total: %.4f " %
                         (illum_match_acc, view_match_acc, total_match_acc))
        self.logger.info("Repeatability: illumination: %.4f, viewpoint: %.4f, total: %.4f" %
                         (illum_repeat, view_repeat, total_repeat))
        self.logger.info("Testing HPatch done.")
        self.logger.info("*****************************************************")

    def generate_predict_point(self, prob, scale=None, top_k=0):
        point_idx = np.where(prob > self.detection_threshold)
        prob = prob[point_idx]
        sorted_idx = np.argsort(prob)[::-1]
        if sorted_idx.shape[0] >= top_k:
            sorted_idx = sorted_idx[:top_k]

        point = np.stack(point_idx, axis=1)  # [n,2]
        top_k_point = []
        for idx in sorted_idx:
            top_k_point.append(point[idx])
        point = np.stack(top_k_point, axis=0).astype(np.float)

        if scale is not None:
            point = point*scale
        return point

    def generate_predict_descriptor(self, point, desp):
        point = torch.from_numpy(point).to(torch.float)  # 由于只有pytorch有gather的接口，因此将点调整为pytorch的格式
        desp = torch.from_numpy(desp)
        dim, h, w = desp.shape
        desp = torch.reshape(desp, (dim, -1))
        desp = torch.transpose(desp, dim0=1, dim1=0)  # [h*w,256]

        # 下采样
        scaled_point = point / 8
        point_y = scaled_point[:, 0:1]  # [n,1]
        point_x = scaled_point[:, 1:2]

        x0 = torch.floor(point_x)
        x1 = x0 + 1
        y0 = torch.floor(point_y)
        y1 = y0 + 1
        x_nearest = torch.round(point_x)
        y_nearest = torch.round(point_y)

        x0_safe = torch.clamp(x0, min=0, max=w-1)
        x1_safe = torch.clamp(x1, min=0, max=w-1)
        y0_safe = torch.clamp(y0, min=0, max=h-1)
        y1_safe = torch.clamp(y1, min=0, max=h-1)

        x_nearest_safe = torch.clamp(x_nearest, min=0, max=w-1)
        y_nearest_safe = torch.clamp(y_nearest, min=0, max=h-1)

        idx_00 = (x0_safe + y0_safe*w).to(torch.long).repeat((1, dim))
        idx_01 = (x0_safe + y1_safe*w).to(torch.long).repeat((1, dim))
        idx_10 = (x1_safe + y0_safe*w).to(torch.long).repeat((1, dim))
        idx_11 = (x1_safe + y1_safe*w).to(torch.long).repeat((1, dim))
        idx_nearest = (x_nearest_safe + y_nearest_safe*w).to(torch.long).repeat((1, dim))

        d_x = point_x - x0_safe
        d_y = point_y - y0_safe
        d_1_x = x1_safe - point_x
        d_1_y = y1_safe - point_y

        desp_00 = torch.gather(desp, dim=0, index=idx_00)
        desp_01 = torch.gather(desp, dim=0, index=idx_01)
        desp_10 = torch.gather(desp, dim=0, index=idx_10)
        desp_11 = torch.gather(desp, dim=0, index=idx_11)
        nearest_desp = torch.gather(desp, dim=0, index=idx_nearest)
        bilinear_desp = desp_00*d_1_x*d_1_y + desp_01*d_1_x*d_y + desp_10*d_x*d_1_y+desp_11*d_x*d_y

        # todo: 插值得到的描述子不再满足模值为1，强行归一化到模值为1，这里可能有问题
        condition = torch.eq(torch.norm(bilinear_desp, dim=1, keepdim=True), 0)
        interpolation_desp = torch.where(condition, nearest_desp, bilinear_desp)
        # interpolation_norm = torch.norm(interpolation_desp, dim=1, keepdim=True)
        # interpolation_desp = interpolation_desp/interpolation_norm

        return interpolation_desp.numpy()

    def _convert_cvpt_to_array(self, cv_pt):
        point = []
        for pt in cv_pt:
            point.append((pt.pt[1], pt.pt[0]))
        point = np.stack(point, axis=0)
        return point






