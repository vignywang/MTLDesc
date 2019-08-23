#
# Created by ZhangYuyang on 2019/8/14
#
import os
import time
import cv2 as cv
import numpy as np
import torch
import torch.nn.functional as f

from nets.superpoint_net import SuperPointNet
from data_utils.synthetic_dataset import SyntheticValTestDataset
from data_utils.hpatch_dataset import HPatchDataset
from utils.evaluation_tools import mAPCalculator
from utils.evaluation_tools import RepeatabilityCalculator
from utils.evaluation_tools import HomoAccuracyCalculator
from utils.tranditional_algorithm import FAST
from utils.utils import spatial_nms, Matcher


class MagicPointSyntheticTester(object):

    def __init__(self, params):
        self.params = params
        self.logger = params.logger
        self.save_threshold_curve = self.params.save_threshold_curve
        if torch.cuda.is_available():
            self.logger.info('gpu is available, set device to cuda!')
            self.device = torch.device('cuda:0')
        else:
            self.logger.info('gpu is not available, set device to cpu!')
            self.device = torch.device('cpu')

        # 初始化测试数据集
        test_dataset = SyntheticValTestDataset(params, dataset_type='validation', add_noise=True)

        # 初始化模型
        model = SuperPointNet()

        # 初始化测评计算子
        mAP_calculator = mAPCalculator()

        self.test_dataset = test_dataset
        self.test_length = len(test_dataset)
        self.model = model
        self.mAP_calculator = mAP_calculator

    def test(self, ckpt_file):

        if ckpt_file == None:
            print("Please input correct checkpoint file dir!")
            return

        curve_name = None
        curve_dir = None
        if self.save_threshold_curve:
            save_root = '/'.join(ckpt_file.split('/')[:-1])
            curve_name = (ckpt_file.split('/')[-1]).split('.')[0]
            curve_dir = os.path.join(save_root, curve_name + '.png')

        # 从预训练的模型中恢复参数
        model_dict = self.model.state_dict()
        pretrain_dict = torch.load(ckpt_file)
        model_dict.update(pretrain_dict)
        self.model.load_state_dict(model_dict)
        self.model.to(self.device)

        # 重置测评算子参数
        self.mAP_calculator.reset()

        self.model.eval()

        self.logger.info("*****************************************************")
        self.logger.info("Testing model %s" % ckpt_file)

        start_time = time.time()
        count = 0

        for i, data in enumerate(self.test_dataset):
            image = data['image']
            gt_point = data['gt_point']
            gt_point = gt_point.numpy()

            image = image.to(self.device).unsqueeze(dim=0)
            # 得到原始的经压缩的概率图，概率图每个通道64维，对应空间每个像素是否为关键点的概率
            _, _, prob = self.model(image)
            # 将概率图展开为原始图像大小
            prob = f.pixel_shuffle(prob, 8)
            # 进行非极大值抑制
            prob = spatial_nms(prob)
            prob = prob.detach().cpu().numpy()[0, 0]

            self.mAP_calculator.update(prob, gt_point)
            if i % 10 == 0:
                print("Having tested %d samples, which takes %.3fs" % (i, (time.time()-start_time)))
                start_time = time.time()
            count += 1
            # if count % 1000 == 0:
            #     break

        mAP, test_data = self.mAP_calculator.compute_mAP()
        if self.save_threshold_curve:
            self.mAP_calculator.plot_threshold_curve(test_data, curve_name, curve_dir)

        self.logger.info("The mean Average Precision : %.4f of %d samples" % (mAP, count))
        self.logger.info("Testing done.")
        self.logger.info("*****************************************************")

    def test_single_image(self, ckpt_file, image_dir):

        if ckpt_file == None:
            print("Please input correct checkpoint file dir!")
            return

        # 从预训练的模型中恢复参数
        model_dict = self.model.state_dict()
        pretrain_dict = torch.load(ckpt_file, map_location=self.device)
        model_dict.update(pretrain_dict)
        self.model.load_state_dict(model_dict)
        self.model.to(self.device)

        self.model.eval()

        cv_image = cv.imread(image_dir, cv.IMREAD_GRAYSCALE)
        image = np.expand_dims(np.expand_dims(cv_image, 0), 0)
        image = torch.from_numpy(image).to(torch.float)

        _, _, prob = self.model(image)
        prob = f.pixel_shuffle(prob, 8)
        # 进行非极大值抑制
        prob = spatial_nms(prob)
        prob = prob.detach().cpu().numpy()[0, 0]

        pred_pt = np.where(prob>0.1)

        cv_pt_list = []
        for i in range(len(pred_pt[0])):
            kpt = cv.KeyPoint()
            kpt.pt = (pred_pt[1][i], pred_pt[0][i])
            cv_pt_list.append(kpt)

        result_dir = os.path.join('/'.join(ckpt_file.split('/')[:-1]), 'test_image.jpg')
        cv_image = cv.drawKeypoints(cv_image, cv_pt_list, None, color=(0, 0, 255))
        cv.imwrite(result_dir, cv_image)


class HPatchTester(object):

    def __init__(self, params):
        self.params = params
        self.logger = params.logger
        self.detection_threshold = params.detection_threshold
        self.correct_epsilon = params.correct_epsilon
        self.rep_top_k = params.rep_top_k
        self.desp_top_k = params.desp_top_k
        if torch.cuda.is_available():
            self.logger.info('gpu is available, set device to cuda!')
            self.device = torch.device('cuda:0')
        else:
            self.logger.info('gpu is not available, set device to cpu!')
            self.device = torch.device('cpu')

        # 初始化测试数据集
        self.test_dataset = HPatchDataset(params)
        self.test_length = len(self.test_dataset)

        # 初始化模型
        self.model = SuperPointNet()
        self.model_fast = FAST(top_k=self.rep_top_k)

        # 初始化测评计算子
        self.logger.info('Initialize the repeatability calculator, detection_threshold: %.4f, coorect_epsilon: %d'
                         % (self.detection_threshold, self.correct_epsilon))
        self.logger.info('Repeatability Top k: %d' % self.rep_top_k)
        self.logger.info('Descriptor Top k: %d' % self.desp_top_k)
        self.illumination_repeatability = RepeatabilityCalculator(params.correct_epsilon)
        self.viewpoint_repeatability = RepeatabilityCalculator(params.correct_epsilon)
        self.homo_accuracy = HomoAccuracyCalculator(params.correct_epsilon, params.height, params.width)
        self.matcher = Matcher()

    def test_keypoint_repeatability(self, ckpt_file):

        if ckpt_file == None:
            print("Please input correct checkpoint file dir!")
            return

        # 从预训练的模型中恢复参数
        model_dict = self.model.state_dict()
        pretrain_dict = torch.load(ckpt_file)
        model_dict.update(pretrain_dict)
        self.model.load_state_dict(model_dict)
        self.model.to(self.device)

        # 重置测评算子参数
        self.illumination_repeatability.reset()
        self.viewpoint_repeatability.reset()

        self.model.eval()

        self.logger.info("*****************************************************")
        self.logger.info("Testing model %s" % ckpt_file)

        start_time = time.time()
        count = 0

        for i, data in enumerate(self.test_dataset):
            first_image = data['first_image']
            second_image = data['second_image']
            gt_homography = data['gt_homography']
            image_type = data['image_type']

            image_pair = np.stack((first_image, second_image), axis=0)
            image_pair = torch.from_numpy(image_pair).to(torch.float).to(self.device).unsqueeze(dim=1)

            _, _, prob_pair = self.model(image_pair)
            prob_pair = f.pixel_shuffle(prob_pair, 8)
            prob_pair = spatial_nms(prob_pair, kernel_size=3)
            prob_pair = prob_pair.detach().cpu().numpy()
            first_prob = prob_pair[0, 0]
            second_prob = prob_pair[1, 0]
            # 得到对应的预测点
            first_point = self.generate_predict_point(first_prob, top_k=self.rep_top_k)  # [n,2]
            second_point = self.generate_predict_point(second_prob, top_k=self.rep_top_k)  # [m,2]

            # 对单样本进行测评
            if image_type == 'illumination':
                self.illumination_repeatability.update(first_point, second_point, gt_homography)
            elif image_type == 'viewpoint':
                self.viewpoint_repeatability.update(first_point, second_point, gt_homography)
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

        self.logger.info("Repeatability: illumination: %.4f, viewpoint: %.4f, total: %.4f" %
                         (illum_repeat, view_repeat, total_repeat))
        self.logger.info("Testing HPatch Repeatability done.")
        self.logger.info("*****************************************************")

    def test_descriptors(self, ckpt_file):

        if ckpt_file == None:
            print("Please input correct checkpoint file dir!")
            return

        # 从预训练的模型中恢复参数
        model_dict = self.model.state_dict()
        pretrain_dict = torch.load(ckpt_file)
        model_dict.update(pretrain_dict)
        self.model.load_state_dict(model_dict)
        self.model.to(self.device)

        # 重置测评算子参数
        self.illumination_repeatability.reset()
        self.viewpoint_repeatability.reset()

        self.model.eval()

        self.logger.info("*****************************************************")
        self.logger.info("Testing model %s" % ckpt_file)

        start_time = time.time()
        count = 0

        for i, data in enumerate(self.test_dataset):
            first_image = data['first_image']
            second_image = data['second_image']
            gt_homography = data['gt_homography']

            image_pair = np.stack((first_image, second_image), axis=0)
            image_pair = torch.from_numpy(image_pair).to(torch.float).to(self.device).unsqueeze(dim=1)

            _, desp_pair, prob_pair = self.model(image_pair)
            prob_pair = f.pixel_shuffle(prob_pair, 8)
            prob_pair = spatial_nms(prob_pair, kernel_size=3)

            desp_pair = desp_pair.detach().cpu().numpy()
            first_desp = desp_pair[0]
            second_desp = desp_pair[1]
            prob_pair = prob_pair.detach().cpu().numpy()
            first_prob = prob_pair[0, 0]
            second_prob = prob_pair[1, 0]

            # 得到对应的预测点
            first_point = self.generate_predict_point(first_prob, top_k=self.desp_top_k)  # [n,2]
            second_point = self.generate_predict_point(second_prob, top_k=self.desp_top_k)  # [m,2]

            # 得到点对应的描述子
            select_first_desp = self.generate_predict_descriptor(first_point, first_desp)
            select_second_desp = self.generate_predict_descriptor(second_point, second_desp)

            # 得到匹配点
            matched_point = self.matcher(first_point, select_first_desp, second_point, select_second_desp)

            # 计算得到单应变换
            pred_homography, _ = cv.findHomography(matched_point[0], matched_point[1], cv.RANSAC)

            # 对单样本进行测评
            self.homo_accuracy.update(pred_homography, gt_homography)

            if i % 10 == 0:
                print("Having tested %d samples, which takes %.3fs" % (i, (time.time()-start_time)))
                start_time = time.time()
            count += 1
            # if count % 1000 == 0:
            #     break

        accuracy = self.homo_accuracy.average()
        self.logger.info("THe average accuracy is %.4f " % accuracy)
        self.logger.info("Testing HPatch descriptors done.")
        self.logger.info("*****************************************************")

    def generate_predict_point(self, prob, scale=None, top_k=0):
        point_idx = np.where(prob > self.detection_threshold)
        prob = prob[point_idx]
        sorted_idx = np.argsort(prob)[::-1]
        if sorted_idx.shape[0] >= top_k:
            sorted_idx = sorted_idx[:300]

        point = np.stack(point_idx, axis=1)  # [n,2]
        top_k_point = []
        for idx in sorted_idx:
            top_k_point.append(point[idx])
        point = np.stack(top_k_point, axis=0)

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
        interpolation_norm = torch.norm(interpolation_desp, dim=1, keepdim=True)
        interpolation_desp = interpolation_desp/interpolation_norm

        return interpolation_desp.numpy()

    def test_fast(self):

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



