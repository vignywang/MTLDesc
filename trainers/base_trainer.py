#
# Created by ZhangYuyang on 2020/8/28
#
import time

import torch
import cv2 as cv
from tensorboardX import SummaryWriter

from utils.utils import Matcher
from utils.evaluation_tools import *
from data_utils import get_dataset


class BaseTrainer(object):

    def __init__(self, **config):
        self.config = config
        self.logger = config['logger']

        if torch.cuda.is_available():
            self.logger.info('gpu is available, set device to cuda !')
            self.device = torch.device('cuda:0')
            self.gpu_count = 1
        else:
            self.logger.info('gpu is not available, set device to cpu !')
            self.device = torch.device('cpu')
        self.multi_gpus = False
        self.drop_last = False
        if torch.cuda.device_count() > 1:
            self.gpu_count = torch.cuda.device_count()
            self.config['train']['batch_size'] *= self.gpu_count
            self.multi_gpus = True
            self.drop_last = True
            self.logger.info("Multi gpus is available, let's use %d GPUS" % torch.cuda.device_count())

        # 初始化summary writer
        self.summary_writer = SummaryWriter(self.config['ckpt_path'])

        # 初始化测试数据集
        # 初始化测试集
        self.logger.info("Initialize {}".format(self.config['test']['dataset']))
        self.test_dataset = get_dataset(self.config['test']['dataset'])(**self.config['test'])
        self.test_length = len(self.test_dataset)

        self._initialize_dataset()
        self._initialize_model()
        self._initialize_optimizer()
        self._initialize_scheduler()
        self._initialize_loss()
        self._initialize_matcher()
        self._initialize_test_calculator()

        self.logger.info("Initialize cat func: _cat_c1c2c3c4")
        self.cat = self._cat_c1c2c3c4

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def _inference_func(self, *args, **kwargs):
        raise NotImplementedError

    def _initialize_dataset(self, *args, **kwargs):
        raise NotImplementedError

    def _initialize_model(self, *args, **kwargs):
        self.model = None
        raise NotImplementedError

    def _initialize_optimizer(self, *args, **kwargs):
        self.optimizer = None
        raise NotImplementedError

    def _initialize_scheduler(self, *args, **kwargs):
        self.scheduler = None
        raise NotImplementedError

    def _train_func(self, *args, **kwargs):
        raise NotImplementedError

    def _initialize_loss(self, *args, **kwargs):
        raise NotImplementedError

    def _train_one_epoch(self, *args, **kwargs):
        raise NotImplementedError

    def train(self):
        start_time = time.time()

        # start training
        for i in range(self.config['train']['epoch_num']):

            # train
            self._train_one_epoch(i)
            # break  # todo

            # validation
            # if i >= int(self.config['train']['epoch_num'] * 2/ 3):
            self._validate_one_epoch(i)

            if self.config['train']['adjust_lr']:
                # adjust learning rate
                self.scheduler.step(i)

        end_time = time.time()
        self.logger.info("The whole training process takes %.3f h" % ((end_time - start_time)/3600))

    def _initialize_matcher(self):
        # 初始化匹配算子
        self.logger.info("Initialize matcher of Nearest Neighbor.")
        self.general_matcher = Matcher('float')

    def _load_model_params(self, ckpt_file, previous_model):
        if ckpt_file is None:
            print("Please input correct checkpoint file dir!")
            return False

        self.logger.info("Load pretrained model %s " % ckpt_file)
        if not self.multi_gpus:
            model_dict = previous_model.state_dict()
            pretrain_dict = torch.load(ckpt_file, map_location=self.device)
            model_dict.update(pretrain_dict)
            previous_model.load_state_dict(model_dict)
        else:
            model_dict = previous_model.module.state_dict()
            pretrain_dict = torch.load(ckpt_file, map_location=self.device)
            model_dict.update(pretrain_dict)
            previous_model.module.load_state_dict(model_dict)
        return previous_model

    def _validate_one_epoch(self, epoch_idx):
        self.logger.info("*****************************************************")
        self.logger.info("Validating epoch %2d begin:" % epoch_idx)

        self._test_func(epoch_idx)

        illum_homo_moving_acc = self.illum_homo_acc_mov.average()
        view_homo_moving_acc = self.view_homo_acc_mov.average()

        illum_mma_moving_acc = self.illum_mma_mov.average()
        view_mma_moving_acc = self.view_mma_mov.average()

        illum_repeat_moving_acc = self.illum_repeat_mov.average()
        view_repeat_moving_acc = self.view_repeat_mov.average()

        current_size = self.view_mma_mov.current_size()

        self.logger.info("---------------------------------------------")
        self.logger.info("Moving Average of %d models:" % current_size)
        self.logger.info("illum_homo_moving_acc=%.4f, view_homo_moving_acc=%.4f" %
                         (illum_homo_moving_acc, view_homo_moving_acc))
        self.logger.info("illum_mma_moving_acc=%.4f, view_mma_moving_acc=%.4f" %
                         (illum_mma_moving_acc, view_mma_moving_acc))
        self.logger.info("illum_repeat_moving_acc=%.4f, view_repeat_moving_acc=%.4f" %
                         (illum_repeat_moving_acc, view_repeat_moving_acc))
        self.logger.info("---------------------------------------------")
        self.logger.info("Validating epoch %2d done." % epoch_idx)
        self.logger.info("*****************************************************")

    def _test_func(self, epoch_idx):
        self.model.eval()
        # 重置测评算子参数
        self.illum_repeat.reset()
        self.illum_homo_acc.reset()
        self.illum_mma.reset()
        self.view_repeat.reset()
        self.view_homo_acc.reset()
        self.view_mma.reset()
        self.point_statistics.reset()

        self.illum_bad_mma.reset()
        self.view_bad_mma.reset()

        start_time = time.time()
        count = 0
        skip = 0
        bad = 0

        for i, data in enumerate(self.test_dataset):
            gt_homography = data['gt_homography']
            image_type = data['image_type']

            first_image = data['first_color_image']
            second_image = data['second_color_image']
            image_pair = np.stack((first_image, second_image), axis=0)
            image_pair = torch.from_numpy(image_pair).to(torch.float).to(self.device).permute((0, 3, 1, 2)).contiguous()

            image_pair = image_pair*2./255. - 1.

            results = self._inference_func(image_pair)

            if results is None:
                skip += 1
                continue

            first_point = results[0]
            first_point_num = results[1]
            second_point = results[2]
            second_point_num = results[3]
            select_first_desp = results[4]
            select_second_desp = results[5]

            # 得到匹配点
            matched_point = self.general_matcher(first_point, select_first_desp,
                                                 second_point, select_second_desp)

            if matched_point is None:
                print("skip this pair because there's no match point!")
                skip += 1
                continue

            # 计算得到单应变换
            pred_homography, _ = cv.findHomography(matched_point[0][:, np.newaxis, ::-1],
                                                   matched_point[1][:, np.newaxis, ::-1], cv.RANSAC)

            if pred_homography is None:
                print("skip this pair because no homo can be predicted!.")
                skip += 1
                continue

            # 对单样本进行测评
            if image_type == 'illumination':
                self.illum_repeat.update(first_point, second_point, gt_homography)
                correct = self.illum_homo_acc.update(pred_homography, gt_homography)
                self.illum_mma.update(gt_homography, matched_point)

                if not correct:
                    self.illum_bad_mma.update(gt_homography, matched_point)
                    bad += 1

            elif image_type == 'viewpoint':
                self.view_repeat.update(first_point, second_point, gt_homography)
                correct = self.view_homo_acc.update(pred_homography, gt_homography)
                self.view_mma.update(gt_homography, matched_point)

                if not correct:
                    self.view_bad_mma.update(gt_homography, matched_point)
                    bad += 1

            else:
                print("The image type magicpoint_tester.test(ckpt_file)must be one of illumination of viewpoint ! "
                      "Please check !")
                assert False

            # 统计检测的点的数目
            self.point_statistics.update((first_point_num+second_point_num)/2.)

            if i % 10 == 0:
                print("Having tested %d samples, which takes %.3fs" % (i, (time.time() - start_time)))
                start_time = time.time()
            count += 1
            # if count % 1000 == 0:
            #     break

        # 计算各自的重复率以及总的重复率
        illum_repeat, view_repeat, total_repeat = self._compute_total_metric(self.illum_repeat,
                                                                             self.view_repeat)
        self.illum_repeat_mov.push(illum_repeat)
        self.view_repeat_mov.push(view_repeat)

        # 计算估计的单应变换准确度
        illum_homo_acc, view_homo_acc, total_homo_acc = self._compute_total_metric(self.illum_homo_acc,
                                                                                   self.view_homo_acc)
        self.illum_homo_acc_mov.push(illum_homo_acc)
        self.view_homo_acc_mov.push(view_homo_acc)

        # 计算匹配的准确度
        illum_match_acc, view_match_acc, total_match_acc = self._compute_total_metric(self.illum_mma,
                                                                                      self.view_mma)
        self.illum_mma_mov.push(illum_match_acc)
        self.view_mma_mov.push(view_match_acc)

        # 计算匹配外点的分布情况
        illum_dis, view_dis = self._compute_match_outlier_distribution(self.illum_mma,
                                                                       self.view_mma)

        illum_bad_dis, view_bad_dis = self._compute_match_outlier_distribution(self.illum_bad_mma,
                                                                               self.view_bad_mma)

        # 统计最终的检测点数目的平均值和方差
        point_avg, point_std = self.point_statistics.average()

        self.logger.info("Having skiped %d test pairs" % skip)

        self.logger.info("Homography Accuracy: illumination: %.4f, viewpoint: %.4f, total: %.4f" %
                         (illum_homo_acc, view_homo_acc, total_homo_acc))
        self.logger.info("Mean Matching Accuracy: illumination: %.4f, viewpoint: %.4f, total: %.4f " %
                         (illum_match_acc, view_match_acc, total_match_acc))
        self.logger.info("Repeatability: illumination: %.4f, viewpoint: %.4f, total: %.4f" %
                         (illum_repeat, view_repeat, total_repeat))
        self.logger.info("Detection point, average: %.4f, variance: %.4f" % (point_avg, point_std))

        # self.logger.info("Bad Illumination Matching Distribution:"
        #                  " [0, e/2]: %.4f, (e/2,e]: %.4f, (e,2e]: %.4f, (2e,4e]: %.4f, (4e,+): %.4f" %
        #                  (illum_bad_dis[0], illum_bad_dis[1], illum_bad_dis[2],
        #                   illum_bad_dis[3], illum_bad_dis[4]))
        self.logger.info("Bad Viewpoint Matching Distribution:"
                         " [0, e/2]: %.4f, (e/2,e]: %.4f, (e,2e]: %.4f, (2e,4e]: %.4f, (4e,+): %.4f" %
                         (view_bad_dis[0], view_bad_dis[1], view_bad_dis[2],
                          view_bad_dis[3], view_bad_dis[4]))

        # self.logger.info("Illumination Matching Distribution:"
        #                  " [0, e/2]: %.4f, (e/2,e]: %.4f, (e,2e]: %.4f, (2e,4e]: %.4f, (4e,+): %.4f" %
        #                  (illum_dis[0], illum_dis[1], illum_dis[2],
        #                   illum_dis[3], illum_dis[4]))
        self.logger.info("Viewpoint Matching Distribution:"
                         " [0, e/2]: %.4f, (e/2,e]: %.4f, (e,2e]: %.4f, (2e,4e]: %.4f, (4e,+): %.4f" %
                         (view_dis[0], view_dis[1], view_dis[2],
                          view_dis[3], view_dis[4]))

        self.summary_writer.add_scalar("illumination/Homography_Accuracy", illum_homo_acc, epoch_idx)
        self.summary_writer.add_scalar("illumination/Mean_Matching_Accuracy", illum_match_acc, epoch_idx)
        self.summary_writer.add_scalar("illumination/Repeatability", illum_repeat, epoch_idx)
        self.summary_writer.add_scalar("viewpoint/Homography_Accuracy", view_homo_acc, epoch_idx)
        self.summary_writer.add_scalar("viewpoint/Mean_Matching_Accuracy", view_match_acc, epoch_idx)
        self.summary_writer.add_scalar("viewpoint/Repeatability", view_repeat, epoch_idx)

    def _initialize_test_calculator(self):
        height = self.config['test']['height']
        width = self.config['test']['width']
        correct_epsilon = self.config['test']['correct_epsilon']
        detection_threshold = self.config['test']['detection_threshold']
        top_k = self.config['test']['top_k']

        # 初始化验证算子
        self.logger.info('homography accuracy calculator, correct_epsilon: {}'.format(
            correct_epsilon))
        self.logger.info('repeatability calculator, detection_threshold: {:.4f}, correct_epsilon: {:.4f}'.format(
            detection_threshold, correct_epsilon))
        self.logger.info('Top k: {}'.format(top_k))

        self.illum_repeat = RepeatabilityCalculator(correct_epsilon, height, width)
        self.illum_repeat_mov = MovingAverage(max_size=15)

        self.view_repeat = RepeatabilityCalculator(correct_epsilon, height, width)
        self.view_repeat_mov = MovingAverage(max_size=15)

        self.illum_homo_acc = HomoAccuracyCalculator(correct_epsilon, height, width)
        self.illum_homo_acc_mov = MovingAverage(max_size=15)

        self.view_homo_acc = HomoAccuracyCalculator(correct_epsilon, height, width)
        self.view_homo_acc_mov = MovingAverage(max_size=15)

        self.illum_mma = MeanMatchingAccuracy(correct_epsilon)
        self.illum_mma_mov = MovingAverage(max_size=15)

        self.view_mma = MeanMatchingAccuracy(correct_epsilon)
        self.view_mma_mov = MovingAverage(max_size=15)

        # 初始化专门用于估计的单应变换较差的点匹配情况统计的算子
        self.view_bad_mma = MeanMatchingAccuracy(correct_epsilon)
        self.illum_bad_mma = MeanMatchingAccuracy(correct_epsilon)

        # 初始化用于浮点型描述子的测试方法
        self.illum_homo_acc_f = HomoAccuracyCalculator(correct_epsilon, height, width)
        self.view_homo_acc_f = HomoAccuracyCalculator(correct_epsilon, height, width)

        self.illum_mma_f = MeanMatchingAccuracy(correct_epsilon)
        self.view_mma_f = MeanMatchingAccuracy(correct_epsilon)

        # 初始化用于二进制描述子的测试方法
        self.illum_homo_acc_b = HomoAccuracyCalculator(correct_epsilon, height, width)
        self.view_homo_acc_b = HomoAccuracyCalculator(correct_epsilon, height, width)

        self.illum_mma_b = MeanMatchingAccuracy(correct_epsilon)
        self.view_mma_b = MeanMatchingAccuracy(correct_epsilon)

        self.point_statistics = PointStatistics()

    @staticmethod
    def _compute_total_metric(illum_metric, view_metric):
        illum_acc, illum_sum, illum_num = illum_metric.average()
        view_acc, view_sum, view_num = view_metric.average()
        return illum_acc, view_acc, (illum_sum+view_sum)/(illum_num+view_num+1e-4)

    @staticmethod
    def _compute_match_outlier_distribution(illum_metric, view_metric):
        illum_distribution = illum_metric.average_outlier()
        view_distribution = view_metric.average_outlier()
        return illum_distribution, view_distribution

    @staticmethod
    def _cat_c1c2c3c4(c1, c2, c3, c4, dim):
        return torch.cat((c1, c2, c3, c4), dim=dim)

    @staticmethod
    def _cat_c2c3c4(c1, c2, c3, c4, dim):
        return torch.cat((c2, c3, c4), dim=dim)

    @staticmethod
    def _cat_c1c2c4(c1, c2, c3, c4, dim):
        return torch.cat((c1, c2, c4), dim=dim)

    @staticmethod
    def _cat_c1c3c4(c1, c2, c3, c4, dim):
        return torch.cat((c1, c3, c4), dim=dim)

    @staticmethod
    def _cat_c1c4(c1, c2, c3, c4, dim):
        return torch.cat((c1, c4), dim=dim)

    @staticmethod
    def _cat_c2c4(c1, c2, c3, c4, dim):
        return torch.cat((c2, c4), dim=dim)

    @staticmethod
    def _cat_c3c4(c1, c2, c3, c4, dim):
        return torch.cat((c3, c4), dim=dim)

    @staticmethod
    def _cat_c4(c1, c2, c3, c4, dim):
        return c4

    @staticmethod
    def _convert_pt2cv(point_list):
        cv_point_list = []

        for i in range(len(point_list)):
            cv_point = cv.KeyPoint()
            cv_point.pt = tuple(point_list[i][::-1])
            cv_point_list.append(cv_point)

        return cv_point_list

    @staticmethod
    def _convert_pt2cv_np(point):
        cv_point_list = []
        for i in range(point.shape[0]):
            cv_point = cv.KeyPoint()
            cv_point.pt = tuple(point[i, ::-1])
            cv_point_list.append(cv_point)

        return cv_point_list

    @staticmethod
    def _convert_cv2pt(cv_point):
        point_list = []
        for i, cv_pt in enumerate(cv_point):
            pt = np.array((cv_pt.pt[1], cv_pt.pt[0]))  # y,x的顺序
            point_list.append(pt)
        point = np.stack(point_list, axis=0)
        return point

    @staticmethod
    def _compute_masked_loss(unmasked_loss, mask):
        total_num = torch.sum(mask, dim=(1, 2))
        loss = torch.sum(mask*unmasked_loss, dim=(1, 2)) / total_num
        loss = torch.mean(loss)
        return loss

    @staticmethod
    def _convert_match2cv(first_point_list, second_point_list, sample_ratio=1.0):
        cv_first_point = []
        cv_second_point = []
        cv_matched_list = []

        assert len(first_point_list) == len(second_point_list)

        inc = 1
        if sample_ratio < 1:
            inc = int(1.0 / sample_ratio)

        count = 0
        if len(first_point_list) > 0:
            for j in range(0, len(first_point_list), inc):
                cv_point = cv.KeyPoint()
                cv_point.pt = tuple(first_point_list[j][::-1])
                cv_first_point.append(cv_point)

                cv_point = cv.KeyPoint()
                cv_point.pt = tuple(second_point_list[j][::-1])
                cv_second_point.append(cv_point)

                cv_match = cv.DMatch()
                cv_match.queryIdx = count
                cv_match.trainIdx = count
                cv_matched_list.append(cv_match)

                count += 1

        return cv_first_point, cv_second_point, cv_matched_list

    @staticmethod
    def _generate_predict_point(prob, detection_threshold, scale=None, top_k=0):
        point_idx = np.where(prob > detection_threshold)

        if len(point_idx[0]) == 0 or len(point_idx[1]) == 0:
            point = np.empty((0, 2))
            return point, 0

        prob = prob[point_idx]
        sorted_idx = np.argsort(prob)[::-1]
        if sorted_idx.shape[0] >= top_k:
            sorted_idx = sorted_idx[:top_k]

        point = np.stack(point_idx, axis=1)  # [n,2]
        top_k_point = []
        for idx in sorted_idx:
            top_k_point.append(point[idx])

        point = np.stack(top_k_point, axis=0)
        point_num = point.shape[0]

        if scale is not None:
            point = point*scale
        return point, point_num

    @staticmethod
    def _cvpoint2numpy(point_cv):
        """将opencv格式的特征点转换成numpy数组"""
        point_list = []
        for pt_cv in point_cv:
            point = np.array((pt_cv.pt[1], pt_cv.pt[0]))
            point_list.append(point)
        point_np = np.stack(point_list, axis=0)
        return point_np










