# 
# Created by ZhangYuyang on 2019/9/18
#
# 训练算子基类
import os
import time

import torch
import torch.nn.functional as f
import numpy as np
import cv2 as cv
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader

from nets.megpoint_net import MegPointShuffleHeatmap
from nets.megpoint_net import MegPointResidualShuffleHeatmap

from data_utils.coco_dataset import COCOMegPointHeatmapTrainDataset
from data_utils.coco_dataset import COCOMegPointHeatmapPreciseTrainDataset
from data_utils.synthetic_dataset import SyntheticHeatmapDataset
from data_utils.synthetic_dataset import SyntheticValTestDataset
from data_utils.hpatch_dataset import HPatchDataset
from data_utils.dataset_tools import draw_image_keypoints

from utils.trainers import MagicPointSynthetic
from utils.evaluation_tools import RepeatabilityCalculator
from utils.evaluation_tools import MovingAverage
from utils.evaluation_tools import PointStatistics
from utils.evaluation_tools import HomoAccuracyCalculator
from utils.evaluation_tools import MeanMatchingAccuracy
from utils.evaluation_tools import mAPCalculator
from utils.utils import spatial_nms
from utils.utils import DescriptorTripletLoss
from utils.utils import DescriptorPreciseTripletLoss
from utils.utils import Matcher
from utils.utils import NearestNeighborThresholdMatcher
from utils.utils import NearestNeighborRatioMatcher
from utils.utils import PointHeatmapWeightedBCELoss


class MagicPointHeatmapTrainer(MagicPointSynthetic):

    def __init__(self, params):
        super(MagicPointHeatmapTrainer, self).__init__(params=params)

        self.point_loss = PointHeatmapWeightedBCELoss(weight=200)

        # 重新初始化带l2正则项的优化器
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=1e-6)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=100, gamma=0.1)

    def _initialize_model(self):
        self.model = MegPointShuffleHeatmap()

        if self.multi_gpus:
            self.model = torch.nn.DataParallel(self.model)
        self.model = self.model.to(self.device)

    def _initialize_dataset(self):
        train_dataset = SyntheticHeatmapDataset(self.params)
        val_dataset = SyntheticValTestDataset(self.params, 'validation')
        test_dataset = SyntheticValTestDataset(self.params, 'validation', add_noise=True)
        return train_dataset, val_dataset, test_dataset

    def _train_one_epoch(self, epoch_idx):
        self.model.train()

        self.logger.info("-----------------------------------------------------")
        self.logger.info("Training epoch %2d begin:" % epoch_idx)

        stime = time.time()
        for i, data in enumerate(self.train_dataloader):
            image = data['image'].to(self.device)
            mask = data['mask'].to(self.device)
            heatmap_gt = data["heatmap"].to(self.device)

            results = self.model(image)
            heatmap_pred = results[0][:, 0, :, :]
            loss = self.point_loss(heatmap_pred, heatmap_gt, mask)

            if torch.isnan(loss):
                self.logger.error('loss is nan!')

            self.optimizer.zero_grad()
            loss.backward()

            self.optimizer.step()

            if i % self.log_freq == 0:

                loss_val = loss.item()
                self.summary_writer.add_scalar('loss', loss_val)
                self.logger.info("[Epoch:%2d][Step:%5d:%5d]: loss = %.4f,"
                                 " one step cost %.4fs. "
                                 % (epoch_idx, i, self.epoch_length, loss_val,
                                    (time.time() - stime) / self.params.log_freq,
                                    ))
                stime = time.time()

        # save the model
        if self.multi_gpus:
            torch.save(self.model.module.state_dict(), os.path.join(self.ckpt_dir, 'model_%02d.pt' % epoch_idx))
        else:
            torch.save(self.model.state_dict(), os.path.join(self.ckpt_dir, 'model_%02d.pt' % epoch_idx))

        self.logger.info("Training epoch %2d done." % epoch_idx)
        self.logger.info("-----------------------------------------------------")

    def _test_func(self, dataset):
        self.model.eval()
        self.mAP_calculator.reset()

        start_time = time.time()
        count = 0

        for i, data in enumerate(dataset):
            image = data['image']
            gt_point = data['gt_point']
            gt_point = gt_point.numpy()

            image = image.to(self.device).unsqueeze(dim=0)
            results = self.model(image)

            heatmap = results[0]
            heatmap = torch.sigmoid(heatmap)
            heatmap = spatial_nms(heatmap)
            heatmap = heatmap.detach().cpu().numpy()[0, 0]

            self.mAP_calculator.update(heatmap, gt_point)
            if i % 10 == 0:
                print("Having tested %d samples, which takes %.3fs" % (i, (time.time()-start_time)))
                start_time = time.time()
            count += 1
            # if count % 1000 == 0:
            #     break

        mAP, test_data = self.mAP_calculator.compute_mAP()

        return mAP, test_data, count


class MegPointTrainerTester(object):

    def __init__(self, params):
        self.params = params
        self.batch_size = params.batch_size
        self.lr = params.lr
        self.epoch_num = params.epoch_num
        self.logger = params.logger
        self.ckpt_dir = params.ckpt_dir
        self.num_workers = params.num_workers
        self.log_freq = params.log_freq
        self.top_k = params.top_k
        self.nms_threshold = params.nms_threshold
        self.detection_threshold = params.detection_threshold
        self.correct_epsilon = params.correct_epsilon

        self.network_arch = params.network_arch
        self.train_mode = params.train_mode
        self.detection_mode = params.detection_mode
        self.homo_pred_mode = params.homo_pred_mode
        self.match_mode = params.match_mode

        # todo:
        self.sift = cv.xfeatures2d.SIFT_create(1000)

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
            self.batch_size *= self.gpu_count
            self.multi_gpus = True
            self.drop_last = True
            self.logger.info("Multi gpus is available, let's use %d GPUS" % torch.cuda.device_count())

        # 初始化summary writer
        self.summary_writer = SummaryWriter(self.ckpt_dir)

        # 初始化模型
        self.model = None

        # 初始化优化器算子
        self.optimizer = None

        # 初始化学习率调整算子
        self.scheduler = None

    def train(self):
        start_time = time.time()

        # start training
        for i in range(self.epoch_num):

            # train
            self._train_one_epoch(i)
            # break  # todo

            # validation
            self._validate_one_epoch(i)

            # adjust learning rate
            # self.scheduler.step(i)

        end_time = time.time()
        self.logger.info("The whole training process takes %.3f h" % ((end_time - start_time)/3600))

    def _train_one_epoch(self, epoch_idx):
        raise NotImplementedError

    def _validate_one_epoch(self, epoch_idx):
        raise NotImplementedError

    def _load_model_params(self, ckpt_file):
        if ckpt_file is None:
            print("Please input correct checkpoint file dir!")
            return False

        self.logger.info("Load pretrained model %s " % ckpt_file)
        if not self.multi_gpus:
            model_dict = self.model.state_dict()
            pretrain_dict = torch.load(ckpt_file, map_location=self.device)
            model_dict.update(pretrain_dict)
            self.model.load_state_dict(model_dict)
        else:
            model_dict = self.model.module.state_dict()
            pretrain_dict = torch.load(ckpt_file, map_location=self.device)
            model_dict.update(pretrain_dict)
            self.model.module.load_state_dict(model_dict)


class MegPointHeatmapTrainer(MegPointTrainerTester):

    def __init__(self, params):
        super(MegPointHeatmapTrainer, self).__init__(params)

        self._initialize_dataset()
        self._initialize_model()
        self._initialize_optimizer()
        self._initialize_train_func()
        self._initialize_loss()
        self._initialize_matcher()
        self._initialize_test_calculator(params)

    def _initialize_dataset(self):
        # 初始化数据集
        if self.train_mode == "with_gt":
            self.logger.info("Initialize COCOMegPointHeatmapTrainDataset")
            self.train_dataset = COCOMegPointHeatmapTrainDataset(self.params)
        elif self.train_mode == "with_precise_gt":
            self.logger.info("Initialize COCOMegPointHeatmapPreciseTrainDataset")
            self.train_dataset = COCOMegPointHeatmapPreciseTrainDataset(self.params)
        else:
            assert False

        self.train_dataloader = DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            drop_last=True
        )
        self.epoch_length = len(self.train_dataset) // self.batch_size

        # 初始化测试集
        self.test_dataset = HPatchDataset(self.params)
        self.test_length = len(self.test_dataset)

    def _initialize_model(self):
        # 初始化模型
        if self.network_arch == "baseline":
            self.logger.info("Initialize network arch : ShuffleHeatmap")
            model = MegPointShuffleHeatmap()
        elif self.network_arch == "residual":
            self.logger.info("Initialize network arch : Residual+ShuffleHeatmap")
            model = MegPointResidualShuffleHeatmap()
        else:
            self.logger.error("unrecognized network_arch:%s" % self.network_arch)
            assert False
        if self.multi_gpus:
            model = torch.nn.DataParallel(model)
        self.model = model.to(self.device)

    def _initialize_loss(self):
        # 初始化loss算子
        # 初始化heatmap loss
        self.point_loss = PointHeatmapWeightedBCELoss()

        # 初始化描述子loss
        if self.train_mode == "with_precise_gt":
            self.logger.info("Initialize the DescriptorPreciseTripletLoss.")
            self.descriptor_loss = DescriptorPreciseTripletLoss(self.device)
        else:
            self.logger.info("Initialize the DescriptorTripletLoss.")
            self.descriptor_loss = DescriptorTripletLoss(self.device)

    def _initialize_matcher(self):
        # 初始化匹配算子
        if self.match_mode == "NN":
            self.logger.info("Initialize matcher of Nearest Neighbor.")
            self.general_matcher = Matcher('float')
        elif self.match_mode == "NNT":
            self.logger.info("Initialize matcher of Nearest Neighbor Threshold of %.2f." % 1.0)
            self.general_matcher = NearestNeighborThresholdMatcher(threshold=1.0)
        elif self.match_mode == "NNR":
            self.logger.info("Initialize matcher of Nearest Neighbor Ratio of %.2f" % 0.9)
            self.general_matcher = NearestNeighborRatioMatcher(ratio=0.9)
        else:
            self.logger.error("Unrecognized match_mode of %s!" % self.match_mode)
            assert False

    def _initialize_train_func(self):
        # 根据不同结构选择不同的训练函数
        if self.network_arch == "baseline":
            self.logger.info("Initialize training func mode of [with_gt] with baseline network.")
            self._train_func = self._train_with_gt
        elif self.network_arch == "residual":
            self.logger.info("Initialize training func mode of [with_residual] with residual network.")
            self._train_func = self._train_with_residual
        else:
            self.logger.error("Unrecognized network_arch: %s" % self.network_arch)

    def _initialize_optimizer(self):
        # 初始化网络训练优化器
        self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr=self.lr)

    def _train_one_epoch(self, epoch_idx):
        self.model.train()

        self.logger.info("-----------------------------------------------------")
        self.logger.info("Training epoch %2d begin:" % epoch_idx)

        self._train_func(epoch_idx)

        self.logger.info("Training epoch %2d done." % epoch_idx)
        self.logger.info("-----------------------------------------------------")

    def _train_with_gt(self, epoch_idx):
        self.model.train()
        stime = time.time()
        for i, data in enumerate(self.train_dataloader):

            image = data['image'].to(self.device)
            heatmap_gt = data['heatmap'].to(self.device)
            point_mask = data['point_mask'].to(self.device)

            warped_image = data['warped_image'].to(self.device)
            warped_heatmap_gt = data['warped_heatmap'].to(self.device)
            warped_point_mask = data['warped_point_mask'].to(self.device)

            matched_idx = data['matched_idx'].to(self.device)
            matched_valid = data['matched_valid'].to(self.device)
            not_search_mask = data['not_search_mask'].to(self.device)

            shape = image.shape

            image_pair = torch.cat((image, warped_image), dim=0)
            heatmap_gt_pair = torch.cat((heatmap_gt, warped_heatmap_gt), dim=0)
            point_mask_pair = torch.cat((point_mask, warped_point_mask), dim=0)
            heatmap_pred_pair, desp_pair = self.model(image_pair)
            heatmap_pred_pair = heatmap_pred_pair.squeeze()

            point_loss = self.point_loss(heatmap_pred_pair, heatmap_gt_pair, point_mask_pair)

            desp_0, desp_1 = torch.split(desp_pair, shape[0], dim=0)

            if self.train_mode == "with_gt":
                desp_loss = self.descriptor_loss(
                    desp_0, desp_1, matched_idx, matched_valid, not_search_mask)
            elif self.train_mode == "with_precise_gt":
                matched_coords = data["matched_coords"].to(self.device)
                desp_1 = self._generate_batched_predict_descriptor(matched_coords, desp_1)
                desp_loss = self.descriptor_loss(desp_0, desp_1, matched_valid)
            else:
                assert False

            loss = point_loss + desp_loss

            if torch.isnan(loss):
                self.logger.error('loss is nan!')

            self.optimizer.zero_grad()
            loss.backward()

            self.optimizer.step()

            if i % self.log_freq == 0:

                point_loss_val = point_loss.item()
                desp_loss_val = desp_loss.item()
                loss_val = loss.item()

                self.logger.info(
                    "[Epoch:%2d][Step:%5d:%5d]: loss = %.4f, point_loss = %.4f, desp_loss = %.4f"
                    " one step cost %.4fs. " % (
                        epoch_idx, i, self.epoch_length,
                        loss_val,
                        point_loss_val,
                        desp_loss_val,
                        (time.time() - stime) / self.params.log_freq,
                    ))
                stime = time.time()

        # save the model
        if self.multi_gpus:
            torch.save(self.model.module.state_dict(), os.path.join(self.ckpt_dir, 'model_%02d.pt' % epoch_idx))
        else:
            torch.save(self.model.state_dict(), os.path.join(self.ckpt_dir, 'model_%02d.pt' % epoch_idx))

    def _train_with_residual(self, epoch_idx):
        self.model.train()
        stime = time.time()
        for i, data in enumerate(self.train_dataloader):

            image = data['image'].to(self.device)
            heatmap_gt = data['heatmap'].to(self.device)
            point_mask = data['point_mask'].to(self.device)

            warped_image = data['warped_image'].to(self.device)
            warped_heatmap_gt = data['warped_heatmap'].to(self.device)
            warped_point_mask = data['warped_point_mask'].to(self.device)

            matched_idx = data['matched_idx'].to(self.device)
            matched_valid = data['matched_valid'].to(self.device)
            not_search_mask = data['not_search_mask'].to(self.device)

            shape = image.shape

            image_pair = torch.cat((image, warped_image), dim=0)
            heatmap_gt_pair = torch.cat((heatmap_gt, warped_heatmap_gt), dim=0)
            point_mask_pair = torch.cat((point_mask, warped_point_mask), dim=0)
            heatmap_pred_pair, desp_deep_pair, desp_shallow_pair = self.model(image_pair)
            heatmap_pred_pair = heatmap_pred_pair.squeeze()

            point_loss = self.point_loss(heatmap_pred_pair, heatmap_gt_pair, point_mask_pair)

            desp_deep_0, desp_deep_1 = torch.split(desp_deep_pair, shape[0], dim=0)
            desp_shallow_0, desp_shallow_1 = torch.split(desp_shallow_pair, shape[0], dim=0)

            desp_deep_loss = self.descriptor_loss(
                desp_deep_0, desp_deep_1, matched_idx, matched_valid, not_search_mask)

            desp_shallow_loss = self.descriptor_loss(
                desp_shallow_0, desp_shallow_1, matched_idx, matched_valid, not_search_mask)

            loss = point_loss + desp_deep_loss + desp_shallow_loss

            if torch.isnan(loss):
                self.logger.error('loss is nan!')

            self.optimizer.zero_grad()
            loss.backward()

            self.optimizer.step()

            if i % self.log_freq == 0:

                point_loss_val = point_loss.item()
                desp_deep_loss_val = desp_deep_loss.item()
                desp_shallow_loss_val = desp_shallow_loss.item()
                loss_val = loss.item()

                self.logger.info(
                    "[Epoch:%2d][Step:%5d:%5d]: loss = %.4f, point_loss = %.4f, desp_deep_loss = %.4f, "
                    "desp_shallow_loss = %.4f"
                    " one step cost %.4fs. " % (
                        epoch_idx, i, self.epoch_length,
                        loss_val,
                        point_loss_val,
                        desp_deep_loss_val,
                        desp_shallow_loss_val,
                        (time.time() - stime) / self.params.log_freq,
                    ))
                stime = time.time()

        # save the model
        if self.multi_gpus:
            torch.save(self.model.module.state_dict(), os.path.join(self.ckpt_dir, 'model_%02d.pt' % epoch_idx))
        else:
            torch.save(self.model.state_dict(), os.path.join(self.ckpt_dir, 'model_%02d.pt' % epoch_idx))

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

    def test(self, ckpt_file):
        self._load_model_params(ckpt_file)

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
            first_image = data['first_image']
            second_image = data['second_image']
            gt_homography = data['gt_homography']
            image_type = data['image_type']

            # if image_type == "illumination":
            #     print("sikp")
            #     skip += 1
            #     continue

            image_pair = np.stack((first_image, second_image), axis=0)
            image_pair = torch.from_numpy(image_pair).to(torch.float).to(self.device).unsqueeze(dim=1)
            image_pair = image_pair*2./255. - 1.

            if self.network_arch == "residual":
                heatmap_pair, desp_pair, _ = self.model(image_pair)
            else:
                heatmap_pair, desp_pair = self.model(image_pair)

            heatmap_pair = torch.sigmoid(heatmap_pair)  # 用BCEloss
            prob_pair = spatial_nms(heatmap_pair, kernel_size=int(self.nms_threshold*2+1))

            desp_pair = desp_pair.detach().cpu().numpy()
            first_desp = desp_pair[0]
            second_desp = desp_pair[1]
            prob_pair = prob_pair.detach().cpu().numpy()
            first_prob = prob_pair[0, 0]
            second_prob = prob_pair[1, 0]

            # 得到对应的预测点
            if self.detection_mode == "use_network":
                first_point, first_point_num = self._generate_predict_point(first_prob, top_k=self.top_k)  # [n,2]
                second_point, second_point_num = self._generate_predict_point(second_prob, top_k=self.top_k)  # [m,2]
            elif self.detection_mode == "use_sift":
                first_point_cv, _ = self.sift.detectAndCompute(first_image, None)
                second_point_cv, _ = self.sift.detectAndCompute(second_image, None)
                first_point_num = len(first_point_cv)
                second_point_num = len(second_point_cv)
                first_point = self._cvpoint2numpy(first_point_cv)
                second_point = self._cvpoint2numpy(second_point_cv)
            else:
                self.logger.error("unrecognized detection_mode: %s" % self.detection_mode)
                assert False

            # debug use
            # first_image, second_image = torch.chunk(image_pair, 2, dim=0)
            # first_image = ((first_image.detach().cpu().squeeze() + 1) * 255. / 2.).to(torch.uint8).numpy()
            # second_image = ((second_image.detach().cpu().squeeze() + 1) * 255. / 2.).to(torch.uint8).numpy()
            # cat_image = np.concatenate((first_image, second_image), axis=1)
            # f_heatmap, s_heatmap = torch.chunk(heatmap_pair, 2, dim=0)
            # f_heatmap = f_heatmap.squeeze().detach().cpu()
            # s_heatmap = s_heatmap.squeeze().detach().cpu()
            # cat_heatmap = torch.cat((f_heatmap, s_heatmap), dim=1)
            # heatmap_image = self._debug_show(cat_heatmap, cat_image, show=False)

            # f_image_point = draw_image_keypoints(first_image, first_point, show=False, color=(0, 0, 255))
            # s_image_point = draw_image_keypoints(second_image, second_point, show=False, color=(0, 0, 255))
            # image_point = np.concatenate((f_image_point, s_image_point), axis=1)
            # cv.imwrite("/home/zhangyuyang/tmp_images/megpoint/image_%03d.jpg" % i, image_point)
            # heatmap_image_point = np.concatenate((heatmap_image, image_point), axis=0)
            # cv.imshow("all", heatmap_image_point)
            # cv.imshow("image_point", image_point)
            # cv.waitKey()

            if first_point_num <= 4 or second_point_num <= 4:
                print("skip this pair because there's little point!")
                skip += 1
                continue

            # 得到点对应的描述子
            select_first_desp = self._generate_predict_descriptor(first_point, first_desp)
            select_second_desp = self._generate_predict_descriptor(second_point, second_desp)

            # 得到匹配点
            matched_point = self.general_matcher(first_point, select_first_desp,
                                                 second_point, select_second_desp)

            if matched_point is None:
                print("skip this pair because there's no match point!")
                skip += 1
                continue

            # 计算得到单应变换
            if self.homo_pred_mode == "RANSAC":
                pred_homography, _ = cv.findHomography(matched_point[0][:, np.newaxis, ::-1],
                                                       matched_point[1][:, np.newaxis, ::-1], cv.RANSAC)
            elif self.homo_pred_mode == "LMEDS":
                pred_homography, _ = cv.findHomography(matched_point[0][:, np.newaxis, ::-1],
                                                       matched_point[1][:, np.newaxis, ::-1], cv.LMEDS)
            else:
                assert False
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
        # 计算估计的单应变换准确度
        illum_homo_acc, view_homo_acc, total_homo_acc = self._compute_total_metric(self.illum_homo_acc,
                                                                                   self.view_homo_acc)
        # 计算匹配的准确度
        illum_match_acc, view_match_acc, total_match_acc = self._compute_total_metric(self.illum_mma,
                                                                                      self.view_mma)

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

    def _debug_show(self, heatmap, image, show=False):
        heatmap = np.clip(heatmap, 0, 1)
        heatmap = heatmap.numpy() * 150
        # heatmap = cv.resize(heatmap, dsize=(self.width, self.height), interpolation=cv.INTER_LINEAR)
        heatmap = cv.applyColorMap(heatmap.astype(np.uint8), colormap=cv.COLORMAP_BONE).astype(np.float)
        # image = (image.squeeze().numpy() + 1) * 255 / 2
        hyper_image = np.clip(heatmap + image[:, :, np.newaxis], 0, 255).astype(np.uint8)
        if show:
            cv.imshow("heat&image", hyper_image)
            cv.waitKey()
        else:
            return hyper_image

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
            first_image = data['first_image']
            second_image = data['second_image']
            gt_homography = data['gt_homography']
            image_type = data['image_type']

            image_pair = np.stack((first_image, second_image), axis=0)
            image_pair = torch.from_numpy(image_pair).to(torch.float).to(self.device).unsqueeze(dim=1)
            image_pair = image_pair*2./255. - 1.

            if self.network_arch == "residual":
                heatmap_pair, desp_pair, _ = self.model(image_pair)
            else:
                heatmap_pair, desp_pair = self.model(image_pair)

            heatmap_pair = torch.sigmoid(heatmap_pair)
            prob_pair = spatial_nms(heatmap_pair, kernel_size=int(self.nms_threshold*2+1))

            desp_pair = desp_pair.detach().cpu().numpy()
            first_desp = desp_pair[0]
            second_desp = desp_pair[1]
            prob_pair = prob_pair.detach().cpu().numpy()
            first_prob = prob_pair[0, 0]
            second_prob = prob_pair[1, 0]

            # 得到对应的预测点
            first_point, first_point_num = self._generate_predict_point(first_prob, top_k=self.top_k)  # [n,2]
            second_point, second_point_num = self._generate_predict_point(second_prob, top_k=self.top_k)  # [m,2]

            if first_point_num <= 4 or second_point_num <= 4:
                print("skip this pair because there's little point!")
                skip += 1
                continue

            # 得到点对应的描述子
            select_first_desp = self._generate_predict_descriptor(first_point, first_desp)
            select_second_desp = self._generate_predict_descriptor(second_point, second_desp)

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

    def _generate_predict_point(self, prob, scale=None, top_k=0):
        point_idx = np.where(prob > self.detection_threshold)

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

    def _cvpoint2numpy(self, point_cv):
        """将opencv格式的特征点转换成numpy数组"""
        point_list = []
        for pt_cv in point_cv:
            point = np.array((pt_cv.pt[1], pt_cv.pt[0]))
            point_list.append(point)
        point_np = np.stack(point_list, axis=0)
        return point_np

    def _generate_predict_descriptor(self, point, desp):
        point = torch.from_numpy(point).to(torch.float)  # 由于只有pytorch有gather的接口，因此将点调整为pytorch的格式
        desp = torch.from_numpy(desp)
        dim, h, w = desp.shape
        desp = torch.reshape(desp, (dim, -1))
        desp = torch.transpose(desp, dim0=1, dim1=0)  # [h*w,256]
        offset = torch.ones_like(point) * 3.5  # offset代表中心点与左上角点的偏移

        # 下采样
        if self.train_mode == "with_gt":
            scaled_point = point / 8
        elif self.train_mode == "with_precise_gt":
            scaled_point = (point - offset) / 8
        else:
            assert False

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

    def _generate_batched_predict_descriptor(self, point, desp):
        bt, dim, h, w = desp.shape
        desp = torch.reshape(desp, (bt, dim, -1))
        desp = torch.transpose(desp, dim0=1, dim1=2)  # [bt,h*w,256]
        offset = torch.ones_like(point) * 3.5  # offset代表中心点与左上角点的偏移

        # 下采样
        scaled_point = (point - offset) / 8
        point_y = scaled_point[:, :, 0:1]  # [bt,n,1]
        point_x = scaled_point[:, :, 1:2]  # [bt,n,1]

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

        idx_00 = (x0_safe + y0_safe*w).to(torch.long).repeat((1, 1, dim))  # [bt,n,dim]
        idx_01 = (x0_safe + y1_safe*w).to(torch.long).repeat((1, 1, dim))
        idx_10 = (x1_safe + y0_safe*w).to(torch.long).repeat((1, 1, dim))
        idx_11 = (x1_safe + y1_safe*w).to(torch.long).repeat((1, 1, dim))
        idx_nearest = (x_nearest_safe + y_nearest_safe*w).to(torch.long).repeat((1, 1, dim))

        d_x = point_x - x0_safe
        d_y = point_y - y0_safe
        d_1_x = x1_safe - point_x
        d_1_y = y1_safe - point_y

        desp_00 = torch.gather(desp, dim=1, index=idx_00)
        desp_01 = torch.gather(desp, dim=1, index=idx_01)
        desp_10 = torch.gather(desp, dim=1, index=idx_10)
        desp_11 = torch.gather(desp, dim=1, index=idx_11)
        nearest_desp = torch.gather(desp, dim=1, index=idx_nearest)
        bilinear_desp = desp_00*d_1_x*d_1_y + desp_01*d_1_x*d_y + desp_10*d_x*d_1_y+desp_11*d_x*d_y

        # todo: 插值得到的描述子不再满足模值为1，强行归一化到模值为1，这里可能有问题
        condition = torch.eq(torch.norm(bilinear_desp, dim=2, keepdim=True), 0)
        interpolation_desp = torch.where(condition, nearest_desp, bilinear_desp)
        interpolation_norm = torch.norm(interpolation_desp, dim=2, keepdim=True)
        interpolation_desp = interpolation_desp / interpolation_norm
        interpolation_desp = interpolation_desp.transpose(1, 2).reshape((bt, 256, h, w))

        return interpolation_desp

    def _initialize_test_calculator(self, params):
        # 初始化验证算子
        self.logger.info('Initialize the homography accuracy calculator, correct_epsilon: %d' % self.correct_epsilon)
        self.logger.info('Initialize the repeatability calculator, detection_threshold: %.4f, correct_epsilon: %d'
                         % (self.detection_threshold, self.correct_epsilon))
        self.logger.info('Top k: %d' % self.top_k)

        self.illum_repeat = RepeatabilityCalculator(params.correct_epsilon)
        self.illum_repeat_mov = MovingAverage()

        self.view_repeat = RepeatabilityCalculator(params.correct_epsilon)
        self.view_repeat_mov = MovingAverage()

        self.illum_homo_acc = HomoAccuracyCalculator(params.correct_epsilon,
                                                     params.hpatch_height, params.hpatch_width)
        self.illum_homo_acc_mov = MovingAverage()

        self.view_homo_acc = HomoAccuracyCalculator(params.correct_epsilon,
                                                    params.hpatch_height, params.hpatch_width)
        self.view_homo_acc_mov = MovingAverage()

        self.illum_mma = MeanMatchingAccuracy(params.correct_epsilon)
        self.illum_mma_mov = MovingAverage()

        self.view_mma = MeanMatchingAccuracy(params.correct_epsilon)
        self.view_mma_mov = MovingAverage()

        # 初始化专门用于估计的单应变换较差的点匹配情况统计的算子
        self.view_bad_mma = MeanMatchingAccuracy(params.correct_epsilon)
        self.illum_bad_mma = MeanMatchingAccuracy(params.correct_epsilon)

        # 初始化用于浮点型描述子的测试方法
        self.illum_homo_acc_f = HomoAccuracyCalculator(params.correct_epsilon,
                                                       params.hpatch_height, params.hpatch_width)
        self.view_homo_acc_f = HomoAccuracyCalculator(params.correct_epsilon,
                                                      params.hpatch_height, params.hpatch_width)

        self.illum_mma_f = MeanMatchingAccuracy(params.correct_epsilon)
        self.view_mma_f = MeanMatchingAccuracy(params.correct_epsilon)

        # 初始化用于二进制描述子的测试方法
        self.illum_homo_acc_b = HomoAccuracyCalculator(params.correct_epsilon,
                                                       params.hpatch_height, params.hpatch_width)
        self.view_homo_acc_b = HomoAccuracyCalculator(params.correct_epsilon,
                                                      params.hpatch_height, params.hpatch_width)

        self.illum_mma_b = MeanMatchingAccuracy(params.correct_epsilon)
        self.view_mma_b = MeanMatchingAccuracy(params.correct_epsilon)

        self.point_statistics = PointStatistics()

    def test_debug(self, ckpt_file):
        self._load_model_params(ckpt_file)

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
        illum_skip = 0
        view_skip = 0
        bad = 0

        for i, data in enumerate(self.test_dataset):
            first_image = data['first_image']
            second_image = data['second_image']
            gt_homography = data['gt_homography']
            image_type = data['image_type']

            # if image_type == "illumination":
            #     print("sikp")
            #     skip += 1
            #     continue

            image_pair = np.stack((first_image, second_image), axis=0)
            image_pair = torch.from_numpy(image_pair).to(torch.float).to(self.device).unsqueeze(dim=1)
            image_pair = image_pair * 2. / 255. - 1.

            if self.network_arch == "residual":
                heatmap_pair, desp_pair, _ = self.model(image_pair)
            else:
                heatmap_pair, desp_pair = self.model(image_pair)

            heatmap_pair = torch.sigmoid(heatmap_pair)  # 用BCEloss
            prob_pair = spatial_nms(heatmap_pair, kernel_size=int(self.nms_threshold * 2 + 1))

            desp_pair = desp_pair.detach().cpu().numpy()
            first_desp = desp_pair[0]
            second_desp = desp_pair[1]
            prob_pair = prob_pair.detach().cpu().numpy()
            first_prob = prob_pair[0, 0]
            second_prob = prob_pair[1, 0]

            # 得到对应的预测点
            if self.detection_mode == "use_network":
                first_point, first_point_num = self._generate_predict_point(first_prob, top_k=self.top_k)  # [n,2]
                second_point, second_point_num = self._generate_predict_point(second_prob, top_k=self.top_k)  # [m,2]
            else:
                self.logger.error("unrecognized detection_mode: %s" % self.detection_mode)
                assert False

            if first_point_num <= 4 or second_point_num <= 4:
                print("skip this pair because there's little point!")
                if image_type == "illumination":
                    illum_skip += 1
                else:
                    view_skip += 1
                continue

            # 得到点对应的描述子
            select_first_desp = self._generate_predict_descriptor(first_point, first_desp)
            select_second_desp = self._generate_predict_descriptor(second_point, second_desp)

            # 得到匹配点
            matched_point = self.general_matcher(first_point, select_first_desp,
                                                 second_point, select_second_desp)
            # 作弊
            correct_first_list = []
            correct_second_list = []
            xy_first_point = matched_point[0][:, ::-1]
            xy1_first_point = np.concatenate((xy_first_point, np.ones((xy_first_point.shape[0], 1))), axis=1)
            xyz_second_point = np.matmul(gt_homography, xy1_first_point[:, :, np.newaxis])[:, :, 0]
            xy_second_point = xyz_second_point[:, :2] / xyz_second_point[:, 2:3]
            diff = np.linalg.norm(xy_second_point - matched_point[1][:, ::-1], axis=1)
            for j in range(xy_first_point.shape[0]):
                if diff[j] < 1:
                    correct_first_list.append(matched_point[0][j])
                    correct_second_list.append(matched_point[1][j])
            # if len(correct_first_list) < 20:
            #     print("skip this pair because there's no good match!")
            #     if image_type == "illumination":
            #         illum_skip += 1
            #     else:
            #         view_skip += 1
            #     continue
            matched_point = (
                np.stack(correct_first_list, axis=0),
                np.stack(correct_second_list, axis=0))

            cv_first_point = []
            cv_second_point = []
            cv_matched_list = []
            if len(correct_first_list) > 0:
                for j in range(len(correct_first_list)):
                    cv_point = cv.KeyPoint()
                    cv_point.pt = tuple(correct_first_list[j][::-1])
                    cv_first_point.append(cv_point)

                    cv_point = cv.KeyPoint()
                    cv_point.pt = tuple(correct_second_list[j][::-1])
                    cv_second_point.append(cv_point)

                    cv_match = cv.DMatch()
                    cv_match.queryIdx = j
                    cv_match.trainIdx = j
                    cv_matched_list.append(cv_match)

            if matched_point is None:
                print("skip this pair because there's no match point!")
                if image_type == "illumination":
                    illum_skip += 1
                else:
                    view_skip += 1
                continue

            # 计算得到单应变换
            if self.homo_pred_mode == "RANSAC":
                pred_homography, _ = cv.findHomography(matched_point[0][:, np.newaxis, ::-1],
                                                       matched_point[1][:, np.newaxis, ::-1], cv.RANSAC)
            elif self.homo_pred_mode == "LMEDS":
                pred_homography, _ = cv.findHomography(matched_point[0][:, np.newaxis, ::-1],
                                                       matched_point[1][:, np.newaxis, ::-1], cv.LMEDS)
            else:
                assert False
            if pred_homography is None:
                print("skip this pair because no homo can be predicted!.")
                if image_type == "illumination":
                    illum_skip += 1
                else:
                    view_skip += 1
                continue

            # 对单样本进行测评
            if image_type == 'illumination':
                self.illum_repeat.update(first_point, second_point, gt_homography)
                correct, diff = self.illum_homo_acc.update(pred_homography, gt_homography, True)
                self.illum_mma.update(gt_homography, matched_point)

                if not correct:
                    self.illum_bad_mma.update(gt_homography, matched_point)
                    # self.logger.info("diff between gt & pred is %.4f" % diff)
                    bad += 1

                    if len(cv_matched_list) != 0:
                        matched_image = cv.drawMatches(first_image, cv_first_point, second_image, cv_second_point,
                                                       cv_matched_list, None)
                        metric_str = "diff between gt & pred is %.4f, correct match: %d/ total: %d, %.4f" % (
                            diff, len(correct_first_list), matched_point[0].shape[0],
                            len(correct_first_list) / matched_point[0].shape[0]
                        )
                        cv.putText(matched_image, metric_str, (0, 40), cv.FONT_HERSHEY_COMPLEX, fontScale=0.8,
                                   color=(0, 0, 255), thickness=2)
                        # cv.imshow("matched_image", matched_image)
                        # cv.waitKey()
                        cv.imwrite("/home/zhangyuyang/tmp_images/megpoint_bad/image_%03d.jpg" % i, matched_image)

            elif image_type == 'viewpoint':
                self.view_repeat.update(first_point, second_point, gt_homography)
                correct, diff = self.view_homo_acc.update(pred_homography, gt_homography, True)
                self.view_mma.update(gt_homography, matched_point)

                if not correct:
                    self.view_bad_mma.update(gt_homography, matched_point)
                    # self.logger.info("diff between gt & pred is %.4f" % diff)
                    bad += 1

                    if len(cv_matched_list) != 0:
                        matched_image = cv.drawMatches(first_image, cv_first_point, second_image, cv_second_point,
                                                       cv_matched_list, None)
                        metric_str = "diff between gt & pred is %.4f, correct match: %d/ total: %d, %.4f" % (
                            diff, len(correct_first_list), matched_point[0].shape[0],
                            len(correct_first_list) / matched_point[0].shape[0]
                        )
                        cv.putText(matched_image, metric_str, (0, 40), cv.FONT_HERSHEY_COMPLEX, fontScale=0.8,
                                   color=(0, 0, 255), thickness=2)
                        cv.imwrite("/home/zhangyuyang/tmp_images/megpoint_bad/image_%03d.jpg" % i, matched_image)

            else:
                print("The image type magicpoint_tester.test(ckpt_file)must be one of illumination of viewpoint ! "
                      "Please check !")
                assert False

            # 统计检测的点的数目
            self.point_statistics.update((first_point_num + second_point_num) / 2.)

            if i % 10 == 0:
                print("Having tested %d samples, which takes %.3fs" % (i, (time.time() - start_time)))
                start_time = time.time()
            count += 1
            # if count % 1000 == 0:
            #     break

        self.logger.info("Totally skip {} illumination samples and {} view samples.".format(illum_skip, view_skip))
        # 计算各自的重复率以及总的重复率
        illum_repeat, view_repeat, total_repeat = self._compute_total_metric(self.illum_repeat,
                                                                             self.view_repeat)
        # 计算估计的单应变换准确度
        illum_homo_acc, view_homo_acc, total_homo_acc = self._compute_total_metric(self.illum_homo_acc,
                                                                                   self.view_homo_acc)
        # 计算匹配的准确度
        illum_match_acc, view_match_acc, total_match_acc = self._compute_total_metric(self.illum_mma,
                                                                                      self.view_mma)

        # 计算匹配外点的分布情况
        illum_dis, view_dis = self._compute_match_outlier_distribution(self.illum_mma,
                                                                       self.view_mma)

        illum_bad_dis, view_bad_dis = self._compute_match_outlier_distribution(self.illum_bad_mma,
                                                                               self.view_bad_mma)

        # 统计最终的检测点数目的平均值和方差
        point_avg, point_std = self.point_statistics.average()

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



