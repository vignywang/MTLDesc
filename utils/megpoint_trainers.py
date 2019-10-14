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

from nets.megpoint_net import MegPointNet
from nets.megpoint_net import STMegPointNet
from nets.megpoint_net import Discriminator

from data_utils.megpoint_dataset import AdaptionDataset, LabelGenerator
from data_utils.coco_dataset import COCOMegPointAdaptionDataset
from data_utils.coco_dataset import COCOMegPointAdversarialDataset
from data_utils.coco_dataset import COCOMegPointSelfTrainingDataset
from data_utils.coco_dataset import COCOMegPointRawDataset
from data_utils.synthetic_dataset import SyntheticAdversarialDataset
from data_utils.synthetic_dataset import SyntheticValTestDataset
from data_utils.hpatch_dataset import HPatchDataset
from data_utils.dataset_tools import HomographyAugmentation
from data_utils.dataset_tools import InfiniteDataLoader

from utils.evaluation_tools import RepeatabilityCalculator
from utils.evaluation_tools import MovingAverage
from utils.evaluation_tools import PointStatistics
from utils.evaluation_tools import HomoAccuracyCalculator
from utils.evaluation_tools import MeanMatchingAccuracy
from utils.evaluation_tools import mAPCalculator
from utils.utils import spatial_nms
from utils.utils import DescriptorTripletLoss
from utils.utils import Matcher


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
        model_dict = self.model.state_dict()
        pretrain_dict = torch.load(ckpt_file, map_location=self.device)
        model_dict.update(pretrain_dict)
        self.model.load_state_dict(model_dict)

        return True


class MegPointSeflTrainingTrainer(MegPointTrainerTester):

    def __init__(self, params):
        super(MegPointSeflTrainingTrainer, self).__init__(params)

        # 读取与该训练类有关的参数
        self.round_num = params.round_num
        self.epoch_each_round = params.epoch_each_round
        self.raw_batch_size = params.raw_batch_size * self.gpu_count
        self.initial_portion = params.initial_portion
        self.portion = self.initial_portion
        self.src_detector_weight = params.src_detector_weight
        self.tgt_detector_weight = params.tgt_detector_weight

        # 初始化源数据与目标数据
        self.source_dataset = SyntheticAdversarialDataset(params)
        self.target_dataset = COCOMegPointSelfTrainingDataset(params)
        self.target_raw_dataset = COCOMegPointRawDataset(params)

        self.epoch_length = min(
            (len(self.source_dataset)//self.batch_size),
            (len(self.target_dataset)//self.batch_size)
        )

        self.raw_epoch_length = len(self.target_raw_dataset) // self.raw_batch_size

        self.source_dataloader = DataLoader(
            self.source_dataset,
            self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            drop_last=True
        )

        self.target_dataloader = DataLoader(
            self.target_dataset,
            self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            drop_last=True
        )

        self.target_raw_dataloader = DataLoader(
            self.target_raw_dataset,
            self.raw_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=False
        )

        self.test_dataset = HPatchDataset(params)
        self.synthetic_test_dataset = SyntheticValTestDataset(params, dataset_type='validation')
        self.test_length = len(self.test_dataset)

        # 初始化各种验证算子
        self._initialize_test_calculator(params)

        # 初始化匹配算子
        self.general_matcher = Matcher('float')

        # 初始化模型
        self.model = STMegPointNet()

        # 恢复预训练的模型
        self._load_model_params(params.magicpoint_ckpt)

        # 初始化loss算子
        self.detector_loss = torch.nn.CrossEntropyLoss(reduction='none')
        self.descriptor_loss = DescriptorTripletLoss(device=self.device)

        # 若有多gpu设置则加载多gpu
        if self.multi_gpus:
            self.model = torch.nn.DataParallel(self.model)
        self.model = self.model.to(self.device)

        # 初始化优化器算子
        self.model_optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        # 初始化学习率调整算子
        self.model_scheduler = torch.optim.lr_scheduler.StepLR(self.model_optimizer, step_size=100, gamma=0.1)

    @staticmethod
    def _compute_masked_loss(unmasked_loss, mask):
        total_num = torch.sum(mask, dim=(-1, -2))
        loss = torch.sum(mask*unmasked_loss, dim=(-1, -2)) / total_num
        loss = torch.mean(loss)
        return loss

    def train(self):
        start_time = time.time()

        # start training
        for i in range(self.round_num):

            # label
            self._label_one_round(i)

            # train & validation
            self._train_one_round(i)

            # adjust learning rate
            # self.scheduler.step(i)

        end_time = time.time()
        self.logger.info("The whole training process takes %.3f h" % ((end_time - start_time)/3600))

    def _compute_threshold(self, portion):
        self.model.eval()
        # 计算每一张图像的关键点的概率图，并统计所有的概率
        self.logger.info("Begin to compute threshold of portion: %.4f" % portion)
        partial_idx = int(self.raw_batch_size*self.params.height*self.params.width*portion)
        start_time = time.time()
        all_prob = np.empty((0,))
        for i, data in enumerate(self.target_raw_dataloader):
            image = data["image"].to(self.device)
            _, prob, _ = self.model(image)  # [bt,64,h/8,w/8]
            prob = f.pixel_shuffle(prob, 8)
            prob = spatial_nms(prob, kernel_size=int(self.nms_threshold * 2 + 1)).squeeze().reshape((-1,))  # [bt*h*w]
            prob, _ = torch.sort(prob, descending=True)
            prob = prob[:partial_idx]
            all_prob = np.concatenate((all_prob, prob.detach().cpu().numpy()))

            if i % 50 == 0:
                self.logger.info(
                    "Batch %04d/%04d have been computed , %.3fs/step" %
                    (
                        i,
                        self.raw_epoch_length,
                        (time.time()-start_time)/50
                    )
                )
                start_time = time.time()

        all_prob = np.sort(all_prob)
        threshold = all_prob[0]
        self.logger.info("Computing Done! the threshold is: %.4f" % threshold)

        return threshold

    def _label_one_round(self, round_idx):
        self.logger.info("-----------------------------------------------------")
        self.logger.info("Start to label the whole target data")

        self.model.eval()

        # 根据当前的round数计算当前的portion
        self.portion = min(self.initial_portion*1.01**round_idx, 0.015)
        label_threshold = self._compute_threshold(self.portion)

        # 再次对数据集中每一张图像进行预测且标注
        self.logger.info("Begin to label all target data by threshold: %.4f" % label_threshold)
        self.target_dataset.reset()
        start_time = time.time()
        for i, data in enumerate(self.target_raw_dataloader):
            image = data["image"].to(self.device)
            _, prob, _ = self.model(image)
            image = ((image + 1) * 255. / 2.).to(torch.uint8).squeeze().cpu().numpy()
            prob = f.pixel_shuffle(prob, 8)
            prob = spatial_nms(prob, kernel_size=int(self.nms_threshold * 2 + 1)).squeeze().detach().cpu().numpy()
            batch_size = prob.shape[0]

            for j in range(batch_size):
                # 得到对应的预测点
                cur_point, cur_point_num = self._generate_predict_point_by_threshold(prob[j], threshold=label_threshold)
                cur_image = image[j]
                self.target_dataset.append(cur_image, cur_point)

            if i % 50 == 0:
                self.logger.info(
                    "Batch %04d/%04d have been labeled, %.3fs/step" %
                    (
                        i,
                        self.raw_epoch_length,
                        (time.time()-start_time)/50
                    )
                )
                start_time = time.time()

        self.logger.info("Labeling done.")
        self.logger.info("-----------------------------------------------------")

    def _train_one_round(self, round_idx):
        self.logger.info("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
        self.logger.info("Start to train round %2d:" % round_idx)

        for i in range(self.epoch_each_round):
            self._train_one_epoch(int(i+round_idx*self.epoch_each_round))

            test_threshold = self._compute_threshold(self.portion)
            self._validate_one_epoch_by_threshold(int(i+round_idx*self.epoch_each_round), test_threshold)

        self.logger.info("Training round %2d done." % round_idx)
        self.logger.info("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")

    def _train_one_epoch(self, epoch_idx):
        self.logger.info("-----------------------------------------------------")
        self.logger.info("Start to train epoch %2d:" % epoch_idx)
        stime = time.time()

        self.model.train()

        sourceloader_iter = enumerate(self.source_dataloader)
        targetloader_iter = enumerate(self.target_dataloader)
        for i in range(self.epoch_length):

            self.model_optimizer.zero_grad()

            _, src_data = sourceloader_iter.__next__()
            _, tgt_data = targetloader_iter.__next__()

            # 1、读取带标签的源数据并训练关键点检测算子
            src_image = src_data["image"].to(self.device)
            src_label = src_data["label"].to(self.device)
            src_mask = src_data["mask"].to(self.device)

            # 送入网络统一运算，充分利用GPU并行计算
            src_logit, *_ = self.model(src_image)

            # 计算关键点检测loss
            unmasked_detector_loss = self.detector_loss(src_logit, src_label)
            src_detector_loss = self._compute_masked_loss(unmasked_detector_loss, src_mask)
            src_loss = self.src_detector_weight * src_detector_loss

            if torch.isnan(src_detector_loss):
                self.logger.error("detector_loss is nan!")
                assert False
            src_loss.backward()

            # 2、读取伪标签的目标数据同时计算描述子loss，关键点检测算子
            tgt_image = tgt_data['image'].to(self.device)
            tgt_label = tgt_data['label'].to(self.device)
            tgt_mask = tgt_data['mask'].to(self.device)

            tgt_warped_image = tgt_data['warped_image'].to(self.device)
            tgt_warped_label = tgt_data['warped_label'].to(self.device)
            tgt_warped_mask = tgt_data['warped_mask'].to(self.device)

            tgt_matched_idx = tgt_data['matched_idx'].to(self.device)
            tgt_matched_valid = tgt_data['matched_valid'].to(self.device)
            tgt_not_search_mask = tgt_data['not_search_mask'].to(self.device)

            shape = tgt_image.shape

            tgt_image_pair = torch.cat((tgt_image, tgt_warped_image), dim=0)
            tgt_label_pair = torch.cat((tgt_label, tgt_warped_label), dim=0)
            tgt_mask_pair = torch.cat((tgt_mask, tgt_warped_mask), dim=0)
            tgt_logit_pair, _, tgt_desp_pair = self.model(tgt_image_pair)

            unmasked_detector_loss = self.detector_loss(tgt_logit_pair, tgt_label_pair)
            tgt_detector_loss = self._compute_masked_loss(unmasked_detector_loss, tgt_mask_pair)

            tgt_desp_0, tgt_desp_1 = torch.split(tgt_desp_pair, shape[0], dim=0)

            tgt_desp_loss = self.descriptor_loss(
                tgt_desp_0, tgt_desp_1, tgt_matched_idx, tgt_matched_valid, tgt_not_search_mask)

            tgt_loss = self.tgt_detector_weight * tgt_detector_loss + tgt_desp_loss

            if torch.isnan(tgt_loss):
                self.logger.error("tgt_loss is nan!")
                assert False
            tgt_loss.backward()

            # 更新网络参数
            self.model_optimizer.step()

            if i % self.log_freq == 0:

                step = int(i + epoch_idx*self.epoch_length)

                src_detector_loss_val = src_detector_loss.item()
                tgt_detector_loss_val = tgt_detector_loss.item()
                tgt_desp_loss_val = tgt_desp_loss.item()

                self.summary_writer.add_scalar("detector_loss/src", src_detector_loss_val, global_step=step)
                self.summary_writer.add_scalar("detector_loss/tgt", tgt_detector_loss_val, global_step=step)
                self.summary_writer.add_scalar("descriptor_loss", tgt_desp_loss_val, global_step=step)
                self.logger.info(
                    "[Epoch:%2d][Step:%5d:%5d]: "
                    "src_detector_loss = %.4f, "
                    "tgt_detector_loss = %.4f, "
                    "desp_loss = %.4f, "
                    "%.2fs/step. " % (
                        epoch_idx,
                        i,
                        self.epoch_length,
                        src_detector_loss_val,
                        tgt_detector_loss_val,
                        tgt_desp_loss_val,
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

    def _validate_one_epoch_by_threshold(self, epoch_idx, threshold):
        self.logger.info("*****************************************************")
        self.logger.info("Validating epoch %2d begin:" % epoch_idx)

        self._test_model_general(epoch_idx, threshold)

        illum_homo_moving_acc = self.illum_homo_acc_mov.average()
        view_homo_moving_acc = self.view_homo_acc_mov.average()

        illum_mma_moving_acc = self.illum_mma_mov.average()
        view_mma_moving_acc = self.view_mma_mov.average()

        illum_repeat_moving_acc = self.illum_repeat_mov.average()
        view_repeat_moving_acc = self.view_repeat_mov.average()

        current_size = self.view_mma_mov.current_size()

        mAP, test_data, count = self._test_func(self.synthetic_test_dataset)
        self.logger.info("The mean Average Precision : %.4f of %d samples" % (mAP, count))

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

    def _test_model_general(self, epoch_idx, threshold):

        self.model.eval()
        # 重置测评算子参数
        self.illum_repeat.reset()
        self.illum_homo_acc.reset()
        self.illum_mma.reset()
        self.view_repeat.reset()
        self.view_homo_acc.reset()
        self.view_mma.reset()
        self.point_statistics.reset()

        start_time = time.time()
        count = 0
        skip = 0

        self.logger.info("Test, detection_threshold=%.4f" % threshold)

        for i, data in enumerate(self.test_dataset):
            first_image = data['first_image']
            second_image = data['second_image']
            gt_homography = data['gt_homography']
            image_type = data['image_type']

            image_pair = np.stack((first_image, second_image), axis=0)
            image_pair = torch.from_numpy(image_pair).to(torch.float).to(self.device).unsqueeze(dim=1)
            # debug released mode use
            # image_pair /= 255.
            image_pair = image_pair * 2. / 255. - 1.

            _, prob_pair, desp_pair = self.model(image_pair)
            prob_pair = f.pixel_shuffle(prob_pair, 8)
            prob_pair = spatial_nms(prob_pair, kernel_size=int(self.nms_threshold * 2 + 1))

            desp_pair = desp_pair.detach().cpu().numpy()
            first_desp = desp_pair[0]
            second_desp = desp_pair[1]
            prob_pair = prob_pair.detach().cpu().numpy()
            first_prob = prob_pair[0, 0]
            second_prob = prob_pair[1, 0]

            # 得到对应的预测点
            first_point, first_point_num = self._generate_predict_point(
                first_prob, threshold=threshold, top_k=self.top_k)  # [n,2]
            second_point, second_point_num = self._generate_predict_point(
                second_prob, threshold=threshold, top_k=self.top_k)  # [m,2]

            # 若未能检测出点，则跳过这一对测试样本
            if first_point_num == 0 or second_point_num == 0:
                skip += 1
                continue

            # 得到点对应的描述子
            select_first_desp = self._generate_predict_descriptor(first_point, first_desp)
            select_second_desp = self._generate_predict_descriptor(second_point, second_desp)

            # 得到匹配点
            matched_point = self.general_matcher(first_point, select_first_desp,
                                                 second_point, select_second_desp)
            if matched_point is None:
                skip += 1
                continue

            # 计算得到单应变换
            pred_homography, _ = cv.findHomography(matched_point[0][:, np.newaxis, ::-1],
                                                   matched_point[1][:, np.newaxis, ::-1], cv.RANSAC)

            if pred_homography is None:
                skip += 1
                continue

            # 对单样本进行测评
            if image_type == 'illumination':
                self.illum_repeat.update(first_point, second_point, gt_homography)
                self.illum_homo_acc.update(pred_homography, gt_homography)
                self.illum_mma.update(gt_homography, matched_point)

            elif image_type == 'viewpoint':
                self.view_repeat.update(first_point, second_point, gt_homography)
                self.view_homo_acc.update(pred_homography, gt_homography)
                self.view_mma.update(gt_homography, matched_point)

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

        # 统计最终的检测点数目的平均值和方差
        point_avg, point_std = self.point_statistics.average()

        self.logger.info("Homography Accuracy: illumination: %.4f, viewpoint: %.4f, total: %.4f" %
                         (illum_homo_acc, view_homo_acc, total_homo_acc))
        self.logger.info("Mean Matching Accuracy: illumination: %.4f, viewpoint: %.4f, total: %.4f " %
                         (illum_match_acc, view_match_acc, total_match_acc))
        self.logger.info("Repeatability: illumination: %.4f, viewpoint: %.4f, total: %.4f" %
                         (illum_repeat, view_repeat, total_repeat))
        self.logger.info("Detection point, average: %.4f, variance: %.4f" % (point_avg, point_std))

        self.summary_writer.add_scalar("illumination/Homography_Accuracy", illum_homo_acc, epoch_idx)
        self.summary_writer.add_scalar("illumination/Mean_Matching_Accuracy", illum_match_acc, epoch_idx)
        self.summary_writer.add_scalar("illumination/Repeatability", illum_repeat, epoch_idx)
        self.summary_writer.add_scalar("viewpoint/Homography_Accuracy", view_homo_acc, epoch_idx)
        self.summary_writer.add_scalar("viewpoint/Mean_Matching_Accuracy", view_match_acc, epoch_idx)
        self.summary_writer.add_scalar("viewpoint/Repeatability", view_repeat, epoch_idx)

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
            # 得到原始的经压缩的概率图，概率图每个通道64维，对应空间每个像素是否为关键点的概率
            _, prob, _ = self.model(image)
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

        return mAP, test_data, count

    @staticmethod
    def _compute_total_metric(illum_metric, view_metric):
        illum_acc, illum_sum, illum_num = illum_metric.average()
        view_acc, view_sum, view_num = view_metric.average()
        return illum_acc, view_acc, (illum_sum+view_sum)/(illum_num+view_num+1e-4)

    def _generate_predict_point_by_threshold(self, prob, threshold):
        point_idx = np.where(prob > threshold)

        if point_idx[0].size == 0:
            point = np.empty((0, 2), dtype=np.int64)
            point_num = 0
        else:
            point = np.stack(point_idx, axis=1)  # [n,2]
            point_num = point.shape[0]

        return point, point_num

    def _generate_predict_point(self, prob, threshold, scale=None, top_k=0):
        point_idx = np.where(prob > threshold)
        prob = prob[point_idx]
        sorted_idx = np.argsort(prob)[::-1]
        if sorted_idx.shape[0] >= top_k:
            sorted_idx = sorted_idx[:top_k]

        point = np.stack(point_idx, axis=1)  # [n,2]
        top_k_point = []
        for idx in sorted_idx:
            top_k_point.append(point[idx])
        if len(top_k_point) <= 1:
            return None, 0
        point = np.stack(top_k_point, axis=0)
        point_num = point.shape[0]

        if scale is not None:
            point = point*scale
        return point, point_num

    def _generate_predict_descriptor(self, point, desp):
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

    def _initialize_test_calculator(self, params):
        # 初始化验证算子
        self.logger.info('Initialize the homography accuracy calculator, correct_epsilon: %d' % self.correct_epsilon)
        self.logger.info('Initialize the repeatability calculator, correct_epsilon: %d'
                         % self.correct_epsilon)
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
        self.mAP_calculator = mAPCalculator()


class MegPointAdversarialTrainer(MegPointTrainerTester):

    def __init__(self, params):
        super(MegPointAdversarialTrainer, self).__init__(params)

        # 初始化源数据与目标数据
        self.source_dataset = SyntheticAdversarialDataset(params)
        self.target_dataset = COCOMegPointAdversarialDataset(params)

        self.epoch_length = max(
            (len(self.source_dataset)//self.batch_size),
            (len(self.target_dataset)//self.batch_size)
        )

        self.source_dataloader = InfiniteDataLoader(
            self.source_dataset,
            self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            drop_last=True
        )
        self.target_dataloader = InfiniteDataLoader(
            self.target_dataset,
            self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            drop_last=True
        )

        self.test_dataset = HPatchDataset(params)
        self.synthetic_test_dataset = SyntheticValTestDataset(params, dataset_type='validation')
        self.test_length = len(self.test_dataset)

        # 初始化各种验证算子
        self._initialize_test_calculator(params)

        # 初始化匹配算子
        self.general_matcher = Matcher('float')

        # 初始化模型
        self.model = MegPointNet()
        self.discriminator = Discriminator()

        # self._load_model_params(params.magicpoint_ckpt)

        # 初始化loss算子
        self.detector_loss = torch.nn.CrossEntropyLoss(reduction='none')
        self.descriptor_loss = DescriptorTripletLoss(device=self.device)
        self.discriminator_loss = torch.nn.BCEWithLogitsLoss()

        # 初始化loss权重
        self.discriminator_weight = params.discriminator_weight

        # 若有多gpu设置则加载多gpu
        if self.multi_gpus:
            self.model = torch.nn.DataParallel(self.model)
            self.discriminator = torch.nn.DataParallel(self.discriminator)
        self.model = self.model.to(self.device)
        self.discriminator = self.discriminator.to(self.device)

        # 初始化优化器算子
        self.model_optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.discriminator_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=self.lr)

        # 初始化学习率调整算子
        self.model_scheduler = torch.optim.lr_scheduler.StepLR(self.model_optimizer, step_size=100, gamma=0.1)
        self.discriminator_scheduler = torch.optim.lr_scheduler.StepLR(
            self.discriminator_optimizer, step_size=100, gamma=0.1
        )

    @staticmethod
    def _compute_masked_loss(unmasked_loss, mask):
        total_num = torch.sum(mask, dim=(-1, -2))
        loss = torch.sum(mask*unmasked_loss, dim=(-1, -2)) / total_num
        loss = torch.mean(loss)
        return loss

    def _train_one_epoch(self, epoch_idx):
        self.logger.info("-----------------------------------------------------")
        self.logger.info("Start to train epoch %2d:" % epoch_idx)
        stime = time.time()

        self.model.train()
        self.discriminator.train()

        sourceloader_iter = enumerate(self.source_dataloader)
        targetloader_iter = enumerate(self.target_dataloader)
        for i in range(self.epoch_length):

            self.model_optimizer.zero_grad()
            # self.discriminator_optimizer.zero_grad()

            # # 固定discriminator的参数不变
            # for param in self.discriminator.parameters():
            #     param.requires_grad = False

            _, src_data = sourceloader_iter.__next__()
            _, tgt_data = targetloader_iter.__next__()

            # 1、读取带标签的源数据并训练关键点检测算子
            src_image = src_data["image"].to(self.device)
            src_label = src_data["label"].to(self.device)
            src_mask = src_data["mask"].to(self.device)

            # 送入网络统一运算，充分利用GPU并行计算
            src_logit, _, _, src_feature = self.model(src_image)

            # 计算关键点检测loss
            unmasked_detector_loss = self.detector_loss(src_logit, src_label)
            detector_loss = self._compute_masked_loss(unmasked_detector_loss, src_mask)

            if torch.isnan(detector_loss):
                self.logger.error("detector_loss is nan!")
                assert False
            detector_loss.backward()

            # 2、读取无标签的目标数据计算描述子loss,对抗loss，用于减少src与tgt之间的domain shift
            tgt_image = tgt_data["image"].to(self.device)
            tgt_warped_image = tgt_data["warped_image"].to(self.device)

            tgt_matched_idx = tgt_data['matched_idx'].to(self.device)
            tgt_matched_valid = tgt_data['matched_valid'].to(self.device)
            tgt_not_search_mask = tgt_data['not_search_mask'].to(self.device)

            tgt_image_pair = torch.cat((tgt_image, tgt_warped_image), dim=0)
            _, _, tgt_desp_pair, tgt_feature_pair = self.model(tgt_image_pair)

            tgt_desp_0, tgt_desp_1 = torch.chunk(tgt_desp_pair, 2, dim=0)
            tgt_desp_loss = self.descriptor_loss(
                tgt_desp_0, tgt_desp_1, tgt_matched_idx, tgt_matched_valid, tgt_not_search_mask)

            # tgt_adverse_label = torch.zeros((self.batch_size*2, 1), dtype=torch.float, device=self.device)
            # tgt_adverse_logit = self.discriminator(tgt_feature_pair)
            # tgt_adverse_loss = self.discriminator_loss(tgt_adverse_logit, tgt_adverse_label)

            # tgt_loss = tgt_desp_loss + self.discriminator_weight*tgt_adverse_loss
            tgt_loss = tgt_desp_loss

            if torch.isnan(tgt_loss):
                self.logger.error("tgt_loss is nan!")
                assert False
            tgt_loss.backward()

            # 3、计算判别器loss并更新
            # for param in self.discriminator.parameters():
            #     param.requires_grad = True
            # feature_cats = torch.cat((src_feature, tgt_feature_pair), dim=0).detach()
            # d_logit = self.discriminator(feature_cats)
            # d_src_label = torch.zeros((self.batch_size, 1), dtype=torch.float, device=self.device)
            # d_tgt_label = torch.ones((self.batch_size*2, 1), dtype=torch.float, device=self.device)
            # d_label = torch.cat((d_src_label, d_tgt_label), dim=0)

            # d_loss = self.discriminator_loss(d_logit, d_label)
            # if torch.isnan(d_loss):
            #     self.logger.error("d_loss is nan!")
            #     assert False
            # d_loss.backward()

            # 更新网络参数
            self.model_optimizer.step()
            self.discriminator_optimizer.step()

            if i % self.log_freq == 0:

                step = int(i + epoch_idx*self.epoch_length)

                detector_loss_val = detector_loss.item()
                desp_loss_val = tgt_desp_loss.item()
                adverse_loss_val = 0  # tgt_adverse_loss.item()
                d_loss_val = 0  # d_loss.item()

                self.summary_writer.add_scalar("loss/detector_loss", detector_loss_val, global_step=step)
                self.summary_writer.add_scalar("loss/desp_loss", desp_loss_val, global_step=step)
                self.summary_writer.add_scalar("loss/adverse_loss", adverse_loss_val, global_step=step)
                self.summary_writer.add_scalar("loss/discriminator_loss", d_loss_val, global_step=step)
                self.logger.info(
                    "[Epoch:%2d][Step:%5d:%5d]: "
                    "detector_loss = %.4f, "
                    "desp_loss = %.4f, "
                    "adverse_loss = %.4f, "
                    "discriminator_loss = %.4f, "
                    "%.2fs/step. " % (
                        epoch_idx,
                        i,
                        self.epoch_length,
                        detector_loss_val,
                        desp_loss_val,
                        adverse_loss_val,
                        d_loss_val,
                        (time.time() - stime) / self.params.log_freq,
                    ))
                stime = time.time()

        # save the model
        if self.multi_gpus:
            torch.save(self.model.module.state_dict(), os.path.join(self.ckpt_dir, 'model_%02d.pt' % epoch_idx))
            torch.save(
                self.discriminator.module.state_dict(),
                os.path.join(self.ckpt_dir, 'discriminator_%02d.pt' % epoch_idx))
        else:
            torch.save(self.model.state_dict(), os.path.join(self.ckpt_dir, 'model_%02d.pt' % epoch_idx))
            torch.save(
                self.discriminator.state_dict(),
                os.path.join(self.ckpt_dir, 'discriminator_%02d.pt' % epoch_idx))

        self.logger.info("Training epoch %2d done." % epoch_idx)
        self.logger.info("-----------------------------------------------------")

    def _validate_one_epoch(self, epoch_idx):
        self.logger.info("*****************************************************")
        self.logger.info("Validating epoch %2d begin:" % epoch_idx)

        self._test_model_general(epoch_idx)

        illum_homo_moving_acc = self.illum_homo_acc_mov.average()
        view_homo_moving_acc = self.view_homo_acc_mov.average()

        illum_mma_moving_acc = self.illum_mma_mov.average()
        view_mma_moving_acc = self.view_mma_mov.average()

        illum_repeat_moving_acc = self.illum_repeat_mov.average()
        view_repeat_moving_acc = self.view_repeat_mov.average()

        current_size = self.view_mma_mov.current_size()

        # mAP, test_data, count = self._test_func(self.synthetic_test_dataset)
        # self.logger.info("[Epoch %2d] The mean Average Precision : %.4f of %d samples" % (epoch_idx, mAP, count))

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

    def _test_model_general(self, epoch_idx):

        self.model.eval()
        # 重置测评算子参数
        self.illum_repeat.reset()
        self.illum_homo_acc.reset()
        self.illum_mma.reset()
        self.view_repeat.reset()
        self.view_homo_acc.reset()
        self.view_mma.reset()
        self.point_statistics.reset()

        start_time = time.time()
        count = 0
        skip = 0

        self.logger.info("Test, detection_threshold=%.4f" % self.detection_threshold)

        for i, data in enumerate(self.test_dataset):
            first_image = data['first_image']
            second_image = data['second_image']
            gt_homography = data['gt_homography']
            image_type = data['image_type']

            image_pair = np.stack((first_image, second_image), axis=0)
            image_pair = torch.from_numpy(image_pair).to(torch.float).to(self.device).unsqueeze(dim=1)
            # debug released mode use
            # image_pair /= 255.
            image_pair = image_pair * 2. / 255. - 1.

            _, prob_pair, desp_pair, tmp = self.model(image_pair)
            prob_pair = f.pixel_shuffle(prob_pair, 8)
            prob_pair = spatial_nms(prob_pair, kernel_size=int(self.nms_threshold * 2 + 1))

            desp_pair = desp_pair.detach().cpu().numpy()
            first_desp = desp_pair[0]
            second_desp = desp_pair[1]
            prob_pair = prob_pair.detach().cpu().numpy()
            first_prob = prob_pair[0, 0]
            second_prob = prob_pair[1, 0]

            # 得到对应的预测点
            first_point, first_point_num = self._generate_predict_point(first_prob, top_k=self.top_k)  # [n,2]
            second_point, second_point_num = self._generate_predict_point(second_prob, top_k=self.top_k)  # [m,2]

            # 若未能检测出点，则跳过这一对测试样本
            if first_point_num == 0 or second_point_num == 0:
                skip += 1
                continue

            # 得到点对应的描述子
            select_first_desp = self._generate_predict_descriptor(first_point, first_desp)
            select_second_desp = self._generate_predict_descriptor(second_point, second_desp)

            # 得到匹配点
            matched_point = self.general_matcher(first_point, select_first_desp,
                                                 second_point, select_second_desp)
            if matched_point is None:
                skip += 1
                continue

            # 计算得到单应变换
            pred_homography, _ = cv.findHomography(matched_point[0][:, np.newaxis, ::-1],
                                                   matched_point[1][:, np.newaxis, ::-1], cv.RANSAC)

            if pred_homography is None:
                skip += 1
                continue

            # 对单样本进行测评
            if image_type == 'illumination':
                self.illum_repeat.update(first_point, second_point, gt_homography)
                self.illum_homo_acc.update(pred_homography, gt_homography)
                self.illum_mma.update(gt_homography, matched_point)

            elif image_type == 'viewpoint':
                self.view_repeat.update(first_point, second_point, gt_homography)
                self.view_homo_acc.update(pred_homography, gt_homography)
                self.view_mma.update(gt_homography, matched_point)

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

        # 统计最终的检测点数目的平均值和方差
        point_avg, point_std = self.point_statistics.average()

        self.logger.info("Homography Accuracy: illumination: %.4f, viewpoint: %.4f, total: %.4f" %
                         (illum_homo_acc, view_homo_acc, total_homo_acc))
        self.logger.info("Mean Matching Accuracy: illumination: %.4f, viewpoint: %.4f, total: %.4f " %
                         (illum_match_acc, view_match_acc, total_match_acc))
        self.logger.info("Repeatability: illumination: %.4f, viewpoint: %.4f, total: %.4f" %
                         (illum_repeat, view_repeat, total_repeat))
        self.logger.info("Detection point, average: %.4f, variance: %.4f" % (point_avg, point_std))

        self.summary_writer.add_scalar("illumination/Homography_Accuracy", illum_homo_acc, epoch_idx)
        self.summary_writer.add_scalar("illumination/Mean_Matching_Accuracy", illum_match_acc, epoch_idx)
        self.summary_writer.add_scalar("illumination/Repeatability", illum_repeat, epoch_idx)
        self.summary_writer.add_scalar("viewpoint/Homography_Accuracy", view_homo_acc, epoch_idx)
        self.summary_writer.add_scalar("viewpoint/Mean_Matching_Accuracy", view_match_acc, epoch_idx)
        self.summary_writer.add_scalar("viewpoint/Repeatability", view_repeat, epoch_idx)

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
            # 得到原始的经压缩的概率图，概率图每个通道64维，对应空间每个像素是否为关键点的概率
            _, prob, _, _ = self.model(image)
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

        return mAP, test_data, count

    @staticmethod
    def _compute_total_metric(illum_metric, view_metric):
        illum_acc, illum_sum, illum_num = illum_metric.average()
        view_acc, view_sum, view_num = view_metric.average()
        return illum_acc, view_acc, (illum_sum+view_sum)/(illum_num+view_num+1e-4)

    def _generate_predict_point(self, prob, scale=None, top_k=0):
        point_idx = np.where(prob > self.detection_threshold)
        prob = prob[point_idx]
        sorted_idx = np.argsort(prob)[::-1]
        if sorted_idx.shape[0] >= top_k:
            sorted_idx = sorted_idx[:top_k]

        point = np.stack(point_idx, axis=1)  # [n,2]
        top_k_point = []
        for idx in sorted_idx:
            top_k_point.append(point[idx])
        if len(top_k_point) <= 1:
            return None, 0
        point = np.stack(top_k_point, axis=0)
        point_num = point.shape[0]

        if scale is not None:
            point = point*scale
        return point, point_num

    def _generate_predict_descriptor(self, point, desp):
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
        self.mAP_calculator = mAPCalculator()


class MegPointAdaptionTrainer(MegPointTrainerTester):

    def __init__(self, params):
        super(MegPointAdaptionTrainer, self).__init__(params)

        # 初始化训练数据的读入接口
        self.train_dataset = COCOMegPointAdaptionDataset(params)
        self.epoch_length = len(self.train_dataset) / self.batch_size

        # debug use
        # self.total_batch = 320
        self.total_batch = 200
        self.logger.info("Only to process %d batches(max: %d), total: %d samples" %
                         (self.total_batch, self.epoch_length, int(self.total_batch*self.batch_size)))

        self.adaption_dataset = AdaptionDataset(
            int(self.total_batch*self.batch_size), params.aug_homography_params, params.photometric_params)
        self.raw_dataloader = DataLoader(self.train_dataset, self.batch_size, shuffle=False,
                                         num_workers=self.num_workers, drop_last=False)
        self.train_dataloader = DataLoader(self.adaption_dataset, self.batch_size, shuffle=True,
                                           num_workers=self.num_workers, drop_last=True)

        self.test_dataset = HPatchDataset(params)
        self.test_length = len(self.test_dataset)

        # 初始化各种验证算子
        self._initialize_test_calculator(params)

        # 初始化匹配算子
        self.general_matcher = Matcher('float')

        # 初始化单应变换采样算子
        self.sample_num = params.sample_num
        self.homography_sampler = HomographyAugmentation(**params.homography_params)

        # 初始化模型
        self.model = MegPointNet()
        self.magicpoint_ckpt_file = params.magicpoint_ckpt_file
        # 加载预训练的magicpoint模型
        self._load_model_params(params.magicpoint_ckpt_file)

        # 初始化伪真值生成器
        self.logger.info('Train top k: %d' % self.train_top_k)
        self.label_generator = LabelGenerator(params)

        # 初始化loss算子
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss(reduction='none')
        self.descriptor_loss = DescriptorTripletLoss(device=self.device)

        # 初始化loss相关的权重
        self.descriptor_weight = params.descriptor_weight

        # 若有多gpu设置则加载多gpu
        if self.multi_gpus:
            self.model = torch.nn.DataParallel(self.model)
            self.label_generator = torch.nn.DataParallel(self.label_generator)
        self.model = self.model.to(self.device)
        self.label_generator = self.label_generator.to(self.device)

        # 初始化优化器算子
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        # 初始化学习率调整算子
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=100, gamma=0.1)

    @staticmethod
    def _compute_masked_loss(unmasked_loss, mask):
        total_num = torch.sum(mask, dim=(1, 2))
        loss = torch.sum(mask*unmasked_loss, dim=(1, 2)) / total_num
        loss = torch.mean(loss)
        return loss

    def _adaption_data(self, epoch_idx, freq=6):
        if epoch_idx % freq != 0:
            return

        # 重置标注数据集以放入新标注的数据
        self.adaption_dataset.reset()

        # 将当前网络的参数读入到标注网络之中
        self.logger.info("Load current model parameters of epoch %d " % epoch_idx)
        if self.multi_gpus:
            model_dict = self.model.module.state_dict()
            generator_model_dict = self.label_generator.module.base_megpoint.state_dict()
            generator_model_dict.update(model_dict)
            self.label_generator.module.base_megpoint.load_state_dict(generator_model_dict)
        else:
            model_dict = self.model.state_dict()
            label_generator_model_dict = self.label_generator.base_megpoint.state_dict()
            label_generator_model_dict.update(model_dict)
            self.label_generator.base_megpoint.load_state_dict(label_generator_model_dict)

        if epoch_idx == 0:
            adaption_threshold = self.detection_threshold
            self.logger.info("Labeling, adaption_detection_threshold=%.4f" % adaption_threshold)
        else:
            adaption_threshold = self.detection_threshold*2.0
            self.logger.info("Labeling, adaption_detection_threshold=%.4f" % adaption_threshold)

        start_time = time.time()
        stime = time.time()
        count = 0
        self.logger.info("Relabeling current dataset")
        for i, data in enumerate(self.raw_dataloader):
            if i == self.total_batch:
                print("Debug use, only to adaption %d batches" % self.total_batch)
                break

            image = data["image"].to(self.device)

            # 采样构成标签需要的单应变换
            sampled_homo, sampled_inv_homo = self._sample_homography(self.batch_size)
            sampled_homo = sampled_homo.to(self.device)
            sampled_inv_homo = sampled_inv_homo.to(self.device)

            image, point, point_mask = self.label_generator(image, sampled_homo, sampled_inv_homo, adaption_threshold)
            self.adaption_dataset.append(image, point, point_mask)
            count += 1
            if i % self.log_freq == 0:
                self.logger.info("[Epoch:%2d][Labeling Step:%5d:%5d],"
                                 " one step cost %.4fs. "
                                 % (epoch_idx, i, self.epoch_length,
                                    (time.time() - stime) / self.log_freq,
                                    ))
                stime = time.time()

        self.logger.info("Relabeling Done. Totally %d batched sample. Takes %.3fs" % (count, (time.time()-start_time)))

    def _train_one_epoch(self, epoch_idx):
        self.logger.info("-----------------------------------------------------")
        self.logger.info("Start to train epoch %2d:" % epoch_idx)
        stime = time.time()

        self._adaption_data(epoch_idx)
        if len(self.adaption_dataset) == 0:
            assert False

        self.model.train()
        for i, data in enumerate(self.train_dataloader):

            image = data['image'].to(self.device)
            label = data['label'].to(self.device)
            mask = data['mask'].to(self.device)

            warped_image = data['warped_image'].to(self.device)
            warped_label = data['warped_label'].to(self.device)
            warped_mask = data['warped_mask'].to(self.device)

            matched_idx = data['matched_idx'].to(self.device)
            matched_valid = data['matched_valid'].to(self.device)
            not_search_mask = data['not_search_mask'].to(self.device)

            shape = image.shape

            image_pair = torch.cat((image, warped_image), dim=0)
            label_pair = torch.cat((label, warped_label), dim=0)
            mask_pair = torch.cat((mask, warped_mask), dim=0)
            logit_pair, _, desp_pair = self.model(image_pair)

            unmasked_point_loss = self.cross_entropy_loss(logit_pair, label_pair)
            point_loss = self._compute_masked_loss(unmasked_point_loss, mask_pair)

            desp_0, desp_1 = torch.split(desp_pair, shape[0], dim=0)

            desp_loss = self.descriptor_loss(
                desp_0, desp_1, matched_idx, matched_valid, not_search_mask)

            loss = point_loss + self.descriptor_weight*desp_loss

            if torch.isnan(loss):
                self.logger.error('loss is nan!')

            self.optimizer.zero_grad()
            loss.backward()

            self.optimizer.step()

            if i % self.log_freq == 0:

                point_loss_val = point_loss.item()
                desp_loss_val = desp_loss.item()
                loss_val = loss.item()

                self.summary_writer.add_histogram('descriptor', desp_pair)
                self.logger.info("[Epoch:%2d][Step:%5d:%5d]: loss = %.4f, point_loss = %.4f, desp_loss = %.4f"
                                 " one step cost %.4fs. "
                                 % (epoch_idx, i, self.epoch_length, loss_val,
                                    point_loss_val, desp_loss_val,
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

    def _validate_one_epoch(self, epoch_idx):
        self.logger.info("*****************************************************")
        self.logger.info("Validating epoch %2d begin:" % epoch_idx)

        self._test_model_general(epoch_idx)

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

    def _test_model_general(self, epoch_idx):

        self.model.eval()
        # 重置测评算子参数
        self.illum_repeat.reset()
        self.illum_homo_acc.reset()
        self.illum_mma.reset()
        self.view_repeat.reset()
        self.view_homo_acc.reset()
        self.view_mma.reset()
        self.point_statistics.reset()

        start_time = time.time()
        count = 0
        skip = 0

        self.logger.info("Test, detection_threshold=%.4f" % self.detection_threshold)

        for i, data in enumerate(self.test_dataset):
            first_image = data['first_image']
            second_image = data['second_image']
            gt_homography = data['gt_homography']
            image_type = data['image_type']

            image_pair = np.stack((first_image, second_image), axis=0)
            image_pair = torch.from_numpy(image_pair).to(torch.float).to(self.device).unsqueeze(dim=1)
            # debug released mode use
            # image_pair /= 255.
            image_pair = image_pair * 2. / 255. - 1.

            _, prob_pair, desp_pair = self.model(image_pair)
            prob_pair = f.pixel_shuffle(prob_pair, 8)
            prob_pair = spatial_nms(prob_pair, kernel_size=int(self.nms_threshold * 2 + 1))

            desp_pair = desp_pair.detach().cpu().numpy()
            first_desp = desp_pair[0]
            second_desp = desp_pair[1]
            prob_pair = prob_pair.detach().cpu().numpy()
            first_prob = prob_pair[0, 0]
            second_prob = prob_pair[1, 0]

            # 得到对应的预测点
            first_point, first_point_num = self._generate_predict_point(first_prob, top_k=self.top_k)  # [n,2]
            second_point, second_point_num = self._generate_predict_point(second_prob, top_k=self.top_k)  # [m,2]

            # 得到点对应的描述子
            select_first_desp = self._generate_predict_descriptor(first_point, first_desp)
            select_second_desp = self._generate_predict_descriptor(second_point, second_desp)

            # 若未能检测出点，则跳过这一对测试样本
            if first_point_num == 0 or second_point_num == 0:
                skip += 1
                continue

            # 得到匹配点
            matched_point = self.general_matcher(first_point, select_first_desp,
                                                 second_point, select_second_desp)

            # 计算得到单应变换
            pred_homography, _ = cv.findHomography(matched_point[0][:, np.newaxis, ::-1],
                                                   matched_point[1][:, np.newaxis, ::-1], cv.RANSAC)

            # 对单样本进行测评
            if image_type == 'illumination':
                self.illum_repeat.update(first_point, second_point, gt_homography)
                self.illum_homo_acc.update(pred_homography, gt_homography)
                self.illum_mma.update(gt_homography, matched_point)

            elif image_type == 'viewpoint':
                self.view_repeat.update(first_point, second_point, gt_homography)
                self.view_homo_acc.update(pred_homography, gt_homography)
                self.view_mma.update(gt_homography, matched_point)

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

        # 统计最终的检测点数目的平均值和方差
        point_avg, point_std = self.point_statistics.average()

        self.logger.info("Homography Accuracy: illumination: %.4f, viewpoint: %.4f, total: %.4f" %
                         (illum_homo_acc, view_homo_acc, total_homo_acc))
        self.logger.info("Mean Matching Accuracy: illumination: %.4f, viewpoint: %.4f, total: %.4f " %
                         (illum_match_acc, view_match_acc, total_match_acc))
        self.logger.info("Repeatability: illumination: %.4f, viewpoint: %.4f, total: %.4f" %
                         (illum_repeat, view_repeat, total_repeat))
        self.logger.info("Detection point, average: %.4f, variance: %.4f" % (point_avg, point_std))

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

    def _generate_predict_point(self, prob, scale=None, top_k=0):
        point_idx = np.where(prob > self.detection_threshold)
        prob = prob[point_idx]
        sorted_idx = np.argsort(prob)[::-1]
        if sorted_idx.shape[0] >= top_k:
            sorted_idx = sorted_idx[:top_k]

        point = np.stack(point_idx, axis=1)  # [n,2]
        top_k_point = []
        for idx in sorted_idx:
            top_k_point.append(point[idx])
        if len(top_k_point) <= 1:
            return None, 0
        point = np.stack(top_k_point, axis=0)
        point_num = point.shape[0]

        if scale is not None:
            point = point*scale
        return point, point_num

    def _generate_predict_descriptor(self, point, desp):
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

    def _sample_homography(self, batch_size):
        # 采样单应变换
        total_sample_num = batch_size*(self.sample_num-1)
        sampled_homo = []
        sampled_inv_homo = []
        for k in range(batch_size):
            sampled_homo.append(np.eye(3))
            sampled_inv_homo.append(np.eye(3))
        for j in range(total_sample_num):
            homo = self.homography_sampler.sample()
            sampled_homo.append(homo)
            sampled_inv_homo.append(np.linalg.inv(homo))
        # print("device in _sample_homography", device)
        sampled_homo = torch.from_numpy(np.stack(sampled_homo, axis=0)).to(torch.float)  # [bt*s,3,3]
        sampled_inv_homo = torch.from_numpy(np.stack(sampled_inv_homo, axis=0)).to(torch.float)  # [bt*s,3,3]

        return sampled_homo, sampled_inv_homo










