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

from nets.megpoint_net import resnet18_fast
from nets.megpoint_net import Extractor

from nets.superpoint_net import SuperPointNetBackbone3
from nets.superpoint_net import SuperPointExtractor
from nets.superpoint_net import SuperPointExtractor128
from nets.superpoint_net import SuperPointExtractor256
from nets.superpoint_net import SuperPointNet
from nets.superpoint_net import SuperPointDetector
from nets.superpoint_net import SuperPointNetDescriptor

from data_utils.coco_dataset import COCOMegPointHeatmapAllTrainDataset
from data_utils.megadepth_dataset import MegaDepthDatasetFromPreprocessed2
from data_utils.megadepth_coco_dataset import MegaDepthCOCODataset
from data_utils.megadepth_coco_dataset import MegaDepthCOCOSuperPointDataset
from data_utils.hpatch_dataset import HPatchDataset

from utils.evaluation_tools import RepeatabilityCalculator
from utils.evaluation_tools import MovingAverage
from utils.evaluation_tools import PointStatistics
from utils.evaluation_tools import HomoAccuracyCalculator
from utils.evaluation_tools import MeanMatchingAccuracy
from utils.utils import spatial_nms
from utils.utils import DescriptorGeneralTripletLoss
from utils.utils import Matcher
from utils.utils import PointHeatmapWeightedBCELoss


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

        self.run_mode = params.run_mode

        self.ckpt_file = params.ckpt_file
        self.adjust_lr = params.adjust_lr

        self.do_augmentation = params.do_augmentation
        self.weight_decay = params.weight_decay

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
            if i >= int(self.epoch_num * 2/ 3):
                self._validate_one_epoch(i)

            if self.adjust_lr:
                # adjust learning rate
                self.scheduler.step(i)

        end_time = time.time()
        self.logger.info("The whole training process takes %.3f h" % ((end_time - start_time)/3600))

    def _train_one_epoch(self, epoch_idx):
        raise NotImplementedError

    def _validate_one_epoch(self, epoch_idx):
        raise NotImplementedError

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


class MegPointHeatmapTrainer(MegPointTrainerTester):

    def __init__(self, params):
        super(MegPointHeatmapTrainer, self).__init__(params)

        self._initialize_dataset()
        self._initialize_model()
        self._initialize_optimizer()
        self._initialize_scheduler()
        self._initialize_train_func()
        self._initialize_inference_func()
        self._initialize_detection_threshold()
        self._initialize_loss()
        self._initialize_matcher()
        self._initialize_test_calculator(params)

    def _initialize_dataset(self):
        # 初始化数据集
        if self.params.model_type in ["SuperPointBackbone256", "SuperPointBackbone128"]:
            if self.params.dataset_type == "coco":
                self.logger.info("Initialize COCOMegPointHeatmapAllTrainDataset")
                self.train_dataset = COCOMegPointHeatmapAllTrainDataset(self.params)
            elif self.params.dataset_type == 'megadepth':
                self.logger.info("Initialize MegaDepthDatasetFromPreprocessed2")
                self.train_dataset = MegaDepthDatasetFromPreprocessed2(self.params.megadepth_dataset_dir,
                                                                       self.params.megadepth_label_dir)
            elif self.params.dataset_type == 'megacoco':
                self.logger.info("Initialize MegaDepthCOCODataset")
                self.train_dataset = MegaDepthCOCODataset(self.params.dataset_dir,
                                                          self.params.megadepth_dataset_dir,
                                                          self.params.megadepth_label_dir)
            else:
                assert False
        elif self.params.model_type == "SuperPoint":
            # only support megacoco training now
            assert self.params.dataset_type == 'megacoco'
            self.logger.info("Initialzie MegaDepthCOCOSuperPointDataset for superpoint")
            self.train_dataset = MegaDepthCOCOSuperPointDataset(self.params.dataset_dir,
                                                                self.params.megadepth_dataset_dir,
                                                                self.params.megadepth_label_dir)

        elif self.params.model_type == 'SuperPointDetector':
            assert self.params.dataset_type == 'megacoco'
            self.logger.info("Initialzie MegaDepthCOCOSuperPointDataset for superpoint detector + fbm")
            self.train_dataset = MegaDepthCOCOSuperPointDataset(self.params.dataset_dir,
                                                                self.params.megadepth_dataset_dir,
                                                                self.params.megadepth_label_dir)

        elif self.params.model_type == "SuperPointDescriptor":
            assert self.params.dataset_type == 'megacoco'
            self.logger.info("Initialize MegaDepthCOCODataset for superpoint descriptor + fsm")
            self.train_dataset = MegaDepthCOCODataset(self.params.dataset_dir,
                                                      self.params.megadepth_dataset_dir,
                                                      self.params.megadepth_label_dir)
        else:
            self.logger.error("Unrecognized model_type : %s" % self.params.model_type)
            assert False

        if self.run_mode == "test":
            pass
        else:
            self.train_dataloader = DataLoader(
                dataset=self.train_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                drop_last=True
            )
            self.epoch_length = len(self.train_dataset) // self.batch_size

        # 初始化测试集
        self.logger.info("Initialize HPatchDataset")
        self.test_dataset = HPatchDataset(self.params)
        self.test_length = len(self.test_dataset)

        # 初始化验证集
        self.val_dataset = None
        self.val_dataloader = None
        self.val_epoch_length = 1

    def _initialize_model(self):
        # 初始化模型
        if self.params.model_type == "SuperPoint":
            self.logger.info("Initialize network arch for SuperPoint: SuperPoint")
            model = SuperPointNet()
        elif self.params.model_type in ["SuperPointBackbone", "SuperPointBackbone256", "SuperPointBackbone128"]:
            self.logger.info("Initialize network arch for SuperPointBackbone3")
            model = SuperPointNetBackbone3()

        elif self.params.model_type == "SuperPointDetector":
            self.logger.info("Initialize network arch for SuperPointDetection+FBM : SuperPointDetection")
            model = SuperPointDetector()

        elif self.params.model_type == "SuperPointDescriptor":
            self.logger.info("Initialize network arch for SuperPointDescriptor+FSM: SuperPointDescriptor")
            model = SuperPointNetDescriptor()

        else:
            self.logger.error("Unrecognized model_type: %s" % self.params.model_type)
            assert False

        if self.multi_gpus:
            model = torch.nn.DataParallel(model)
        self.model = model.to(self.device)

        if self.params.model_type == "SuperPointBackbone128":
            self.logger.info("Initialize SuperPointBackbone extractor")
            extractor = SuperPointExtractor128()
        elif self.params.model_type == "SuperPointBackbone256":
            self.logger.info("Initialize SuperPointBackbone extractor256")
            extractor = SuperPointExtractor256()

        elif self.params.model_type == "SuperPointDetector":
            self.logger.info("Initialize SuperPointDetector extractor")
            extractor = SuperPointExtractor128()
        elif self.params.model_type in ["SuperPoint", "SuperPointDescriptor"]:
            self.logger.info("Initialize SuperPoint extractor to None(means not using)")
            extractor = None
        else:
            self.logger.error("Unrecognized model_type: %s" % self.params.model_type)
            assert False

        self.logger.info("Initialize cat func: _cat_c1c2c3c4")
        self.cat = self._cat_c1c2c3c4

        if extractor is not None:
            if self.multi_gpus:
                extractor = torch.nn.DataParallel(extractor)
            self.extractor = extractor.to(self.device)
        else:
            self.extractor = None

    def _initialize_loss(self):
        # 初始化loss算子
        if self.params.model_type in ["SuperPointBackbone128", "SuperPointBackbone256"]:
            # 初始化heatmap loss
            self.logger.info("Initialize the PointHeatmapWeightedBCELoss.")
            self.point_loss = PointHeatmapWeightedBCELoss()

            # 初始化描述子loss
            self.logger.info("Initialize the DescriptorGeneralTripletLoss.")
            self.descriptor_loss = DescriptorGeneralTripletLoss(self.device)
        elif self.params.model_type == "SuperPoint":
            # 初始化point loss
            self.logger.info("Initialize the CrossEntropyLoss for SuperPoint.")
            self.point_loss = torch.nn.CrossEntropyLoss(reduction="none")

            # 初始化描述子loss
            self.logger.info("Initialize the DescriptorTripletLoss for SuperPoint.")
            self.descriptor_loss = DescriptorGeneralTripletLoss(self.device)
        elif self.params.model_type == "SuperPointDetector":
            # 初始化point loss
            self.logger.info("Initialize the CrossEntropyLoss for SuperPoint Detection.")
            self.point_loss = torch.nn.CrossEntropyLoss(reduction="none")

            # 初始化描述子loss
            self.logger.info("Initialize the DescriptorGeneralTripletLoss for FBM.")
            self.descriptor_loss = DescriptorGeneralTripletLoss(self.device)
        elif self.params.model_type == "SuperPointDescriptor":
            # 初始化heatmap loss
            self.logger.info("Initialize the PointHeatmapWeightedBCELoss for FSM.")
            self.point_loss = PointHeatmapWeightedBCELoss()

            # 初始化描述子loss
            self.logger.info("Initialize the DescriptorTripletLoss for SuperPoint Descriptor.")
            self.descriptor_loss = DescriptorGeneralTripletLoss(self.device)
        else:
            self.logger.info("Unrecognized model_type : %s" % self.params.model_type)
            assert False

    def _initialize_matcher(self):
        # 初始化匹配算子
        self.logger.info("Initialize matcher of Nearest Neighbor.")
        self.general_matcher = Matcher('float')

    def _initialize_train_func(self):
        # 根据不同结构选择不同的训练函数
        # self.logger.info("Initialize training func mode of [with_gt] with baseline network.")
        # self._train_func = self._train_with_gt
        if self.params.model_type in ["SuperPointBackbone256", "SuperPointBackbone128"]:
            self.logger.info("Initialize training func mode of _train_with_gt_fast")
            self._train_func = self._train_with_gt_fast
        elif self.params.model_type == "SuperPoint":
            self.logger.info("Initialize training func mode of _train_for_superpoint")
            self._train_func = self._train_for_superpoint
        elif self.params.model_type == "SuperPointDetector":
            self.logger.info("Initialize training func mode of _train_for_superpoint_detection_fbm")
            self._train_func = self._train_for_superpoint_detection_fbm
        elif self.params.model_type == "SuperPointDescriptor":
            self.logger.info("Initialize training func mode of _train_for_superpoint_descriptor_fsm")
            self._train_func = self._train_for_superpoint_descriptor_fsm
        else:
            self.logger.info("Unrecognized model_type : %s" % self.params.model_type)
            assert False

    def _initialize_inference_func(self):
        if self.params.model_type in ["SuperPointBackbone256", "SuperPointBackbone128"]:
            self.logger.info("Initialize inference func mode of _inference_func_fast")
            self._inference_func = self._inference_func_fast
        elif self.params.model_type == "SuperPoint":
            self.logger.info("Initialize inference func mode of _inference_func_for_superpoint")
            self._inference_func = self._inference_func_for_superpoint
        elif self.params.model_type == "SuperPointDetector":
            self.logger.info("Initialize inference func mode of _inference_func_for_superpoint_detection_fbm")
            self._inference_func = self._inference_func_for_superpoint_detection_fbm
        elif self.params.model_type == 'SuperPointDescriptor':
            self.logger.info("Initialize inference func mode of _inference_func_for_superpoint_descriptor_fsm")
            self._inference_func = self._inference_func_for_superpoint_descriptor_fsm
        else:
            self.logger.info("Unrecognized model_type : %s" % self.params.model_type)
            assert False

    def _initialize_detection_threshold(self):
        if self.params.model_type in ["SuperPointBackbone256", "SuperPointBackbone128", "SuperPointDescriptor"]:
            self.logger.info("Initialize detection_threshold: %.3f" % 0.9)
            self.detection_threshold = 0.9
        elif self.params.model_type in ["SuperPoint", "SuperPointDetector", ]:
            self.logger.info("Initialize detection_threshold: %.3f" % 0.015)
            self.detection_threshold = 0.015
        else:
            self.logger.info("Unrecognized model_type : %s" % self.params.model_type)
            assert False

    def _initialize_optimizer(self):
        # 初始化网络训练优化器
        self.logger.info("Initialize Adam optimizer with weight_decay: %.5f." % self.params.weight_decay)
        self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        if self.extractor is not None:
            self.extractor_optimizer = torch.optim.Adam(params=self.extractor.parameters(), lr=self.lr)
        else:
            self.extractor_optimizer = None

    def _initialize_scheduler(self):
        # 初始化学习率调整算子
        milestones = [20, 30]
        self.logger.info("Initialize lr_scheduler of MultiStepLR: (%d, %d)" % (milestones[0], milestones[1]))
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=milestones, gamma=0.1)

    def _train_one_epoch(self, epoch_idx):
        self.model.train()

        self.logger.info("-----------------------------------------------------")
        self.logger.info("Training epoch %2d begin:" % epoch_idx)

        self._train_func(epoch_idx)

        self.logger.info("Training epoch %2d done." % epoch_idx)
        self.logger.info("-----------------------------------------------------")

    def _train_with_gt_fast(self, epoch_idx):
        self.model.train()
        self.extractor.train()
        stime = time.time()
        for i, data in enumerate(self.train_dataloader):

            # 读取相关数据
            image = data["image"].to(self.device)
            heatmap_gt = data['heatmap'].to(self.device)
            point_mask = data['point_mask'].to(self.device)
            desp_point = data["desp_point"].to(self.device)

            warped_image = data["warped_image"].to(self.device)
            warped_heatmap_gt = data['warped_heatmap'].to(self.device)
            warped_point_mask = data['warped_point_mask'].to(self.device)
            warped_desp_point = data["warped_desp_point"].to(self.device)

            valid_mask = data["valid_mask"].to(self.device)
            not_search_mask = data["not_search_mask"].to(self.device)

            image_pair = torch.cat((image, warped_image), dim=0)

            # 模型预测
            heatmap_pred_pair, c1_pair, c2_pair, c3_pair, c4_pair = self.model(image_pair)

            # 计算描述子loss
            desp_point_pair = torch.cat((desp_point, warped_desp_point), dim=0)
            c1_feature_pair = f.grid_sample(c1_pair, desp_point_pair, mode="bilinear", padding_mode="border")
            c2_feature_pair = f.grid_sample(c2_pair, desp_point_pair, mode="bilinear", padding_mode="border")
            c3_feature_pair = f.grid_sample(c3_pair, desp_point_pair, mode="bilinear", padding_mode="border")
            c4_feature_pair = f.grid_sample(c4_pair, desp_point_pair, mode="bilinear", padding_mode="border")

            feature_pair = self.cat(c1_feature_pair, c2_feature_pair, c3_feature_pair, c4_feature_pair, dim=1)
            feature_pair = feature_pair[:, :, :, 0].transpose(1, 2)
            desp_pair = self.extractor(feature_pair)
            desp_0, desp_1 = torch.chunk(desp_pair, 2, dim=0)

            desp_loss = self.descriptor_loss(desp_0, desp_1, valid_mask, not_search_mask)

            # 计算关键点loss
            heatmap_gt_pair = torch.cat((heatmap_gt, warped_heatmap_gt), dim=0)
            point_mask_pair = torch.cat((point_mask, warped_point_mask), dim=0)
            point_loss = self.point_loss(heatmap_pred_pair[:, 0, :, :], heatmap_gt_pair, point_mask_pair)

            loss = desp_loss + 0.1 * point_loss

            if torch.isnan(loss):
                self.logger.error('loss is nan!')

            self.optimizer.zero_grad()
            self.extractor_optimizer.zero_grad()

            loss.backward()

            self.optimizer.step()
            self.extractor_optimizer.step()

            # debug use
            # if i == 200:
            #     break

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
            torch.save(self.extractor.module.state_dict(), os.path.join(self.ckpt_dir, 'extractor_%02d.pt' % epoch_idx))
        else:
            torch.save(self.model.state_dict(), os.path.join(self.ckpt_dir, 'model_%02d.pt' % epoch_idx))
            torch.save(self.extractor.state_dict(), os.path.join(self.ckpt_dir, 'extractor_%02d.pt' % epoch_idx))

    def _train_for_superpoint(self, epoch_idx):
        self.model.train()

        stime = time.time()
        for i, data in enumerate(self.train_dataloader):

            image = data['image'].to(self.device)
            label = data['label'].to(self.device)
            mask = data['mask'].to(self.device)

            warped_image = data['warped_image'].to(self.device)
            warped_label = data['warped_label'].to(self.device)
            warped_mask = data['warped_mask'].to(self.device)

            desp_point = data["desp_point"].to(self.device)
            warped_desp_point = data["warped_desp_point"].to(self.device)
            valid_mask = data["valid_mask"].to(self.device)
            not_search_mask = data["not_search_mask"].to(self.device)

            shape = image.shape

            image_pair = torch.cat((image, warped_image), dim=0)
            label_pair = torch.cat((label, warped_label), dim=0)
            mask_pair = torch.cat((mask, warped_mask), dim=0)

            logit_pair, desp_pair, _ = self.model(image_pair)

            unmasked_point_loss = self.point_loss(logit_pair, label_pair)
            point_loss = self._compute_masked_loss(unmasked_point_loss, mask_pair)

            # compute descriptor loss
            desp_point_pair = torch.cat((desp_point, warped_desp_point), dim=0)
            desp_pair = f.grid_sample(desp_pair, desp_point_pair, mode="bilinear", padding_mode="border")
            desp_pair = desp_pair[:, :, :, 0].transpose(1, 2)
            desp_pair = desp_pair / torch.norm(desp_pair, dim=2, keepdim=True)
            desp_0, desp_1 = torch.chunk(desp_pair, 2, dim=0)

            desp_loss = self.descriptor_loss(desp_0, desp_1, valid_mask, not_search_mask)

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

    def _train_for_superpoint_detection_fbm(self, epoch_idx):
        self.model.train()
        self.extractor.train()

        stime = time.time()
        for i, data in enumerate(self.train_dataloader):

            image = data['image'].to(self.device)
            label = data['label'].to(self.device)
            mask = data['mask'].to(self.device)

            warped_image = data['warped_image'].to(self.device)
            warped_label = data['warped_label'].to(self.device)
            warped_mask = data['warped_mask'].to(self.device)

            desp_point = data["desp_point"].to(self.device)
            warped_desp_point = data["warped_desp_point"].to(self.device)
            valid_mask = data["valid_mask"].to(self.device)
            not_search_mask = data["not_search_mask"].to(self.device)

            image_pair = torch.cat((image, warped_image), dim=0)
            label_pair = torch.cat((label, warped_label), dim=0)
            mask_pair = torch.cat((mask, warped_mask), dim=0)
            logit_pair, prob_pair, c1_pair, c2_pair, c3_pair, c4_pair = self.model(image_pair)

            # 计算描述子loss
            desp_point_pair = torch.cat((desp_point, warped_desp_point), dim=0)
            c1_feature_pair = f.grid_sample(c1_pair, desp_point_pair, mode="bilinear", padding_mode="border")
            c2_feature_pair = f.grid_sample(c2_pair, desp_point_pair, mode="bilinear", padding_mode="border")
            c3_feature_pair = f.grid_sample(c3_pair, desp_point_pair, mode="bilinear", padding_mode="border")
            c4_feature_pair = f.grid_sample(c4_pair, desp_point_pair, mode="bilinear", padding_mode="border")

            feature_pair = self.cat(c1_feature_pair, c2_feature_pair, c3_feature_pair, c4_feature_pair, dim=1)
            feature_pair = feature_pair[:, :, :, 0].transpose(1, 2)
            desp_pair = self.extractor(feature_pair)
            desp_0, desp_1 = torch.chunk(desp_pair, 2, dim=0)

            desp_loss = self.descriptor_loss(desp_0, desp_1, valid_mask, not_search_mask)

            # 计算关键点loss
            unmasked_point_loss = self.point_loss(logit_pair, label_pair)
            point_loss = self._compute_masked_loss(unmasked_point_loss, mask_pair)

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
            torch.save(self.extractor.module.state_dict(), os.path.join(self.ckpt_dir, 'extractor_%02d.pt' % epoch_idx))
        else:
            torch.save(self.model.state_dict(), os.path.join(self.ckpt_dir, 'model_%02d.pt' % epoch_idx))
            torch.save(self.extractor.state_dict(), os.path.join(self.ckpt_dir, 'extractor_%02d.pt' % epoch_idx))

    def _train_for_superpoint_descriptor_fsm(self, epoch_idx):
        self.model.train()

        stime = time.time()
        for i, data in enumerate(self.train_dataloader):

            image = data["image"].to(self.device)
            heatmap_gt = data['heatmap'].to(self.device)
            point_mask = data['point_mask'].to(self.device)

            warped_image = data["warped_image"].to(self.device)
            warped_heatmap_gt = data['warped_heatmap'].to(self.device)
            warped_point_mask = data['warped_point_mask'].to(self.device)

            desp_point = data["desp_point"].to(self.device)
            warped_desp_point = data["warped_desp_point"].to(self.device)
            valid_mask = data["valid_mask"].to(self.device)
            not_search_mask = data["not_search_mask"].to(self.device)

            shape = image.shape

            image_pair = torch.cat((image, warped_image), dim=0)
            heatmap_pred_pair, desp_pair = self.model(image_pair)

            # 计算关键点loss
            heatmap_gt_pair = torch.cat((heatmap_gt, warped_heatmap_gt), dim=0)
            point_mask_pair = torch.cat((point_mask, warped_point_mask), dim=0)
            point_loss = self.point_loss(heatmap_pred_pair[:, 0, :, :], heatmap_gt_pair, point_mask_pair)

            # compute descriptor loss
            desp_point_pair = torch.cat((desp_point, warped_desp_point), dim=0)
            desp_pair = f.grid_sample(desp_pair, desp_point_pair, mode="bilinear", padding_mode="border")
            desp_pair = desp_pair[:, :, :, 0].transpose(1, 2)
            desp_pair = desp_pair / torch.norm(desp_pair, dim=2, keepdim=True)
            desp_0, desp_1 = torch.chunk(desp_pair, 2, dim=0)

            desp_loss = self.descriptor_loss(desp_0, desp_1, valid_mask, not_search_mask)

            loss = 0.1 * point_loss + desp_loss

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

    def test(self, ckpt_file, extrator_ckpt_file=None):
        self.model = self._load_model_params(ckpt_file, self.model)
        if extrator_ckpt_file is not None and self.extractor is not None:
            self.extractor = self._load_model_params(extrator_ckpt_file, self.extractor)

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
            image_pair = image_pair * 2. / 255. - 1.

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
            self.point_statistics.update((first_point_num + second_point_num) / 2.)

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

    def _inference_func_fast(self, image_pair):
        """
        image_pair: [2,1,h,w]
        """
        self.model.eval()
        self.extractor.eval()
        _, _, height, width = image_pair.shape
        heatmap_pair, c1_pair, c2_pair, c3_pair, c4_pair = self.model(image_pair)

        c1_0, c1_1 = torch.chunk(c1_pair, 2, dim=0)
        c2_0, c2_1 = torch.chunk(c2_pair, 2, dim=0)
        c3_0, c3_1 = torch.chunk(c3_pair, 2, dim=0)
        c4_0, c4_1 = torch.chunk(c4_pair, 2, dim=0)

        heatmap_pair = torch.sigmoid(heatmap_pair)
        prob_pair = spatial_nms(heatmap_pair, kernel_size=int(self.nms_threshold * 2 + 1))

        prob_pair = prob_pair.detach().cpu().numpy()
        first_prob = prob_pair[0, 0]
        second_prob = prob_pair[1, 0]

        # 得到对应的预测点
        first_point, first_point_num = self._generate_predict_point(first_prob, top_k=self.top_k)  # [n,2]
        second_point, second_point_num = self._generate_predict_point(second_prob, top_k=self.top_k)  # [m,2]

        if first_point_num <= 4 or second_point_num <= 4:
            print("skip this pair because there's little point!")
            return None

        # 得到点对应的描述子
        select_first_desp = self._generate_combined_descriptor_fast(first_point, c1_0, c2_0, c3_0, c4_0, height, width)
        select_second_desp = self._generate_combined_descriptor_fast(second_point, c1_1, c2_1, c3_1, c4_1, height, width)

        return first_point, first_point_num, second_point, second_point_num, select_first_desp, select_second_desp

    def _inference_func_for_superpoint(self, image_pair):
        """
        image_pair: [2,1,h,w]
        """
        self.model.eval()
        _, _, height, width = image_pair.shape
        _, desp_pair, prob_pair = self.model(image_pair)
        prob_pair = f.pixel_shuffle(prob_pair, 8)
        prob_pair = spatial_nms(prob_pair, kernel_size=int(self.nms_threshold * 2 + 1))

        # 得到对应的预测点
        prob_pair = prob_pair.detach().cpu().numpy()
        first_prob = prob_pair[0, 0]
        second_prob = prob_pair[1, 0]

        first_point, first_point_num = self._generate_predict_point(first_prob, top_k=self.top_k)  # [n,2]
        second_point, second_point_num = self._generate_predict_point(second_prob, top_k=self.top_k)  # [m,2]

        if first_point_num <= 4 or second_point_num <= 4:
            print("skip this pair because there's little point!")
            return None

        # 得到点对应的描述子
        first_desp, second_desp = torch.chunk(desp_pair, 2, dim=0)

        select_first_desp = self._generate_descriptor_for_superpoint_desp_head(first_point, first_desp, height, width)
        select_second_desp = self._generate_descriptor_for_superpoint_desp_head(second_point, second_desp, height, width)

        return first_point, first_point_num, second_point, second_point_num, select_first_desp, select_second_desp

    def _inference_func_for_superpoint_descriptor_fsm(self, image_pair):
        """
        image_pair: [2,1,h,w]
        """
        self.model.eval()
        _, _, height, width = image_pair.shape
        heatmap_pair, desp_pair = self.model(image_pair)

        # 得到对应的关键点
        heatmap_pair = torch.sigmoid(heatmap_pair)
        prob_pair = spatial_nms(heatmap_pair, kernel_size=int(self.nms_threshold * 2 + 1))

        prob_pair = prob_pair.detach().cpu().numpy()
        first_prob = prob_pair[0, 0]
        second_prob = prob_pair[1, 0]

        first_point, first_point_num = self._generate_predict_point(first_prob, top_k=self.top_k)  # [n,2]
        second_point, second_point_num = self._generate_predict_point(second_prob, top_k=self.top_k)  # [m,2]

        if first_point_num <= 4 or second_point_num <= 4:
            print("skip this pair because there's little point!")
            return None

        # 得到点对应的描述子
        first_desp, second_desp = torch.chunk(desp_pair, 2, dim=0)

        select_first_desp = self._generate_descriptor_for_superpoint_desp_head(first_point, first_desp, height, width)
        select_second_desp = self._generate_descriptor_for_superpoint_desp_head(second_point, second_desp, height, width)

        return first_point, first_point_num, second_point, second_point_num, select_first_desp, select_second_desp

    def _inference_func_for_superpoint_detection_fbm(self, image_pair):
        """
        image_pair: [2,1,h,w]
        """
        self.model.eval()
        self.extractor.eval()
        _, _, height, width = image_pair.shape
        _, prob_pair, c1_pair, c2_pair, c3_pair, c4_pair = self.model(image_pair)

        c1_0, c1_1 = torch.chunk(c1_pair, 2, dim=0)
        c2_0, c2_1 = torch.chunk(c2_pair, 2, dim=0)
        c3_0, c3_1 = torch.chunk(c3_pair, 2, dim=0)
        c4_0, c4_1 = torch.chunk(c4_pair, 2, dim=0)

        # 得到对应的预测点
        prob_pair = f.pixel_shuffle(prob_pair, 8)
        prob_pair = spatial_nms(prob_pair, kernel_size=int(self.nms_threshold * 2 + 1))

        prob_pair = prob_pair.detach().cpu().numpy()
        first_prob = prob_pair[0, 0]
        second_prob = prob_pair[1, 0]

        first_point, first_point_num = self._generate_predict_point(first_prob, top_k=self.top_k)  # [n,2]
        second_point, second_point_num = self._generate_predict_point(second_prob, top_k=self.top_k)  # [m,2]

        if first_point_num <= 4 or second_point_num <= 4:
            print("skip this pair because there's little point!")
            return None

        # 得到点对应的描述子
        select_first_desp = self._generate_combined_descriptor_fast(first_point, c1_0, c2_0, c3_0, c4_0, height, width)
        select_second_desp = self._generate_combined_descriptor_fast(second_point, c1_1, c2_1, c3_1, c4_1, height, width)

        return first_point, first_point_num, second_point, second_point_num, select_first_desp, select_second_desp

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

    def _generate_combined_descriptor_fast(self, point, c1, c2, c3, c4, height, width):
        """
        用多层级的组合特征构造描述子
        Args:
            point: [n,2] 顺序是y,x
            c1,c2,c3,c4: 分别对应resnet4个block输出的特征,batchsize都是1
        Returns:
            desp: [n,dim]
        """
        point = torch.from_numpy(point[:, ::-1].copy()).to(torch.float).to(self.device)
        # 归一化采样坐标到[-1,1]
        point = point * 2. / torch.tensor((width-1, height-1), dtype=torch.float, device=self.device) - 1
        point = point.unsqueeze(dim=0).unsqueeze(dim=2)  # [1,n,1,2]

        c1_feature = f.grid_sample(c1, point, mode="bilinear")[:, :, :, 0].transpose(1, 2)
        c2_feature = f.grid_sample(c2, point, mode="bilinear")[:, :, :, 0].transpose(1, 2)
        c3_feature = f.grid_sample(c3, point, mode="bilinear")[:, :, :, 0].transpose(1, 2)
        c4_feature = f.grid_sample(c4, point, mode="bilinear")[:, :, :, 0].transpose(1, 2)

        feature = self.cat(c1_feature, c2_feature, c3_feature, c4_feature, dim=2)
        desp = self.extractor(feature)[0]  # [n,128]

        desp = desp.detach().cpu().numpy()

        return desp

    def _generate_descriptor_for_superpoint_desp_head(self, point, desp, height, width):
        """
        构建superpoint描述子端的描述子
        """
        point = torch.from_numpy(point[:, ::-1].copy()).to(torch.float).to(self.device)
        # 归一化采样坐标到[-1,1]
        point = point * 2. / torch.tensor((width-1, height-1), dtype=torch.float, device=self.device) - 1
        point = point.unsqueeze(dim=0).unsqueeze(dim=2)  # [1,n,1,2]

        desp = f.grid_sample(desp, point, mode="bilinear")[0, :, :, 0].transpose(0, 1)

        desp = desp.detach().cpu().numpy()

        return desp

    def _initialize_test_calculator(self, params):
        # 初始化验证算子
        self.logger.info('Initialize the homography accuracy calculator, correct_epsilon: %d' % self.correct_epsilon)
        self.logger.info('Initialize the repeatability calculator, detection_threshold: %.4f, correct_epsilon: %d'
                         % (self.detection_threshold, self.correct_epsilon))
        self.logger.info('Top k: %d' % self.top_k)

        self.illum_repeat = RepeatabilityCalculator(params.correct_epsilon, self.params.hpatch_height, self.params.hpatch_width)
        self.illum_repeat_mov = MovingAverage(max_size=15)

        self.view_repeat = RepeatabilityCalculator(params.correct_epsilon, self.params.hpatch_height, self.params.hpatch_width)
        self.view_repeat_mov = MovingAverage(max_size=15)

        self.illum_homo_acc = HomoAccuracyCalculator(params.correct_epsilon,
                                                     params.hpatch_height, params.hpatch_width)
        self.illum_homo_acc_mov = MovingAverage(max_size=15)

        self.view_homo_acc = HomoAccuracyCalculator(params.correct_epsilon,
                                                    params.hpatch_height, params.hpatch_width)
        self.view_homo_acc_mov = MovingAverage(max_size=15)

        self.illum_mma = MeanMatchingAccuracy(params.correct_epsilon)
        self.illum_mma_mov = MovingAverage(max_size=15)

        self.view_mma = MeanMatchingAccuracy(params.correct_epsilon)
        self.view_mma_mov = MovingAverage(max_size=15)

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



