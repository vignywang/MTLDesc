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
from nets.megpoint_net import MegPointShuffleHeatmapOld
from nets.megpoint_net import resnet18
from nets.megpoint_net import resnet34

from nets.megpoint_net import resnet18_c1
from nets.megpoint_net import resnet18_c2
from nets.megpoint_net import resnet18_c3
from nets.megpoint_net import resnet18_c4
from nets.megpoint_net import resnet18_c1c2
from nets.megpoint_net import resnet18_c1c3
from nets.megpoint_net import resnet18_c1c4
from nets.megpoint_net import resnet18_c2c3
from nets.megpoint_net import resnet18_c2c4
from nets.megpoint_net import resnet18_c3c4
from nets.megpoint_net import resnet18_c1c2c3
from nets.megpoint_net import resnet18_c1c2c4
from nets.megpoint_net import resnet18_c1c3c4
from nets.megpoint_net import resnet18_c2c3c4
from nets.megpoint_net import resnet18_c1c2c3c4

from nets.megpoint_net import resnet18_s1s2s3
from nets.megpoint_net import resnet18_s1s2s4
from nets.megpoint_net import resnet18_s1s3s4
from nets.megpoint_net import resnet18_s3s4
from nets.megpoint_net import resnet18_s1s2s3s4

from nets.segment_net import deeplabv3_resnet50
from nets.superpoint_net import SuperPointNetFloat

from data_utils.coco_dataset import COCOMegPointHeatmapTrainDataset
from data_utils.coco_dataset import COCOMegPointHeatmapOnlyDataset
from data_utils.coco_dataset import COCOMegPointHeatmapOnlyIndexDataset
from data_utils.coco_dataset import COCOMegPointDescriptorOnlyDataset
from data_utils.coco_dataset import COCOMegPointHeatmapAllTrainDataset
from data_utils.megadepth_dataset import MegaDepthDatasetFromPreprocessed
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
from utils.utils import spatial_nms
from utils.utils import DescriptorTripletLoss
from utils.utils import DescriptorGeneralTripletLoss
from utils.utils import DescriptorRankedListLoss
from utils.utils import DescriptorValidator
from utils.utils import Matcher
from utils.utils import NearestNeighborThresholdMatcher
from utils.utils import NearestNeighborRatioMatcher
from utils.utils import PointHeatmapWeightedBCELoss
from utils.utils import PointHeatmapSpatialFocalWeightedBCELoss
from utils.utils import HeatmapAlignLoss
from utils.utils import HeatmapWeightedAlignLoss
from utils.utils import HomographyReprojectionLoss
from utils.utils import ReprojectionLoss


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
        self.ckpt_file = params.ckpt_file
        self.adjust_lr = params.adjust_lr
        self.align_weight = params.align_weight
        self.align_type = params.align_type
        self.point_type = params.point_type
        self.fn_scale = params.fn_scale
        self.homo_weight = params.homo_weight
        self.repro_weight = params.repro_weight
        self.half_region_size = params.half_region_size
        self.dataset_type = params.dataset_type

        self.desp_loss_type = params.desp_loss_type

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
        self._initialize_loss()
        self._initialize_matcher()
        self._initialize_test_calculator(params)

        self._initialize_general_coords(self.half_region_size)

    def _initialize_dataset(self):
        # 初始化数据集
        if self.train_mode == "with_gt":
            # self.logger.info("Initialize COCOMegPointHeatmapTrainDataset")
            # self.train_dataset = COCOMegPointHeatmapTrainDataset(self.params)
            self.logger.info("Initialize COCOMegPointHeatmapAllTrainDataset")
            self.train_dataset = COCOMegPointHeatmapAllTrainDataset(self.params)
        elif self.train_mode == "only_detector":
            self.logger.info("Initialize COCOMegPointHeatmapOnlyDataset")
            self.train_dataset = COCOMegPointHeatmapOnlyDataset(self.params)
        elif self.train_mode == "only_detector_index":
            self.logger.info("Initialize COCOMegPointHeatmapOnlyIndexDataset")
            self.train_dataset = COCOMegPointHeatmapOnlyIndexDataset(self.params)
        elif self.train_mode == "only_descriptor":
            # if self.params.height != 224 or self.params.width != 224:
            #     self.logger.error("Training only descriptor only support 224x224 image input size!")
            #     assert False
            # self.logger.info("Initialize COCOMegPointDescriptorOnlyDataset")
            # self.train_dataset = COCOMegPointDescriptorOnlyDataset(self.params)
            self.logger.info("Initialize MegaDepthDataset")
            self.train_dataset = MegaDepthDatasetFromPreprocessed(dataset_dir=self.params.mega_dataset_dir,
                                                                  do_augmentation=False)
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

        # 初始化验证集
        if self.train_mode == "only_descriptor":
            self.logger.info("Initialize MegaDepth validation dataset")
            self.val_dataset = MegaDepthDatasetFromPreprocessed(dataset_dir=self.params.mega_val_dataset_dir,
                                                                do_augmentation=False)
            self.val_dataloader = DataLoader(
                dataset=self.val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                drop_last=True,
            )
            self.val_epoch_length = len(self.val_dataset) // self.batch_size
        else:
            self.val_dataset = None
            self.val_dataloader = None
            self.val_epoch_length = 1

    def _initialize_model(self):
        # 初始化模型
        if self.network_arch == "resnet18":
            self.logger.info("Initialize network arch : restnet18")
            model = resnet18()
            # model = MegPointShuffleHeatmapOld()
        elif self.network_arch == "resnet34":
            self.logger.info("Initialize network arch : resnet34")
            model = resnet34()
        elif self.network_arch == "deeplabv3_resnet50":
            self.logger.info("Initialize network arch : deeplabv3_resnet50")
            model = deeplabv3_resnet50()

        elif self.network_arch == "resnet18_c1":
            self.logger.info("Initialize network arch : restnet18_c1")
            model = resnet18_c1()
        elif self.network_arch == "resnet18_c2":
            self.logger.info("Initialize network arch : restnet18_c2")
            model = resnet18_c2()
        elif self.network_arch == "resnet18_c3":
            self.logger.info("Initialize network arch : restnet18_c3")
            model = resnet18_c3()
        elif self.network_arch == "resnet18_c4":
            self.logger.info("Initialize network arch : restnet18_c4")
            model = resnet18_c4()
        elif self.network_arch == "resnet18_c1c2":
            self.logger.info("Initialize network arch : restnet18_c1c2")
            model = resnet18_c1c2()
        elif self.network_arch == "resnet18_c1c3":
            self.logger.info("Initialize network arch : restnet18_c1c3")
            model = resnet18_c1c3()
        elif self.network_arch == "resnet18_c1c4":
            self.logger.info("Initialize network arch : restnet18_c1c4")
            model = resnet18_c1c4()
        elif self.network_arch == "resnet18_c2c3":
            self.logger.info("Initialize network arch : restnet18_c2c3")
            model = resnet18_c2c3()
        elif self.network_arch == "resnet18_c2c4":
            self.logger.info("Initialize network arch : restnet18_c2c4")
            model = resnet18_c2c4()
        elif self.network_arch == "resnet18_c3c4":
            self.logger.info("Initialize network arch : restnet18_c3c4")
            model = resnet18_c3c4()
        elif self.network_arch == "resnet18_c1c2c3":
            self.logger.info("Initialize network arch : restnet18_c1c2c3")
            model = resnet18_c1c2c3()
        elif self.network_arch == "resnet18_c1c2c4":
            self.logger.info("Initialize network arch : restnet18_c1c2c4")
            model = resnet18_c1c2c4()
        elif self.network_arch == "resnet18_c1c3c4":
            self.logger.info("Initialize network arch : restnet18_c1c3c4")
            model = resnet18_c1c3c4()
        elif self.network_arch == "resnet18_c2c3c4":
            self.logger.info("Initialize network arch : restnet18_c2c3c4")
            model = resnet18_c2c3c4()
        elif self.network_arch == "resnet18_c1c2c3c4":
            self.logger.info("Initialize network arch : restnet18_c1c2c3c4")
            model = resnet18_c1c2c3c4()

        elif self.network_arch == "resnet18_s1s2s3":
            self.logger.info("Initialize network arch : restnet18_s1s2s3")
            model = resnet18_s1s2s3()
        elif self.network_arch == "resnet18_s1s2s4":
            self.logger.info("Initialize network arch : restnet18_s1s2s4")
            model = resnet18_s1s2s4()
        elif self.network_arch == "resnet18_s1s3s4":
            self.logger.info("Initialize network arch : restnet18_s1s3s4")
            model = resnet18_s1s3s4()
        elif self.network_arch == "resnet18_s3s4":
            self.logger.info("Initialize network arch : restnet18_s3s4")
            model = resnet18_s3s4()
        elif self.network_arch == "resnet18_s1s2s3s4":
            self.logger.info("Initialize network arch : restnet18_s1s2s3s4")
            model = resnet18_s1s2s3s4()

        else:
            self.logger.error("unrecognized network_arch:%s" % self.network_arch)
            assert False
        if self.multi_gpus:
            model = torch.nn.DataParallel(model)
        self.model = model.to(self.device)

        # extractor = DescriptorExtractor()
        # if self.multi_gpus:
        #     extractor = torch.nn.DataParallel(extractor)
        # self.extractor = extractor.to(self.device)

        self.descriptor = None
        if self.train_mode in ["only_detector", "only_detector_index"]:
            self.logger.info("Initialize pretrained descriptor generator")
            descriptor = MegPointShuffleHeatmapOld()
            if self.multi_gpus:
                descriptor = torch.nn.DataParallel(descriptor)
            descriptor = descriptor.to(self.device)
            self.descriptor = self._load_model_params(self.ckpt_file, descriptor)

        self.detector = None
        if self.train_mode in ["only_descriptor"]:
            # if self.network_arch not in  ["resnet18", "resnet34", "deeplabv3_resnet50"]:
            #     self.logger.error("Only support resnet18/resnet34 as the descriptor model to be trained.")
            #     assert False
            self.logger.info("Initialize pretrained detector")
            detector = MegPointShuffleHeatmapOld()
            if self.multi_gpus:
                detector = torch.nn.DataParallel(detector)
            detector = detector.to(self.device)
            self.detector = self._load_model_params(self.ckpt_file, detector)

    def _initialize_loss(self):
        # 初始化loss算子
        # 初始化heatmap loss
        if self.point_type == "general":
            self.logger.info("Initialize the PointHeatmapWeightedBCELoss.")
            self.point_loss = PointHeatmapWeightedBCELoss()
        elif self.point_type == "spatial":
            self.logger.info("Initialize the PointHeatmapSpatialFocalWeightedBCELoss, fn_scale:%.2f." % self.fn_scale)
            self.point_loss = PointHeatmapSpatialFocalWeightedBCELoss(device=self.device, fn_scale=self.fn_scale)
        else:
            self.logger.error("Unrecogized point_type: %s" % self.point_type)
            assert False

        # 初始化heatmap对齐loss
        if self.align_type == "general":
            self.logger.info("Initialize the unweighted align loss.")
            self.align_loss = HeatmapAlignLoss()
        elif self.align_type == "weighted":
            self.logger.info("Initialize the weighted align loss.")
            self.align_loss = HeatmapWeightedAlignLoss()
        else:
            self.logger.error("Unrecognized align_type: %s" % self.align_type)
            assert False

        # 初始化描述子loss
        if self.train_mode == "only_descriptor":
            if self.desp_loss_type == "general":
                self.logger.info("Initialize the DescriptorGeneralTripletLoss.")
                self.descriptor_loss = DescriptorGeneralTripletLoss(self.device)
            elif self.desp_loss_type == "rank":
                self.logger.info("Initialize the DescriptorRnakedListLoss")
                self.descriptor_loss = DescriptorRankedListLoss(margin=1.2, alpha=1.2, t=10, device=self.device)
            else:
                self.logger.error("Unrecognized desp_loss_type: %s" % self.desp_loss_type)
                assert False
        else:
            # self.logger.info("Initialize the DescriptorTripletLoss.")
            # self.descriptor_loss = DescriptorTripletLoss(self.device)
            self.logger.info("Initialize the DescriptorGeneralTripletLoss.")
            self.descriptor_loss = DescriptorGeneralTripletLoss(self.device)

        # 初始化单应重定位误差loss
        self.logger.info("Initialize the HomographyReprojectionLoss and ReprojectionLoss")
        self.homo_loss = HomographyReprojectionLoss(self.device, self.params.height, self.params.width, self.half_region_size)
        self.repro_loss = ReprojectionLoss(self.device, self.params.height, self.params.width, self.half_region_size)

        # 初始化验证算子
        self.descriptor_validator = DescriptorValidator()

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
        if self.network_arch in ["resnet18", "resnet34", "deeplabv3_resnet50",
                                 "resnet18_c1", "resnet18_c2", "resnet18_c3", "resnet18_c4",
                                 "resnet18_c1c2", "resnet18_c1c3", "resnet18_c1c4", "resnet18_c2c3",
                                 "resnet18_c2c4", "resnet18_c3c4", "resnet18_c1c2c3", "resnet18_c1c2c4",
                                 "resnet18_c1c3c4", "resnet18_c2c3c4", "resnet18_c1c2c3c4",
                                 "resnet18_s1s2s3", "resnet18_s1s2s4", "resnet18_s1s3s4",
                                 "resnet18_s3s4", "resnet18_s1s2s3s4", ]:
            if self.train_mode == "only_detector":
                self.logger.info("Initialize training func mode of [only_detector] with baseline network.")
                self._train_func = self._train_only_detector
            elif self.train_mode == "only_detector_index":
                self.logger.info("Initialize training func mode of [only_detector_index] with baseline network.")
                self._train_func = self._train_only_detector_index
            elif self.train_mode == "only_descriptor":
                if self.desp_loss_type == "general":
                    self.logger.info("Initialize training func mode of [only_descriptor] with general train func.")
                    self._train_func = self._train_only_descriptor
                elif self.desp_loss_type == "rank":
                    self.logger.info("Initialize training func mode of [only_descriptor] with rank train func.")
                    self._train_func = self._train_only_descriptor_rank
            else:
                # self.logger.info("Initialize training func mode of [with_gt] with baseline network.")
                # self._train_func = self._train_with_gt
                self.logger.info("Initialize training func mode of _train_with_gt_all.")
                self._train_func = self._train_with_gt_all

        else:
            self.logger.error("Unrecognized network_arch: %s" % self.network_arch)
            assert False

    def _initialize_optimizer(self):
        # 初始化网络训练优化器
        self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr=self.lr)
        # self.extractor_optimizer = torch.optim.Adam(params=self.extractor.parameters(), lr=self.lr)

    def _initialize_scheduler(self):
        # 初始化学习率调整算子
        milestones = [10, 20, 30]
        self.logger.info("Initialize lr_scheduler of MultiStepLR: (%d, %d, %d)" % (
            milestones[0], milestones[1], milestones[2]))
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=milestones, gamma=0.1)

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

            desp_loss = self.descriptor_loss(
                desp_0, desp_1, matched_idx, matched_valid, not_search_mask)

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

    def _train_only_detector(self, epoch_idx):
        self.model.train()
        stime = time.time()
        for i, data in enumerate(self.train_dataloader):

            image = data['image'].to(self.device)
            heatmap_gt = data['heatmap'].to(self.device)
            point_mask = data['point_mask'].to(self.device)

            warped_image = data['warped_image'].to(self.device)
            warped_heatmap_gt = data['warped_heatmap'].to(self.device)
            warped_point_mask = data['warped_point_mask'].to(self.device)

            # homography = data["homography"].to(self.device)

            image_pair = torch.cat((image, warped_image), dim=0)
            heatmap_gt_pair = torch.cat((heatmap_gt, warped_heatmap_gt), dim=0)
            point_mask_pair = torch.cat((point_mask, warped_point_mask), dim=0)

            heatmap_pred_pair, _ = self.model(image_pair)

            # heatmap_pred, warped_heatmap_pred = torch.chunk(heatmap_pred_pair, 2, dim=0)
            # align_loss = self.align_loss(heatmap_pred, warped_heatmap_pred, homography, warped_point_mask,
            #                              heatmap_gt_t=warped_heatmap_gt)

            heatmap_pred_pair = heatmap_pred_pair.squeeze()

            point_loss = self.point_loss(heatmap_pred_pair, heatmap_gt_pair, point_mask_pair)

            # loss = point_loss + self.align_weight*align_loss
            loss = point_loss

            if torch.isnan(loss):
                self.logger.error('loss is nan!')

            self.optimizer.zero_grad()
            loss.backward()

            self.optimizer.step()

            if i % self.log_freq == 0:

                point_loss_val = point_loss.item()
                # align_loss_val = align_loss.item()
                align_loss_val = 0
                loss_val = loss.item()

                self.logger.info(
                    "[Epoch:%2d][Step:%5d:%5d]: loss = %.4f, point_loss = %.4f, align_loss=%.4f"
                    " one step cost %.4fs. " % (
                        epoch_idx, i, self.epoch_length,
                        loss_val,
                        point_loss_val,
                        align_loss_val,
                        (time.time() - stime) / self.params.log_freq,
                    ))
                stime = time.time()

        # save the model
        if self.multi_gpus:
            torch.save(self.model.module.state_dict(), os.path.join(self.ckpt_dir, 'model_%02d.pt' % epoch_idx))
        else:
            torch.save(self.model.state_dict(), os.path.join(self.ckpt_dir, 'model_%02d.pt' % epoch_idx))

    def _train_only_detector_index(self, epoch_idx):
        self.model.train()
        stime = time.time()
        for i, data in enumerate(self.train_dataloader):

            image = data['image'].to(self.device)
            heatmap_gt = data['heatmap'].to(self.device)
            point_mask = data['point_mask'].to(self.device)
            ridx = data["ridx"].to(self.device)
            center = data["center"].to(self.device)

            warped_image = data['warped_image'].to(self.device)
            warped_heatmap_gt = data['warped_heatmap'].to(self.device)
            warped_point_mask = data['warped_point_mask'].to(self.device)
            warped_ridx = data["warped_ridx"].to(self.device)
            warped_center = data["warped_center"].to(self.device)

            # idx = data["idx"]
            homography = data["homography"].to(self.device)

            image_pair = torch.cat((image, warped_image), dim=0)
            heatmap_gt_pair = torch.cat((heatmap_gt, warped_heatmap_gt), dim=0)
            point_mask_pair = torch.cat((point_mask, warped_point_mask), dim=0)

            heatmap_pred_pair, _ = self.model(image_pair)

            heatmap_pred, warped_heatmap_pred = torch.chunk(heatmap_pred_pair, 2, dim=0)
            align_loss = self.align_loss(heatmap_pred, warped_heatmap_pred, homography, warped_point_mask,
                                         heatmap_gt_t=warped_heatmap_gt)

            heatmap_pred_pair = heatmap_pred_pair.squeeze()

            point_loss = self.point_loss(heatmap_pred_pair, heatmap_gt_pair, point_mask_pair)

            homo_loss = self.homo_loss(heatmap_pred, heatmap_gt, ridx, center,
                                       warped_heatmap_pred, warped_heatmap_gt, warped_ridx, warped_center,
                                       homography)

            repro_loss = self.repro_loss(heatmap_pred, ridx, center,
                                         warped_heatmap_pred, warped_ridx, warped_center, homography)

            loss = point_loss + self.align_weight*align_loss + self.homo_weight*homo_loss + self.repro_weight * repro_loss

            if torch.isnan(loss):
                self.logger.error('loss is nan!')

            self.optimizer.zero_grad()
            loss.backward()

            self.optimizer.step()

            if i % self.log_freq == 0:

                point_loss_val = point_loss.item()
                align_loss_val = align_loss.item()
                homo_loss_val = homo_loss.item()
                repro_loss_val = repro_loss.item()
                loss_val = loss.item()

                self.summary_writer.add_scalar("homo_loss", homo_loss_val, global_step=int(i + epoch_idx*self.epoch_length))
                self.summary_writer.add_scalar("repro_loss", repro_loss_val, global_step=int(i + epoch_idx*self.epoch_length))

                self.logger.info(
                    "[Epoch:%2d][Step:%5d:%5d]: loss = %.4f, point_loss = %.4f, align_loss=%.4f, homo_loss=%03.4f,"
                    " repro_loss=%.4f,"
                    " %.4fs step. " % (
                        epoch_idx, i, self.epoch_length,
                        loss_val,
                        point_loss_val,
                        align_loss_val,
                        homo_loss_val,
                        repro_loss_val,
                        (time.time() - stime) / self.params.log_freq,
                    ))
                stime = time.time()

        # save the model
        if self.multi_gpus:
            torch.save(self.model.module.state_dict(), os.path.join(self.ckpt_dir, 'model_%02d.pt' % epoch_idx))
        else:
            torch.save(self.model.state_dict(), os.path.join(self.ckpt_dir, 'model_%02d.pt' % epoch_idx))

    def _train_with_gt_all(self, epoch_idx):
        self.model.train()
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
            # heatmap_pred_pair, c1_pair, c2_pair = self.model(image_pair)
            # heatmap_pred_pair, c1_pair = self.model(image_pair)

            # 计算描述子loss
            desp_point_pair = torch.cat((desp_point, warped_desp_point), dim=0)
            c1_feature_pair = f.grid_sample(c1_pair, desp_point_pair, mode="bilinear", padding_mode="border")
            c2_feature_pair = f.grid_sample(c2_pair, desp_point_pair, mode="bilinear", padding_mode="border")
            c3_feature_pair = f.grid_sample(c3_pair, desp_point_pair, mode="bilinear", padding_mode="border")
            c4_feature_pair = f.grid_sample(c4_pair, desp_point_pair, mode="bilinear", padding_mode="border")

            c1_0, c1_1 = torch.chunk(c1_feature_pair, 2, dim=0)
            c2_0, c2_1 = torch.chunk(c2_feature_pair, 2, dim=0)
            c3_0, c3_1 = torch.chunk(c3_feature_pair, 2, dim=0)
            c4_0, c4_1 = torch.chunk(c4_feature_pair, 2, dim=0)

            desp_0 = torch.cat((c1_0, c2_0, c3_0, c4_0), dim=1)[:, :, :, 0].transpose(1, 2)
            # desp_0 = torch.cat((c1_0, c2_0), dim=1)[:, :, :, 0].transpose(1, 2)
            # desp_0 = c1_0[:, :, :, 0].transpose(1, 2)
            desp_1 = torch.cat((c1_1, c2_1, c3_1, c4_1), dim=1)[:, :, :, 0].transpose(1, 2)
            # desp_1 = torch.cat((c1_1, c2_1), dim=1)[:, :, :, 0].transpose(1, 2)
            # desp_1 = c1_1[:, :, :, 0].transpose(1, 2)

            desp_0 = desp_0 / torch.norm(desp_0, dim=2, keepdim=True)
            desp_1 = desp_1 / torch.norm(desp_1, dim=2, keepdim=True)

            desp_loss = self.descriptor_loss(desp_0, desp_1, valid_mask, not_search_mask)

            # 计算关键点loss
            heatmap_gt_pair = torch.cat((heatmap_gt, warped_heatmap_gt), dim=0)
            point_mask_pair = torch.cat((point_mask, warped_point_mask), dim=0)
            point_loss = self.point_loss(heatmap_pred_pair[:, 0, :, :], heatmap_gt_pair, point_mask_pair)

            loss = desp_loss + point_loss

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

    def _train_only_descriptor(self, epoch_idx):
        self.model.train()
        # self.extractor.train()
        stime = time.time()
        for i, data in enumerate(self.train_dataloader):

            image = data["image"].to(self.device)
            desp_point = data["desp_point"].to(self.device)

            warped_image = data["warped_image"].to(self.device)
            warped_desp_point = data["warped_desp_point"].to(self.device)

            valid_mask = data["valid_mask"].to(self.device)
            not_search_mask = data["not_search_mask"].to(self.device)

            shape = image.shape

            image_pair = torch.cat((image, warped_image), dim=0)
            point_pair = torch.cat((desp_point, warped_desp_point), dim=0)

            # _, coarse_desp_pair = self.model(image_pair)
            # desp_pair = f.grid_sample(coarse_desp_pair, point_pair, mode="bilinear")[:, :, :, 0].transpose(1, 2)
            # desp_0, desp_1 = torch.chunk(desp_pair, 2, dim=0)

            # c1_pair, c2_pair, c3_pair, c4_pair = self.model(image_pair)
            desp_pair = self.model(image_pair, point_pair)
            desp_0, desp_1 = torch.chunk(desp_pair, 2, dim=0)

            # c1_feature_pair = f.grid_sample(c1_pair, point_pair, mode="bilinear", padding_mode="border")[:, :, :, 0].transpose(1, 2)
            # c2_feature_pair = f.grid_sample(c2_pair, point_pair, mode="bilinear", padding_mode="border")[:, :, :, 0].transpose(1, 2)
            # c3_feature_pair = f.grid_sample(c3_pair, point_pair, mode="bilinear", padding_mode="border")[:, :, :, 0].transpose(1, 2)
            # c4_feature_pair = f.grid_sample(c4_pair, point_pair, mode="bilinear", padding_mode="border")[:, :, :, 0].transpose(1, 2)

            # c1_0, c1_1 = torch.chunk(c1_feature_pair, 2, dim=0)
            # c2_0, c2_1 = torch.chunk(c2_feature_pair, 2, dim=0)
            # c3_0, c3_1 = torch.chunk(c3_feature_pair, 2, dim=0)
            # c4_0, c4_1 = torch.chunk(c4_feature_pair, 2, dim=0)

            # feature_0 = torch.cat((c1_0, c2_0, c3_0, c4_0), dim=2)
            # feature_1 = torch.cat((c1_1, c2_1, c3_1, c4_1), dim=2)

            # extract descriptor
            # desp_0 = self.extractor(feature_0)
            # desp_1 = self.extractor(feature_1)

            desp_loss = self.descriptor_loss(desp_0, desp_1, valid_mask, not_search_mask)

            loss = desp_loss

            if torch.isnan(loss):
                self.logger.error('loss is nan!')

            self.optimizer.zero_grad()
            # self.extractor_optimizer.zero_grad()
            loss.backward()

            self.optimizer.step()
            # self.extractor_optimizer.step()

            if i % self.log_freq == 0:

                desp_loss_val = desp_loss.item()
                loss_val = loss.item()
                self.summary_writer.add_scalar("loss", loss_val, global_step=int(i+epoch_idx*self.epoch_length))

                self.logger.info(
                    "[Epoch:%2d][Step:%5d:%5d]: "
                    "loss=%.4f, "
                    "desp=%.4f, "
                    "cost %.4fs/step. " % (
                        epoch_idx, i, self.epoch_length,
                        loss_val,
                        desp_loss_val,
                        (time.time() - stime) / self.params.log_freq,
                    ))
                stime = time.time()

        # save the model
        if self.multi_gpus:
            torch.save(self.model.module.state_dict(), os.path.join(self.ckpt_dir, 'model_%02d.pt' % epoch_idx))
            # torch.save(self.extractor.module.state_dict(), os.path.join(self.ckpt_dir, 'extractor_%02d.pt' % epoch_idx))
        else:
            torch.save(self.model.state_dict(), os.path.join(self.ckpt_dir, 'model_%02d.pt' % epoch_idx))
            # torch.save(self.extractor.state_dict(), os.path.join(self.ckpt_dir, 'extractor_%02d.pt' % epoch_idx))

    def _train_only_descriptor_rank(self, epoch_idx):
        self.model.train()
        stime = time.time()

        # todo: debug use
        torch.autograd.set_detect_anomaly(True)

        for i, data in enumerate(self.train_dataloader):

            image = data["image"].to(self.device)
            desp_point = data["desp_point"].to(self.device)

            warped_image = data["warped_image"].to(self.device)
            warped_desp_point = data["warped_desp_point"].to(self.device)

            valid_mask = data["valid_mask"].to(self.device)
            not_search_mask = data["not_search_mask"].to(self.device)

            image_pair = torch.cat((image, warped_image), dim=0)
            point_pair = torch.cat((desp_point, warped_desp_point), dim=0)

            desp_pair = self.model(image_pair, point_pair)
            desp_0, desp_1 = torch.chunk(desp_pair, 2, dim=0)

            desp_loss, desp_positive_loss, desp_negative_loss = self.descriptor_loss(
                desp_0, desp_1, valid_mask, not_search_mask)

            loss = desp_loss

            if torch.isnan(loss):
                self.logger.error('loss is nan!')

            self.optimizer.zero_grad()
            loss.backward()

            self.optimizer.step()

            if i % self.log_freq == 0:

                desp_loss_val = desp_loss.item()
                positive_loss_val = desp_positive_loss.item()
                negative_loss_val = desp_negative_loss.item()

                self.summary_writer.add_scalar("loss", desp_loss_val, global_step=int(i+epoch_idx*self.epoch_length))
                self.summary_writer.add_scalar("positive_loss", positive_loss_val, int(i+epoch_idx*self.epoch_length))
                self.summary_writer.add_scalar("negative_loss", negative_loss_val, int(i+epoch_idx*self.epoch_length))

                self.logger.info(
                    "[Epoch:%2d][Step:%5d:%5d]: "
                    "loss=%.4f, "
                    "positive_loss=%.4f, "
                    "negative_loss=%.4f, "
                    "cost %.4fs/step. " % (
                        epoch_idx, i, self.epoch_length,
                        desp_loss_val,
                        positive_loss_val,
                        negative_loss_val,
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

        if self.train_mode == "only_descriptor":
            self._megadepth_validation_func(epoch_idx)
            self.logger.info("Validating epoch %2d done." % epoch_idx)
            self.logger.info("*****************************************************")
            return
        else:
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
        self.model = self._load_model_params(ckpt_file, self.model)

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

            # 若模式是只训练描述子，则检测子端采用预训练的网络实现
            if self.train_mode in ["only_descriptor"]:
                heatmap_pair, _ = self.detector(image_pair)

            # 若模式是只训检测子，则描述子端采用预训练的网络实现
            if self.train_mode in ["only_detector", "only_detector_index"]:
                _, desp_pair = self.descriptor(image_pair)

            heatmap_pair = torch.sigmoid(heatmap_pair)  # 用BCEloss
            prob_pair = spatial_nms(heatmap_pair, kernel_size=int(self.nms_threshold*2+1))

            desp_pair = desp_pair.detach().cpu().numpy()
            first_desp = desp_pair[0]
            second_desp = desp_pair[1]
            prob_pair = prob_pair.detach().cpu().numpy()
            first_prob = prob_pair[0, 0]
            second_prob = prob_pair[1, 0]

            if self.train_mode == "only_detector_index":
                org_prob_pair = heatmap_pair
                org_prob_pair = org_prob_pair.detach().cpu().numpy()
                org_first_prob = org_prob_pair[0, 0]
                org_second_prob = org_prob_pair[1, 0]

            # 得到对应的预测点
            if self.detection_mode == "use_network":
                first_point, first_point_num = self._generate_predict_point(first_prob, top_k=self.top_k)  # [n,2]
                second_point, second_point_num = self._generate_predict_point(second_prob, top_k=self.top_k)  # [m,2]

                if self.train_mode == "only_detector_index":
                    first_point = self._generate_subpixel_point(org_first_prob, first_point)
                    second_point = self._generate_subpixel_point(org_second_prob, second_point)

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

            if self.train_mode == "only_descriptor":
                results = self._descriptor_inference_func(image_pair)
            else:
                # results = self._baseline_inference_func(image_pair)
                results = self._inference_func(image_pair)
            # results = self._baseline_inference_func(image_pair)

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

    def _baseline_inference_func(self, image_pair):
        """
        用于估计点和对应描述子的函数
        """
        if self.network_arch == "residual":
            heatmap_pair, desp_pair, _ = self.model(image_pair)
        else:
            heatmap_pair, desp_pair = self.model(image_pair)

        # 若模式是只训练描述子，则检测子端采用预训练的网络实现
        if self.train_mode in ["only_descriptor"]:
            heatmap_pair, _ = self.detector(image_pair)

        # 若模式是只训检测子，则描述子端采用预训练的网络实现
        if self.train_mode in ["only_detector", "only_detector_index"]:
            _, desp_pair = self.descriptor(image_pair)

        heatmap_pair = torch.sigmoid(heatmap_pair)
        prob_pair = spatial_nms(heatmap_pair, kernel_size=int(self.nms_threshold * 2 + 1))

        desp_pair = desp_pair.detach().cpu().numpy()
        first_desp = desp_pair[0]
        second_desp = desp_pair[1]
        prob_pair = prob_pair.detach().cpu().numpy()
        first_prob = prob_pair[0, 0]
        second_prob = prob_pair[1, 0]

        # if self.train_mode == "only_detector_index":
        #     org_prob_pair = heatmap_pair
        #     org_prob_pair = org_prob_pair.detach().cpu().numpy()
        #     org_first_prob = org_prob_pair[0, 0]
        #     org_second_prob = org_prob_pair[1, 0]

        # 得到对应的预测点
        first_point, first_point_num = self._generate_predict_point(first_prob, top_k=self.top_k)  # [n,2]
        second_point, second_point_num = self._generate_predict_point(second_prob, top_k=self.top_k)  # [m,2]

        # if self.train_mode == "only_detector_index":
        #     first_point = self._generate_subpixel_point(org_first_prob, first_point)
        #     second_point = self._generate_subpixel_point(org_second_prob, second_point)

        if first_point_num <= 4 or second_point_num <= 4:
            print("skip this pair because there's little point!")
            return None

        # 得到点对应的描述子
        select_first_desp = self._generate_predict_descriptor(first_point, first_desp)
        select_second_desp = self._generate_predict_descriptor(second_point, second_desp)

        return first_point, first_point_num, second_point, second_point_num, select_first_desp, select_second_desp

    def _inference_func(self, image_pair):
        """
        image_pair: [2,1,h,w]
        """
        self.model.eval()
        _, _, height, width = image_pair.shape
        heatmap_pair, c1_pair, c2_pair, c3_pair, c4_pair = self.model(image_pair)
        # heatmap_pair, c1_pair, c2_pair = self.model(image_pair)
        # heatmap_pair, c1_pair = self.model(image_pair)

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
        select_first_desp = self._generate_combined_descriptor(first_point, c1_0, c2_0, c3_0, c4_0, height, width)
        # select_first_desp = self._generate_combined_descriptor(first_point, c1_0, c2_0, height, width)
        # select_first_desp = self._generate_combined_descriptor(first_point, c1_0, height, width)
        select_second_desp = self._generate_combined_descriptor(second_point, c1_1, c2_1, c3_1, c4_1, height, width)
        # select_second_desp = self._generate_combined_descriptor(second_point, c1_1, c2_1, height, width)
        # select_second_desp = self._generate_combined_descriptor(second_point, c1_1, height, width)

        return first_point, first_point_num, second_point, second_point_num, select_first_desp, select_second_desp

    def _megadepth_validation_func(self, epoch_idx):
        """
        用megadepth validation dataset进行验证的函数
        """
        self.model.eval()
        # self.extractor.eval()
        avg_correct_ratio = []
        stime = time.time()
        for i, data in enumerate(self.val_dataloader):

            image = data["image"].to(self.device)
            desp_point = data["desp_point"].to(self.device)

            warped_image = data["warped_image"].to(self.device)
            warped_desp_point = data["warped_desp_point"].to(self.device)

            valid_mask = data["valid_mask"].to(self.device)
            not_search_mask = data["not_search_mask"].to(self.device)

            image_pair = torch.cat((image, warped_image), dim=0)
            point_pair = torch.cat((desp_point, warped_desp_point), dim=0)

            with torch.no_grad():
                desp_pair = self.model(image_pair, point_pair)
                desp_0, desp_1 = torch.chunk(desp_pair, 2, dim=0)

            correct_ratio = self.descriptor_validator(desp_0, desp_1, valid_mask, not_search_mask)

            correct_ratio_val = correct_ratio.item()

            torch.cuda.empty_cache()

            if i % self.log_freq == 0:

                self.logger.info(
                    "[Val Epoch:%2d][Step:%5d:%5d]: "
                    "correct_ratio=%.4f, "
                    "cost %.4fs/step. " % (
                        epoch_idx, i, self.val_epoch_length,
                        correct_ratio_val,
                        (time.time() - stime) / self.params.log_freq,
                    ))
                stime = time.time()

            avg_correct_ratio.append(correct_ratio_val)

        avg_correct_ratio = np.mean(np.stack(avg_correct_ratio))

        self.summary_writer.add_scalar("validation/avg_correct_ratio", avg_correct_ratio, global_step=epoch_idx)

        self.logger.info(
            "[Val Epoch:%2d], "
            "avg_correct_ratio=%.4f, "
            % (
                epoch_idx,
                avg_correct_ratio,
            )
        )

    def _descriptor_inference_func(self, image_pair):
        """
        专用于descriptor网络估计点和对应描述子的函数
        image_pair: [2,1,h,w]
        """
        self.model.eval()
        self.detector.eval()
        # self.extractor.eval()
        _, c, height, width = image_pair.shape
        if c == 1:
            image_pair_4_desp = image_pair.repeat((1, 3, 1, 1))
        else:
            image_pair_4_desp = image_pair
        c1_pair, c2_pair, c3_pair, c4_pair = self.model(image_pair_4_desp)
        # _, coarse_desp_pair = self.model(image_pair)
        # coarse_desp_0, coarse_desp_1 = torch.chunk(coarse_desp_pair, 2, dim=0)

        c1_0, c1_1 = torch.chunk(c1_pair, 2, dim=0)
        c2_0, c2_1 = torch.chunk(c2_pair, 2, dim=0)
        c3_0, c3_1 = torch.chunk(c3_pair, 2, dim=0)
        c4_0, c4_1 = torch.chunk(c4_pair, 2, dim=0)

        heatmap_pair, _ = self.detector(image_pair)

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
        select_first_desp = self._generate_combined_descriptor(first_point, c1_0, c2_0, c3_0, c4_0, height, width)
        select_second_desp = self._generate_combined_descriptor(second_point, c1_1, c2_1, c3_1, c4_1, height, width)

        return first_point, first_point_num, second_point, second_point_num, select_first_desp, select_second_desp

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

    def _generate_subpixel_point(self, prob, point):
        """以输入点位置为中心取框，计算框中的加权平均点的位置"""
        org_point = point
        num_point = point.shape[0]
        ridx_list = []
        center_list = []
        for i in range(num_point):
            ridx, center_point = self._generate_region_index(point[i])
            ridx_list.append(ridx)
            center_list.append(center_point)
        ridx = torch.from_numpy(np.stack(ridx_list, axis=0)).reshape((-1,))  # [n*k*k]
        center = torch.from_numpy(np.stack(center_list, axis=0))  # [n,2]

        prob = torch.from_numpy(prob).reshape((-1,))
        select_prob = torch.gather(prob, dim=0, index=ridx).reshape((num_point, -1))  # [n,k*k]
        select_prob = select_prob / torch.sum(select_prob, dim=1, keepdim=True)
        select_prob = select_prob.unsqueeze(dim=2)  # [n,k*k,1]

        general_coords = torch.from_numpy(self.general_coords).unsqueeze(dim=0)  # [1,k*k,2]
        point = torch.sum(general_coords * select_prob, dim=1) + center
        point = point.detach().numpy()

        return point

    def _generate_region_index(self, point):
        """输入一个点，增加扰动并输出包含该点的指定大小区域的各个点的index
        Args:
            point: (2,) 分别为y,x坐标值
        Returns：
            region_idx: (rsize, rsize)，各个位置处的值为其在图像中的索引值
            upleft_coord: 区域左上角的坐标
        """
        point = np.round(point).astype(np.long)
        pt_y, pt_x = point
        height = self.params.hpatch_height
        width = self.params.hpatch_width

        # 保证所取区域在图像范围内,pt代表所取区域的中心点位置
        if pt_y - self.half_region_size < 0:
            pt_y = self.half_region_size
        elif pt_y + self.half_region_size > height - 1:
            pt_y = height - 1 - self.half_region_size

        if pt_x - self.half_region_size < 0:
            pt_x = self.half_region_size
        elif pt_x + self.half_region_size > width - 1:
            pt_x = width - 1 - self.half_region_size

        center_point = np.array((pt_y, pt_x))

        # 得到区域中各个点的坐标
        region_coords = self.general_coords.copy()
        region_coords += center_point

        # 将坐标转换为idx
        coords_y, coords_x = np.split(region_coords, 2, axis=1)
        region_idx = (coords_y * width + coords_x).astype(np.long).reshape((-1,))
        return region_idx, center_point.astype(np.float32)

    def _initialize_general_coords(self, half_region_size):
        """
        构造区域点中的初始坐标
        Returns:
            coords: (ksize*kszie,2)
        """
        region_size = int(2*half_region_size+1)
        coords_y = np.tile(
            np.arange(-half_region_size, half_region_size+1)[:, np.newaxis], (1, region_size))
        coords_x = np.tile(
            np.arange(-half_region_size, half_region_size+1)[np.newaxis, :], (region_size, 1))
        coords = np.stack((coords_y, coords_x), axis=2).reshape((region_size**2, 2)).astype(np.float32)
        self.general_coords = coords

    def _cvpoint2numpy(self, point_cv):
        """将opencv格式的特征点转换成numpy数组"""
        point_list = []
        for pt_cv in point_cv:
            point = np.array((pt_cv.pt[1], pt_cv.pt[0]))
            point_list.append(point)
        point_np = np.stack(point_list, axis=0)
        return point_np

    def _generate_combined_descriptor(self, point, c1, c2, c3, c4, height, width):
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

        c1_feature = f.grid_sample(c1, point, mode="bilinear")[0, :, :, 0].transpose(0, 1)
        c2_feature = f.grid_sample(c2, point, mode="bilinear")[0, :, :, 0].transpose(0, 1)
        c3_feature = f.grid_sample(c3, point, mode="bilinear")[0, :, :, 0].transpose(0, 1)
        c4_feature = f.grid_sample(c4, point, mode="bilinear")[0, :, :, 0].transpose(0, 1)

        # 分开进行归一化
        c1_feature = c1_feature / torch.norm(c1_feature, dim=1, keepdim=True)
        c2_feature = c2_feature / torch.norm(c2_feature, dim=1, keepdim=True)
        c3_feature = c3_feature / torch.norm(c3_feature, dim=1, keepdim=True)
        c4_feature = c4_feature / torch.norm(c4_feature, dim=1, keepdim=True)

        desp = torch.cat((c1_feature, c2_feature, c3_feature, c4_feature), dim=1)
        # desp = torch.cat((c1_feature, c2_feature), dim=1)
        # desp = c1_feature
        # desp = desp / torch.norm(desp, dim=1, keepdim=True)
        desp = desp / 2.

        # c1_feature = f.grid_sample(c1, point, mode="bilinear")[:, :, :, 0].transpose(1, 2)
        # c2_feature = f.grid_sample(c2, point, mode="bilinear")[:, :, :, 0].transpose(1, 2)
        # c3_feature = f.grid_sample(c3, point, mode="bilinear")[:, :, :, 0].transpose(1, 2)
        # c4_feature = f.grid_sample(c4, point, mode="bilinear")[:, :, :, 0].transpose(1, 2)
        # feature = torch.cat((c1_feature, c2_feature, c3_feature, c4_feature), dim=2)
        # desp = self.extractor(feature)[0]  # [n,128]

        desp = desp.detach().cpu().numpy()

        return desp

    def _generate_predict_descriptor(self, point, desp, return_diff=False):
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
        nonorm_desp = interpolation_desp
        interpolation_norm = torch.norm(interpolation_desp, dim=1, keepdim=True)
        interpolation_desp = interpolation_desp/interpolation_norm

        if not return_diff:
            return interpolation_desp.numpy()
        else:
            diff = torch.norm((nonorm_desp - interpolation_desp), dim=1, keepdim=False)
            return interpolation_desp.numpy(), diff.numpy()

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

    def test_debug(self, ckpt_file, mode="MegPoint", mode_ckpt=None, use_sift=False):
        if mode == "MegPoint":
            model = MegPointShuffleHeatmapOld()
            self.model = model.to(self.device)
            self.model = self._load_model_params(ckpt_file, self.model)
        elif mode == "MagicLeap":
            model = SuperPointNetFloat()
            self.model = model.to(self.device)
            self.model = self._load_model_params(ckpt_file, self.model)
        else:
            self.logger.error("Unrecognized mode: %s" % mode)
            assert False

        if use_sift:
            self.logger.info("Use sift as feature detector.")
        sift = cv.xfeatures2d_SIFT.create(1000)

        desp = None
        if mode == "MagicLeap":
            assert mode_ckpt is not None
            desp = resnet18()
            # desp = MegPointNew()
            desp = desp.to(self.device)
            desp = self._load_model_params(mode_ckpt, desp)

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

            if mode == "MagicLeap":
                image_pair = image_pair / 255.
                _, desp_pair, heatmap_pair, _ = self.model(image_pair)
                heatmap_pair = f.pixel_shuffle(heatmap_pair, 8)
            else:
                image_pair = image_pair * 2. / 255. - 1.
                heatmap_pair, desp_pair = self.model(image_pair)
                heatmap_pair = torch.sigmoid(heatmap_pair)

            prob_pair = spatial_nms(heatmap_pair, kernel_size=int(self.nms_threshold * 2 + 1))

            desp_pair = desp_pair.detach().cpu().numpy()
            first_desp = desp_pair[0]
            second_desp = desp_pair[1]
            prob_pair = prob_pair.detach().cpu().numpy()
            first_prob = prob_pair[0, 0]
            second_prob = prob_pair[1, 0]

            # 得到对应的预测点
            if not use_sift:
                first_point, first_point_num = self._generate_predict_point(first_prob, top_k=self.top_k)  # [n,2]
                second_point, second_point_num = self._generate_predict_point(second_prob, top_k=self.top_k)  # [m,2]
            else:
                first_point = sift.detect(first_image)
                second_point = sift.detect(second_image)
                first_point = self._convert_cv2pt(first_point)
                second_point = self._convert_cv2pt(second_point)
                first_point_num = first_point.shape[0]
                second_point_num = second_point.shape[0]

            if first_point_num <= 4 or second_point_num <= 4:
                print("skip this pair because there's little point!")
                if image_type == "illumination":
                    illum_skip += 1
                else:
                    view_skip += 1
                continue

            # 得到点对应的描述子
            if desp is None:
                select_first_desp = self._generate_predict_descriptor(first_point, first_desp)
                select_second_desp = self._generate_predict_descriptor(second_point, second_desp)
            else:
                _, _, height, width = image_pair.shape
                c1_pair, c2_pair, c3_pair, c4_pair = desp(image_pair)

                c1_0, c1_1 = torch.chunk(c1_pair, 2, dim=0)
                c2_0, c2_1 = torch.chunk(c2_pair, 2, dim=0)
                c3_0, c3_1 = torch.chunk(c3_pair, 2, dim=0)
                c4_0, c4_1 = torch.chunk(c4_pair, 2, dim=0)

                select_first_desp = self._generate_combined_descriptor(first_point, c1_0, c2_0, c3_0, c4_0, height,
                                                                       width)
                select_second_desp = self._generate_combined_descriptor(second_point, c1_1, c2_1, c3_1, c4_1, height,
                                                                        width)

            # 得到匹配点
            matched_point = self.general_matcher(first_point, select_first_desp,
                                                 second_point, select_second_desp)
            # debug
            # 误差缩小指数
            reduce_diff = 0.
            correct_first_list = []
            correct_second_list = []
            wrong_first_list = []
            wrong_second_list = []
            xy_first_point = matched_point[0][:, ::-1]
            xy1_first_point = np.concatenate((xy_first_point, np.ones((xy_first_point.shape[0], 1))), axis=1)
            xyz_second_point = np.matmul(gt_homography, xy1_first_point[:, :, np.newaxis])[:, :, 0]
            xy_second_point = xyz_second_point[:, :2] / xyz_second_point[:, 2:3]
            # delta_diff = xy_second_point - matched_point[1][:, ::-1]
            # matched_point = (
            #     matched_point[0],
            #     (matched_point[1][:, ::-1] + reduce_diff*delta_diff)[:, ::-1])

            matched_second_point = []
            project_second_point = []
            select_second_point = []
            select_first_point = []
            # 重新计算经误差缩小后的投影误差
            diff = np.linalg.norm(xy_second_point - matched_point[1][:, ::-1], axis=1)
            for j in range(xy_first_point.shape[0]):
                # 重投影误差小于3的判断为正确匹配
                if diff[j] < 3:
                    correct_first_list.append(matched_point[0][j])
                    delta_diff = reduce_diff*(xy_second_point[j] - matched_point[1][j, ::-1])[::-1]
                    correct_second_list.append(matched_point[1][j]+delta_diff)
                    matched_second_point.append(np.round(matched_point[1][j]+delta_diff))  # 整型最近点
                    # matched_second_point.append(matched_point[1][j]+delta_diff)  # 浮点型最近点
                    if diff[j] >= 3 or diff[j] <= 5:
                        select_first_point.append(matched_point[0][j])
                        select_second_point.append(matched_point[1][j])
                        project_second_point.append(xy_second_point[j][::-1])
                else:
                    wrong_first_list.append(matched_point[0][j])
                    wrong_second_list.append(matched_point[1][j])
                    matched_second_point.append(matched_point[1][j])

            matched_second_point = np.stack(matched_second_point, axis=0)
            matched_point = (matched_point[0], matched_second_point)

            cv_correct_first, cv_correct_second, cv_correct_matched = self._convert_match2cv(
                correct_first_list,
                correct_second_list)
            cv_wrong_first, cv_wrong_second, cv_wrong_matched = self._convert_match2cv(
                wrong_first_list,
                wrong_second_list,
                0.25)

            cv_matched_first_point = self._convert_pt2cv(select_first_point)
            cv_matched_second_point = self._convert_pt2cv(select_second_point)
            cv_project_second_point = self._convert_pt2cv(project_second_point)
            assert len(cv_matched_second_point) == len(cv_project_second_point)
            if len(cv_matched_second_point) > 0:
                point_image_0 = cv.drawKeypoints(first_image, cv_matched_first_point, None, color=(255, 0, 0))
                point_image = cv.drawKeypoints(second_image, cv_matched_second_point, None, color=(0, 0, 255))
                point_image_2 = cv.drawKeypoints(point_image, cv_project_second_point, None, color=(0, 255, 0))
                # point_image_0 = cv.resize(point_image_0, (640, 480), interpolation=cv.INTER_LINEAR)
                # point_image = cv.resize(point_image, (640, 480), interpolation=cv.INTER_LINEAR)
                # point_image_2 = cv.resize(point_image_2, (640, 480), interpolation=cv.INTER_LINEAR)
                first_image_tmp = np.tile(first_image[:, :, np.newaxis], (1, 1, 3))
                second_image_tmp = np.tile(second_image[:, :, np.newaxis], (1, 1, 3))
                point_image = np.concatenate((first_image_tmp, point_image_0, point_image, point_image_2, second_image_tmp), axis=1)
                cv.imwrite("/home/zhangyuyang/tmp_images/megpoint_pointimage/image_%03d.jpg" % i, point_image)

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
                                                       matched_point[1][:, np.newaxis, ::-1], cv.RANSAC, maxIters=3000)
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
                repeat_0, nonrepeat_0, repeat_1, nonrepeat_1 = self.illum_repeat.update(
                    first_point, second_point, gt_homography, return_repeat=True)

                repeat_0 = self._convert_pt2cv_np(repeat_0)
                nonrepeat_0 = self._convert_pt2cv_np(nonrepeat_0)
                repeat_1 = self._convert_pt2cv_np(repeat_1)
                nonrepeat_1 = self._convert_pt2cv_np(nonrepeat_1)

                debug_first_img = cv.drawKeypoints(first_image, repeat_0, None, color=(0, 255, 0))
                debug_first_img = cv.drawKeypoints(debug_first_img, nonrepeat_0, None, color=(0, 0, 255))

                debug_second_img = cv.drawKeypoints(second_image, repeat_1, None, color=(0, 255, 0))
                debug_second_img = cv.drawKeypoints(debug_second_img, nonrepeat_1, None, color=(0, 0, 255))
                debug_img = np.concatenate((debug_first_img, debug_second_img), axis=1)
                cv.imwrite("/home/zhangyuyang/tmp_images/megpoint_repeat/image_%03d.jpg" % i, debug_img)

                correct, diff = self.illum_homo_acc.update(pred_homography, gt_homography, True)
                self.illum_mma.update(gt_homography, matched_point)

                if not correct:
                    self.illum_bad_mma.update(gt_homography, matched_point)
                    bad += 1

                    # if len(cv_correct_matched) != 0:
                    correct_matched_image = cv.drawMatches(
                        first_image, cv_correct_first, second_image, cv_correct_second,
                        cv_correct_matched, None)
                    wrong_matched_image = cv.drawMatches(
                        first_image, cv_wrong_first, second_image, cv_wrong_second,
                        cv_wrong_matched, None)
                    metric_str = "diff between gt & pred is %.4f, correct match: %d/ total: %d, %.4f" % (
                        diff, len(correct_first_list), matched_point[0].shape[0],
                        len(correct_first_list) / matched_point[0].shape[0]
                    )
                    cv.putText(correct_matched_image, metric_str, (0, 40), cv.FONT_HERSHEY_COMPLEX, fontScale=0.8,
                               color=(0, 0, 255), thickness=2)
                    matched_image = np.concatenate((correct_matched_image, wrong_matched_image), axis=0)
                    cv.imwrite("/home/zhangyuyang/tmp_images/megpoint_bad/image_%03d.jpg" % i, matched_image)

            elif image_type == 'viewpoint':
                repeat_0, nonrepeat_0, repeat_1, nonrepeat_1 = self.view_repeat.update(
                    first_point, second_point, gt_homography, return_repeat=True)

                repeat_0 = self._convert_pt2cv_np(repeat_0)
                nonrepeat_0 = self._convert_pt2cv_np(nonrepeat_0)
                repeat_1 = self._convert_pt2cv_np(repeat_1)
                nonrepeat_1 = self._convert_pt2cv_np(nonrepeat_1)

                debug_first_img = cv.drawKeypoints(first_image, repeat_0, None, color=(0, 255, 0))
                debug_first_img = cv.drawKeypoints(debug_first_img, nonrepeat_0, None, color=(0, 0, 255))

                debug_second_img = cv.drawKeypoints(second_image, repeat_1, None, color=(0, 255, 0))
                debug_second_img = cv.drawKeypoints(debug_second_img, nonrepeat_1, None, color=(0, 0, 255))
                debug_img = np.concatenate((debug_first_img, debug_second_img), axis=1)
                cv.imwrite("/home/zhangyuyang/tmp_images/megpoint_repeat/image_%03d.jpg" % i, debug_img)

                correct, diff = self.view_homo_acc.update(pred_homography, gt_homography, True)
                self.view_mma.update(gt_homography, matched_point)

                if not correct:
                    self.view_bad_mma.update(gt_homography, matched_point)
                    bad += 1

                    # if len(cv_correct_matched) != 0:
                    correct_matched_image = cv.drawMatches(
                        first_image, cv_correct_first, second_image, cv_correct_second,
                        cv_correct_matched, None)
                    wrong_matched_image = cv.drawMatches(
                        first_image, cv_wrong_first, second_image, cv_wrong_second,
                        cv_wrong_matched, None)
                    metric_str = "diff between gt & pred is %.4f, correct match: %d/ total: %d, %.4f" % (
                        diff, len(correct_first_list), matched_point[0].shape[0],
                        len(correct_first_list) / matched_point[0].shape[0]
                    )
                    cv.putText(correct_matched_image, metric_str, (0, 40), cv.FONT_HERSHEY_COMPLEX, fontScale=0.8,
                               color=(0, 0, 255), thickness=2)
                    matched_image = np.concatenate((correct_matched_image, wrong_matched_image), axis=0)
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



