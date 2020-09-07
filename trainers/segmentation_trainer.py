#
# Created by ZhangYuyang on 2020/8/31
#
import os
import time

import torch
import cv2
from tqdm import tqdm
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tensorboardX import SummaryWriter

from data_utils import get_dataset
from .utils import resize_labels
from .utils import PolynomialLR
from .utils import scores
from utils.utils import DescriptorGeneralTripletLoss, PointHeatmapWeightedBCELoss
from utils.utils import Matcher
from utils.utils import spatial_nms
from utils.evaluation_tools import *
from nets import get_model


class _BaseTrainer(object):

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

        # todo: 初始化测试数据集
        # 初始化测试集
        # self.logger.info("Initialize {}".format(self.config['test']['dataset']))
        # self.test_dataset = get_dataset(self.config['test']['dataset'])(**self.config['test'])
        # self.test_length = len(self.test_dataset)

        self._initialize_dataset()
        self._initialize_model()
        self._initialize_loss()
        self._initialize_optimizer()
        self._initialize_scheduler()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

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

    def _validate_one_epoch(self, *args, **kwargs):
        raise NotImplementedError

    def _load_model_params(self, ckpt_file, previous_model):
        if ckpt_file is None:
            print("Please input correct checkpoint file dir!")
            return False

        self.logger.info("Load pretrained model %s " % ckpt_file)
        if not self.multi_gpus:
            model_dict = previous_model.state_dict()
            pretrain_dict = torch.load(ckpt_file, map_location=self.device)
            pretrain_dict = {k: v for k, v in pretrain_dict.items() if k in model_dict}
            model_dict.update(pretrain_dict)
            previous_model.load_state_dict(model_dict)
        else:
            model_dict = previous_model.module.state_dict()
            pretrain_dict = torch.load(ckpt_file, map_location=self.device)
            pretrain_dict = {k: v for k, v in pretrain_dict.items() if k in model_dict}
            model_dict.update(pretrain_dict)
            previous_model.module.load_state_dict(model_dict)
        return previous_model

    def train(self):
        start_time = time.time()

        # start training
        for i in range(self.config['train']['epoch_num']):

            # train
            self._train_one_epoch(i)

            # todo: validation
            if i >= int(self.config['train']['epoch_num'] * self.config['train']['validate_after']):
                self._validate_one_epoch(i)

            # todo: move to the inside of training
            # if self.config['train']['adjust_lr']:
            #     # adjust learning rate
            #     self.scheduler.step(i)

        end_time = time.time()
        self.logger.info("The whole training process takes %.3f h" % ((end_time - start_time)/3600))


class BiSeNetV1SegmentationTrainer(_BaseTrainer):

    def __init__(self, **config):
        super(BiSeNetV1SegmentationTrainer, self).__init__(**config)

    def _initialize_dataset(self):
        # 初始化训练集
        self.logger.info('Initialize {}'.format(self.config['train']['dataset']))
        self.train_dataset = get_dataset(self.config['train']['dataset'])(**self.config['train'])

        self.train_dataloader = DataLoader(
            dataset=self.train_dataset,
            batch_size=self.config['train']['batch_size'],
            shuffle=True,
            num_workers=self.config['train']['num_workers'],
            drop_last=True
        )
        self.epoch_length = len(self.train_dataset) // self.config['train']['batch_size']

        # 初始化测试集
        self.logger.info('Initialize test {}'.format(self.config['test']['dataset']))
        self.test_dataset = get_dataset(self.config['test']['dataset'])(**self.config['test'])

    def _initialize_model(self):
        self.logger.info("Initialize network arch {}".format(self.config['model']['backbone']))
        model = get_model(self.config['model']['backbone'])(**self.config['model'])

        if self.multi_gpus:
            model = torch.nn.DataParallel(model)
        self.model = model.to(self.device)

    def _initialize_loss(self):
        # 初始化loss算子
        self.logger.info("Initialize the CrossEntropyLoss.")
        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=self.config['train']['ignore_label']).to(self.device)

    def _initialize_optimizer(self):
        # 初始化网络训练优化器
        self.logger.info("Initialize Adam optimizer with weight_decay: {:.5f}.".format(self.config['train']['weight_decay']))
        self.optimizer = torch.optim.Adam(
            params=self.model.parameters(),
            lr=self.config['train']['lr'],
            weight_decay=self.config['train']['weight_decay'])

    def _initialize_scheduler(self):
        self.logger.info('Initialize PolynomialLR')
        self.scheduler = PolynomialLR(
            optimizer=self.optimizer,
            step_size=self.config['train']['lr_decay'],
            iter_max=self.config['train']['epoch_num']*self.epoch_length,
            power=self.config['train']['poly_power'],
        )

    def _train_one_epoch(self, epoch_idx):
        self.model.train()

        self.logger.info("-----------------------------------------------------")
        self.logger.info("Training epoch %2d begin:" % epoch_idx)

        self._train_func(epoch_idx)

        self.logger.info("Training epoch %2d done." % epoch_idx)
        self.logger.info("-----------------------------------------------------")

    def _train_func(self, epoch_idx):
        stime = time.time()
        for i, data in enumerate(self.train_dataloader):
            # read
            images = data["image"]
            labels = data['label']

            # forward
            logits = self.model(images.to(self.device))

            # loss
            loss = []
            for logit in logits:
                _, _, H, W = logit.shape
                label_ = resize_labels(labels, size=(W, H))
                loss.append(self.criterion(logit, label_.to(self.device)))

            loss = torch.mean(torch.stack(loss))

            if torch.isnan(loss):
                self.logger.error('loss is nan!')

            self.optimizer.zero_grad()

            loss.backward()

            self.optimizer.step()

            self.scheduler.step(epoch=i+epoch_idx*self.epoch_length)

            # debug use
            # if i == 200:
            #     break
            if i % self.config['train']['log_freq'] == 0:
                loss_val = loss.item()

                self.summary_writer.add_scalar("loss/train", loss_val, i+epoch_idx*self.epoch_length)
                for k, o in enumerate(self.optimizer.param_groups):
                    self.summary_writer.add_scalar("lr/group_{}".format(k), o["lr"], i+epoch_idx*self.epoch_length)

                self.logger.info(
                    "[Epoch:%2d][Step:%5d:%5d]: loss = %.4f, one step cost %.4fs. " % (
                        epoch_idx, i, self.epoch_length,
                        loss_val,
                        (time.time() - stime) / self.config['train']['log_freq'],
                    ))
                stime = time.time()

        # save the model
        if self.multi_gpus:
            # torch.save(self.model.module.state_dict(), os.path.join(self.config['ckpt_path'], 'model_%02d.pt' % epoch_idx))
            torch.save(self.model.module.state_dict(), os.path.join(self.config['ckpt_path'], 'model_final.pt'))
        else:
            # torch.save(self.model.state_dict(), os.path.join(self.config['ckpt_path'], 'model_%02d.pt' % epoch_idx))
            torch.save(self.model.state_dict(), os.path.join(self.config['ckpt_path'], 'model_final.pt'))

    def _validate_one_epoch(self, epoch_idx):
        self.model.eval()

        # self.logger.info("-----------------------------------------------------")
        # self.logger.info("Validate epoch %2d begin:" % epoch_idx)

        preds = []
        gts = []
        for i, data in enumerate(tqdm(self.test_dataset)):
            image = torch.from_numpy(data['image']).unsqueeze(dim=0)
            gt_label = data['label']

            # Forward propagation
            logit = self.model(image.to(self.device))

            # Pixel-wise labeling
            H, W = gt_label.shape
            logit = F.interpolate(logit, size=(H, W), mode="bilinear", align_corners=True)
            prob = F.softmax(logit, dim=1)
            label = torch.argmax(prob, dim=1)

            preds.append(label[0].detach().cpu().numpy())
            gts.append(gt_label)

        # Pixel Accuracy, Mean Accuracy, Class IoU, Mean IoU, Freq Weighted IoU
        score = scores(gts, preds, n_class=self.config['model']['n_classes'])
        for k, v in score.items():
            # self.logger.info('{}: {:.5f}'.format(k, v))
            self.summary_writer.add_scalar("metric/{}".format('_'.join(k.split(' '))), v, epoch_idx)

        # self.logger.info("Validate epoch %2d done." % epoch_idx)
        # self.logger.info("-----------------------------------------------------")


class PointSegmentationTrainer(_BaseTrainer):

    def __init__(self, **config):
        super(PointSegmentationTrainer, self).__init__(**config)
        self._initialize_test_calculator()
        # 初始化匹配算子
        self.logger.info("Initialize matcher of Nearest Neighbor.")
        self.general_matcher = Matcher('float')

        if self.config['train']['train_seg']:
            self.seg_weight = 1.0
        else:
            self.seg_weight = 0

    def _initialize_dataset(self):
        # 初始化训练集
        self.logger.info('Initialize {}'.format(self.config['train']['dataset']['name']))
        self.train_dataset = get_dataset(self.config['train']['dataset']['name'])(**self.config['train']['dataset'])

        self.train_dataloader = DataLoader(
            dataset=self.train_dataset,
            batch_size=self.config['train']['batch_size'],
            shuffle=True,
            num_workers=self.config['train']['num_workers'],
            drop_last=True
        )
        self.epoch_length = len(self.train_dataset) // self.config['train']['batch_size']

        # 初始化关键点描述子测试集
        self.logger.info('Initialize test {}'.format(self.config['test']['point_dataset']['name']))
        self.point_dataset = get_dataset(self.config['test']['point_dataset']['name'])(**self.config['test']['point_dataset'])

        # 初始化分割测试集
        self.logger.info('Initialize test {}'.format(self.config['test']['seg_dataset']['name']))
        self.seg_dataset = get_dataset(self.config['test']['seg_dataset']['name'])(**self.config['test']['seg_dataset'])

    def _initialize_model(self):
        self.logger.info("Initialize network arch {}".format(self.config['model']['backbone']))
        model = get_model(self.config['model']['backbone'])(**self.config['model'])

        self.logger.info("Initialize network arch {}".format(self.config['model']['extractor']))
        extractor = get_model(self.config['model']['extractor'])()

        self.logger.info('Initialize network arch {}'.format(self.config['model']['detector']))
        detector = get_model(self.config['model']['detector'])()

        if self.multi_gpus:
            model = torch.nn.DataParallel(model)
            extractor = torch.nn.DataParallel(extractor)
            detector = torch.nn.DataParallel(detector)
        self.model = model.to(self.device)
        self.extractor = extractor.to(self.device)
        self.detector = detector.to(self.device)

        # load segmentor ckpt
        # self.logger.info('Initialize network arch {}'.format(self.config['model']['segmentor']))
        # segmentor = get_model(self.config['model']['segmentor'])()
        # self.segmentor = segmentor.to(self.device)
        # self.segmentor = self._load_model_params(self.config['model']['segmentor_ckpt'], self.segmentor)
        # self.segmentor.eval()

        # load pretrained model
        # self.model = self._load_model_params(self.config['model']['model_ckpt'], self.model)

        # load pretrained ckpt
        self.detector = self._load_model_params(self.config['model']['detector_ckpt'], self.detector)
        for p in self.detector.parameters():
            p.requires_grad = False

    def _initialize_loss(self):
        # 初始化loss算子
        self.logger.info("Initialize the CrossEntropyLoss.")
        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=255).to(self.device)

        # 初始化heatmap loss
        # self.logger.info("Initialize the L1Loss.")
        # self.point_loss = torch.nn.L1Loss()
        self.logger.info("Initialize the PointHeatmapWeightedBCELoss.")
        self.point_loss = PointHeatmapWeightedBCELoss()

        # 初始化描述子loss
        self.logger.info("Initialize the DescriptorGeneralTripletLoss.")
        self.descriptor_loss = DescriptorGeneralTripletLoss(self.device)

    def _initialize_optimizer(self):
        # 初始化网络训练优化器
        self.logger.info("Initialize Adam optimizer with weight_decay: {:.5f}.".format(self.config['train']['weight_decay']))
        self.optimizer = torch.optim.Adam(
            params=[
                {'params': self.model.parameters()},
                {'params': self.extractor.parameters()}],
            lr=self.config['train']['lr'],
            weight_decay=self.config['train']['weight_decay'])

    def _initialize_scheduler(self):
        self.logger.info('Initialize PolynomialLR')
        self.scheduler = PolynomialLR(
            optimizer=self.optimizer,
            step_size=10,
            iter_max=self.config['train']['epoch_num']*self.epoch_length,
            power=0.9,
        )

    def _train_one_epoch(self, epoch_idx):
        self.model.train()

        self.logger.info("-----------------------------------------------------")
        self.logger.info("Training epoch %2d begin:" % epoch_idx)

        self._train_func(epoch_idx)

        self.logger.info("Training epoch %2d done." % epoch_idx)
        self.logger.info("-----------------------------------------------------")

    def _train_func(self, epoch_idx):
        stime = time.time()
        for i, data in enumerate(self.train_dataloader):
            # read
            image = data["image"].to(self.device)
            label = data['label']
            desp_point = data["desp_point"].to(self.device)

            warped_image = data["warped_image"].to(self.device)
            warped_label = data['warped_label']
            warped_desp_point = data["warped_desp_point"].to(self.device)

            valid_mask = data["valid_mask"].to(self.device)
            not_search_mask = data["not_search_mask"].to(self.device)

            image_pair = torch.cat((image, warped_image), dim=0)
            label_pair = torch.cat((label, warped_label), dim=0)

            # 模型预测
            heatmap_pred_pair, c1_pair, c2_pair, c3_pair, c4_pair, seg_logits_pair = self.model(image_pair)
            teacher_heatmap_pred_pair, _, _, _, _ = self.detector(image_pair)
            teacher_heatmap = spatial_nms(torch.sigmoid(teacher_heatmap_pred_pair))
            heatmap_gt = torch.where(teacher_heatmap > 0.9, torch.ones_like(teacher_heatmap), torch.zeros_like(teacher_heatmap))

            # 计算描述子loss
            desp_point_pair = torch.cat((desp_point, warped_desp_point), dim=0)
            c1_feature_pair = F.grid_sample(c1_pair, desp_point_pair, mode="bilinear", padding_mode="border")
            c2_feature_pair = F.grid_sample(c2_pair, desp_point_pair, mode="bilinear", padding_mode="border")
            c3_feature_pair = F.grid_sample(c3_pair, desp_point_pair, mode="bilinear", padding_mode="border")
            c4_feature_pair = F.grid_sample(c4_pair, desp_point_pair, mode="bilinear", padding_mode="border")

            feature_pair = torch.cat((c1_feature_pair, c2_feature_pair, c3_feature_pair, c4_feature_pair), dim=1)
            feature_pair = feature_pair[:, :, :, 0].transpose(1, 2)
            desp_pair = self.extractor(feature_pair)
            desp_0, desp_1 = torch.chunk(desp_pair, 2, dim=0)

            desp_loss = self.descriptor_loss(desp_0, desp_1, valid_mask, not_search_mask)

            # 计算关键点loss
            detector_loss = self.point_loss(heatmap_pred_pair, heatmap_gt, torch.ones_like(heatmap_gt))

            # compute seg loss
            seg_loss = []
            for logit_pair in seg_logits_pair:
                _, _, H, W = logit_pair.shape
                label_ = resize_labels(label_pair, size=(W, H))
                seg_loss.append(self.criterion(logit_pair, label_.to(self.device)))

            seg_loss = torch.mean(torch.stack(seg_loss))

            loss = desp_loss + detector_loss + self.seg_weight * seg_loss

            if torch.isnan(loss):
                self.logger.error('loss is nan!')

            self.optimizer.zero_grad()

            loss.backward()

            self.optimizer.step()

            self.scheduler.step(epoch=i+epoch_idx*self.epoch_length)

            if i % self.config['train']['log_freq'] == 0:
                loss_val = loss.item()
                desp_val = desp_loss.item()
                detector_val = detector_loss.item()
                seg_val = seg_loss.item()

                self.summary_writer.add_scalar("loss/desp_loss", desp_val, i+epoch_idx*self.epoch_length)
                self.summary_writer.add_scalar("loss/detector_loss", detector_val, i+epoch_idx*self.epoch_length)
                self.summary_writer.add_scalar("loss/seg_loss", seg_val, i+epoch_idx*self.epoch_length)

                for k, o in enumerate(self.optimizer.param_groups):
                    self.summary_writer.add_scalar("lr/group_{}".format(k), o["lr"], i+epoch_idx*self.epoch_length)

                self.logger.info(
                    "[Epoch:%2d][Step:%5d:%5d]: point_loss=%.4f, desp_loss = %.4f, seg_loss=%.4f, one step cost %.4fs. " % (
                        epoch_idx, i, self.epoch_length,
                        detector_val, desp_val, seg_val,
                        (time.time() - stime) / self.config['train']['log_freq'],
                    ))
                stime = time.time()

        # save the model
        if self.multi_gpus:
            torch.save(self.model.module.state_dict(), os.path.join(self.config['ckpt_path'], 'model_final.pt'))
            torch.save(self.extractor.module.state_dict(), os.path.join(self.config['ckpt_path'], 'extractor_final.pt'))
        else:
            torch.save(self.model.state_dict(), os.path.join(self.config['ckpt_path'], 'model_final.pt'))
            torch.save(self.extractor.state_dict(), os.path.join(self.config['ckpt_path'], 'extractor_final.pt'))

    def _validate_one_epoch(self, epoch_idx):
        if self.config['train']['train_seg']:
            self._seg_validate_one_epoch(epoch_idx)
        self._point_validate_one_epoch(epoch_idx)

    def _seg_validate_one_epoch(self, epoch_idx):
        self.model.eval()

        preds = []
        gts = []
        for i, data in enumerate(tqdm(self.seg_dataset)):
            image = torch.from_numpy(data['image']).unsqueeze(dim=0)
            # image = torch.from_numpy(data['image2']).unsqueeze(dim=0)
            gt_label = data['label']

            # Forward propagation
            logit = self.model(image.to(self.device))[-1]
            # logit = self.segmentor(image.to(self.device))[0]

            # Pixel-wise labeling
            H, W = gt_label.shape
            logit = F.interpolate(logit, size=(H, W), mode="bilinear", align_corners=True)
            prob = F.softmax(logit, dim=1)
            label = torch.argmax(prob, dim=1)

            preds.append(label[0].detach().cpu().numpy())
            gts.append(gt_label)

        # Pixel Accuracy, Mean Accuracy, Class IoU, Mean IoU, Freq Weighted IoU
        score = scores(gts, preds, n_class=self.config['model']['n_classes'])
        for k, v in score.items():
            self.logger.info('seg {}: {:.5f}'.format(k, v))
            self.summary_writer.add_scalar("metric/{}".format('_'.join(k.split(' '))), v, epoch_idx)

    def _point_validate_one_epoch(self, epoch_idx):
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

        for i, data in tqdm(enumerate(self.point_dataset), total=len(self.point_dataset)):
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
                # print("skip this pair because there's no match point!")
                skip += 1
                continue

            # 计算得到单应变换
            pred_homography, _ = cv2.findHomography(matched_point[0][:, np.newaxis, ::-1],
                                                    matched_point[1][:, np.newaxis, ::-1], cv2.RANSAC)

            if pred_homography is None:
                # print("skip this pair because no homo can be predicted!.")
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

            # if i % 10 == 0:
            #     print("Having tested %d samples, which takes %.3fs" % (i, (time.time() - start_time)))
            #     start_time = time.time()
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

    def _inference_func(self, image_pair):
        """
        image_pair: [2,1,h,w]
        """
        self.model.eval()
        self.extractor.eval()
        _, _, height, width = image_pair.shape

        heatmap_pair, c1_pair, c2_pair, c3_pair, c4_pair, seg_pair, _ = self.model(image_pair)

        c1_0, c1_1 = torch.chunk(c1_pair, 2, dim=0)
        c2_0, c2_1 = torch.chunk(c2_pair, 2, dim=0)
        c3_0, c3_1 = torch.chunk(c3_pair, 2, dim=0)
        c4_0, c4_1 = torch.chunk(c4_pair, 2, dim=0)
        seg_0, seg_1 = torch.chunk(seg_pair, 2, dim=0)

        heatmap_pair = torch.sigmoid(heatmap_pair)
        prob_pair = spatial_nms(heatmap_pair)

        prob_pair = prob_pair.detach().cpu().numpy()
        first_prob = prob_pair[0, 0]
        second_prob = prob_pair[1, 0]

        # 得到对应的预测点
        first_point, first_point_num = self._generate_predict_point(
            first_prob,
            detection_threshold=self.config['test']['detection_threshold'],
            top_k=self.config['test']['top_k'])  # [n,2]

        second_point, second_point_num = self._generate_predict_point(
            second_prob,
            detection_threshold=self.config['test']['detection_threshold'],
            top_k=self.config['test']['top_k'])  # [n,2]

        if first_point_num <= 4 or second_point_num <= 4:
            print("skip this pair because there's little point!")
            return None

        # 得到点对应的描述子
        select_first_desp = self._generate_combined_descriptor_fast(first_point, c1_0, c2_0, c3_0, c4_0, seg_0,height, width)
        select_second_desp = self._generate_combined_descriptor_fast(second_point, c1_1, c2_1, c3_1, c4_1, seg_1,height, width)

        return first_point, first_point_num, second_point, second_point_num, select_first_desp, select_second_desp

    def _generate_combined_descriptor_fast(self, point, c1, c2, c3, c4, seg, height, width):
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
        point = point * 2. / torch.tensor((width - 1, height - 1), dtype=torch.float, device=self.device) - 1
        point = point.unsqueeze(dim=0).unsqueeze(dim=2)  # [1,n,1,2]

        c1_feature = F.grid_sample(c1, point, mode="bilinear")[:, :, :, 0].transpose(1, 2)
        c2_feature = F.grid_sample(c2, point, mode="bilinear")[:, :, :, 0].transpose(1, 2)
        c3_feature = F.grid_sample(c3, point, mode="bilinear")[:, :, :, 0].transpose(1, 2)
        c4_feature = F.grid_sample(c4, point, mode="bilinear")[:, :, :, 0].transpose(1, 2)

        feature = torch.cat((c1_feature, c2_feature, c3_feature, c4_feature), dim=2)
        desp = self.extractor(feature)[0]  # [n,128]
        if self.config['train']['train_seg']:
            seg_feature = F.grid_sample(seg, point, mode="bilinear")[0, :, :, 0].transpose(0, 1)
            seg_feature = seg_feature / torch.norm(seg_feature, 2, dim=1, keepdim=True)
            desp = torch.cat((desp, seg_feature), dim=1) / np.sqrt(2)

        desp = desp.detach().cpu().numpy()

        return desp

    def _initialize_test_calculator(self):
        height = self.config['test']['point_dataset']['height']
        width = self.config['test']['point_dataset']['width']
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


class SegmentationMixTrainer(PointSegmentationTrainer):

    def __init__(self, **config):
        super(SegmentationMixTrainer, self).__init__(**config)

    def _initialize_model(self):
        self.logger.info("Initialize network arch {}".format(self.config['model']['backbone']))
        model = get_model(self.config['model']['backbone'])(**self.config['model'])

        self.logger.info("Initialize network arch {}".format(self.config['model']['extractor']))
        extractor = get_model(self.config['model']['extractor'])()

        if self.multi_gpus:
            model = torch.nn.DataParallel(model)
            extractor = torch.nn.DataParallel(extractor)
        self.model = model.to(self.device)
        self.extractor = extractor.to(self.device)

    def _train_func(self, epoch_idx):
        stime = time.time()
        for i, data in enumerate(self.train_dataloader):
            # read
            image = data["image"].to(self.device)
            label = data['label']
            desp_point = data["desp_point"].to(self.device)
            heatmap_gt = data['heatmap'].to(self.device)
            point_mask = data['point_mask'].to(self.device)

            warped_image = data["warped_image"].to(self.device)
            warped_label = data['warped_label']
            warped_desp_point = data["warped_desp_point"].to(self.device)
            warped_heatmap_gt = data['warped_heatmap'].to(self.device)
            warped_point_mask = data['warped_point_mask'].to(self.device)

            valid_mask = data["valid_mask"].to(self.device)
            not_search_mask = data["not_search_mask"].to(self.device)

            image_pair = torch.cat((image, warped_image), dim=0)
            label_pair = torch.cat((label, warped_label), dim=0)

            # 模型预测
            heatmap_pred_pair, c1_pair, c2_pair, c3_pair, c4_pair, seg_logits_pair = self.model(image_pair)

            # 计算描述子loss
            desp_point_pair = torch.cat((desp_point, warped_desp_point), dim=0)
            c1_feature_pair = F.grid_sample(c1_pair, desp_point_pair, mode="bilinear", padding_mode="border")
            c2_feature_pair = F.grid_sample(c2_pair, desp_point_pair, mode="bilinear", padding_mode="border")
            c3_feature_pair = F.grid_sample(c3_pair, desp_point_pair, mode="bilinear", padding_mode="border")
            c4_feature_pair = F.grid_sample(c4_pair, desp_point_pair, mode="bilinear", padding_mode="border")

            feature_pair = torch.cat((c1_feature_pair, c2_feature_pair, c3_feature_pair, c4_feature_pair), dim=1)
            feature_pair = feature_pair[:, :, :, 0].transpose(1, 2)
            desp_pair = self.extractor(feature_pair)
            desp_0, desp_1 = torch.chunk(desp_pair, 2, dim=0)

            desp_loss = self.descriptor_loss(desp_0, desp_1, valid_mask, not_search_mask)

            # 计算关键点loss
            heatmap_gt_pair = torch.cat((heatmap_gt, warped_heatmap_gt), dim=0)
            point_mask_pair = torch.cat((point_mask, warped_point_mask), dim=0)
            detector_loss = self.point_loss(heatmap_pred_pair[:, 0, :, :], heatmap_gt_pair, point_mask_pair)

            # compute seg loss
            seg_loss = []
            for logit_pair in seg_logits_pair:
                _, _, H, W = logit_pair.shape
                label_ = resize_labels(label_pair, size=(W, H))
                seg_loss.append(self.criterion(logit_pair, label_.to(self.device)))

            seg_loss = torch.mean(torch.stack(seg_loss))

            loss = desp_loss + detector_loss + self.seg_weight * seg_loss

            if torch.isnan(loss):
                self.logger.error('loss is nan!')

            self.optimizer.zero_grad()

            loss.backward()

            self.optimizer.step()

            self.scheduler.step(epoch=i+epoch_idx*self.epoch_length)

            if i % self.config['train']['log_freq'] == 0:
                loss_val = loss.item()
                desp_val = desp_loss.item()
                detector_val = detector_loss.item()
                seg_val = seg_loss.item()

                self.summary_writer.add_scalar("loss/desp_loss", desp_val, i+epoch_idx*self.epoch_length)
                self.summary_writer.add_scalar("loss/detector_loss", detector_val, i+epoch_idx*self.epoch_length)
                self.summary_writer.add_scalar("loss/seg_loss", seg_val, i+epoch_idx*self.epoch_length)

                for k, o in enumerate(self.optimizer.param_groups):
                    self.summary_writer.add_scalar("lr/group_{}".format(k), o["lr"], i+epoch_idx*self.epoch_length)

                self.logger.info(
                    "[Epoch:%2d][Step:%5d:%5d]: point_loss=%.4f, desp_loss = %.4f, seg_loss=%.4f, one step cost %.4fs. " % (
                        epoch_idx, i, self.epoch_length,
                        detector_val, desp_val, seg_val,
                        (time.time() - stime) / self.config['train']['log_freq'],
                    ))
                stime = time.time()

        # save the model
        if self.multi_gpus:
            torch.save(self.model.module.state_dict(), os.path.join(self.config['ckpt_path'], 'model_final.pt'))
            torch.save(self.extractor.module.state_dict(), os.path.join(self.config['ckpt_path'], 'extractor_final.pt'))
        else:
            torch.save(self.model.state_dict(), os.path.join(self.config['ckpt_path'], 'model_final.pt'))
            torch.save(self.extractor.state_dict(), os.path.join(self.config['ckpt_path'], 'extractor_final.pt'))














