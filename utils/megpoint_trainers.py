# 
# Created by ZhangYuyang on 2019/9/18
#
# 训练算子基类
import os
import time

import torch
import torch.nn.functional as f
import numpy as np
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader

from nets.megpoint_net import AdaptionDataset
from nets.megpoint_net import MegPointNet
from nets.megpoint_net import LabelGenerator
from utils.utils import spatial_nms
from data_utils.coco_dataset import COCOMegPointAdaptionDataset
from data_utils.hpatch_dataset import HPatchDataset
from utils.evaluation_tools import RepeatabilityCalculator
from utils.evaluation_tools import MovingAverage
from utils.evaluation_tools import PointStatistics
from data_utils.dataset_tools import HomographyAugmentation
from data_utils.dataset_tools import debug_draw_image_keypoints


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
        self.train_top_k = params.train_top_k
        self.nms_threshold = params.nms_threshold
        self.detection_threshold = params.detection_threshold
        self.correct_epsilon = params.correct_epsilon
        if torch.cuda.is_available():
            self.logger.info('gpu is available, set device to cuda !')
            self.device = torch.device('cuda:0')
        else:
            self.logger.info('gpu is not available, set device to cpu !')
            self.device = torch.device('cpu')
        self.multi_gpus = False
        self.drop_last = False
        if torch.cuda.device_count() > 1:
            count = torch.cuda.device_count()
            self.batch_size *= count
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
            self.scheduler.step(i)

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
        model_dict = self.model.base_megpoint.state_dict()
        pretrain_dict = torch.load(ckpt_file, map_location=self.device)
        model_dict.update(pretrain_dict)
        self.model.base_megpoint.load_state_dict(model_dict)

        return True


class MegPointAdaptionTrainer(MegPointTrainerTester):

    def __init__(self, params):
        super(MegPointAdaptionTrainer, self).__init__(params)
        # 初始化训练数据的读入接口
        self.train_dataset = COCOMegPointAdaptionDataset(params)
        self.adaption_dataset = AdaptionDataset()
        # self.train_dataset = COCOAdaptionDataset(params, 'train2014')
        self.raw_dataloader = DataLoader(self.train_dataset, self.batch_size, shuffle=False,
                                         num_workers=self.num_workers, drop_last=True)
        self.epoch_length = len(self.train_dataset) / self.batch_size

        self.test_dataset = HPatchDataset(params)
        self.test_length = len(self.test_dataset)

        # 初始化验证算子
        self.logger.info('Initialize the repeatability calculator, detection_threshold: %.4f, correct_epsilon: %d'
                         % (self.detection_threshold, self.correct_epsilon))
        self.logger.info('Top k: %d' % self.top_k)

        self.illum_repeat = RepeatabilityCalculator(params.correct_epsilon)
        self.illum_repeat_mov = MovingAverage()

        self.view_repeat = RepeatabilityCalculator(params.correct_epsilon)
        self.view_repeat_mov = MovingAverage()

        self.point_statistics = PointStatistics()

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
        # loss_weight = torch.ones(64, dtype=torch.float, device=self.device)  # *100
        # loss_weight = torch.cat((loss_weight, torch.ones((1,), device=self.device)), dim=0)
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss(reduction='none')

        # 若有多gpu设置则加载多gpu
        if self.multi_gpus:
            self.model = torch.nn.DataParallel(self.model)
            self.label_generator = torch.nn.DataParallel(self.label_generator)
        self.model = self.model.to(self.device)
        self.label_generator = self.label_generator.to(self.device)

        # 初始化优化器算子
        # self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr)

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
            model_dict = self.model.module.base_megpoint.state_dict()
            generator_model_dict = self.label_generator.module.base_megpoint.state_dict()
            generator_model_dict.update(model_dict)
            self.label_generator.module.base_megpoint.load_state_dict(generator_model_dict)
        else:
            model_dict = self.model.base_megpoint.state_dict()
            label_generator_model_dict = self.label_generator.base_megpoint.state_dict()
            label_generator_model_dict.update(model_dict)
            self.label_generator.base_megpoint.load_state_dict(label_generator_model_dict)

        if epoch_idx == 0:
            detection_threshold = self.detection_threshold
            self.logger.info("Labeling, detection_threshold=%.4f" % detection_threshold)
        else:
            detection_threshold = self.detection_threshold*2.0
            self.logger.info("Labeling, detection_threshold=%.4f" % detection_threshold)

        start_time = time.time()
        stime = time.time()
        count = 0
        self.logger.info("Relabeling current dataset")
        for i, data in enumerate(self.raw_dataloader):
            image = data["image"].to(self.device)

            # 采样构成标签需要的单应变换
            sampled_homo, sampled_inv_homo = self._sample_homography(self.batch_size)
            sampled_homo = sampled_homo.to(self.device)
            sampled_inv_homo = sampled_inv_homo.to(self.device)

            image, point, point_mask = self.label_generator(image, sampled_homo, sampled_inv_homo, detection_threshold)
            self.adaption_dataset.append(image, point, point_mask)
            count += 1
            if i % self.log_freq == 0:
                self.logger.info("[Epoch:%2d][Labeling Step:%5d:%5d],"
                                 " one step cost %.4fs. "
                                 % (epoch_idx, i, self.epoch_length,
                                    (time.time() - stime) / self.log_freq,
                                    ))
                stime = time.time()

            if i == 200:
                print("Debug use, only to adaption 200 batches")
                break

        self.logger.info("Relabeling Done. Totally %d batched sample. Takes %.3fs" % (count, (time.time()-start_time)))

    def _train_one_epoch(self, epoch_idx):
        self.logger.info("-----------------------------------------------------")
        self.logger.info("Start to train epoch %2d:" % epoch_idx)
        stime = time.time()

        self._adaption_data(epoch_idx)
        if len(self.adaption_dataset) == 0:
            assert False

        self.model.train()
        for i, data in enumerate(self.adaption_dataset):
            image = data['image'].to(self.device)  # [bt,1,240,320]
            label = data["sparse_label"].to(self.device)
            mask = data["sparse_mask"].to(self.device)
            prob = data["prob"].to(self.device)

            logit, _ = self.model(image)
            unmasked_loss = self.cross_entropy_loss(logit, label)
            loss = self._compute_masked_loss(unmasked_loss, mask)

            if torch.isnan(loss):
                self.logger.error('loss is nan!')

            self.optimizer.zero_grad()
            loss.backward()

            self.optimizer.step()

            if i % self.log_freq == 0:
                step = int(i+epoch_idx*self.batch_size)
                debug_image = ((image+1.)*255./2.).detach().cpu().numpy()
                debug_prob = prob.detach().cpu().numpy()
                image_points = debug_draw_image_keypoints(debug_image, debug_prob)
                loss_val = loss.item()
                self.summary_writer.add_image("image_points", image_points, step)
                self.summary_writer.add_scalar('loss', loss, step)
                self.summary_writer.add_histogram("logit", logit, step)
                if not self.multi_gpus:
                    self.summary_writer.add_histogram(
                        "convPb.weight", self.model.base_megpoint.state_dict()["convPb.weight"], step)
                else:
                    self.summary_writer.add_histogram(
                        "convPb.weight", self.model.module.base_megpoint.state_dict()["convPb.weight"], step)
                self.logger.info("[Epoch:%2d][Step:%5d:%5d]: loss = %.4f,"
                                 " one step cost %.4fs. "
                                 % (epoch_idx, i, self.epoch_length, loss_val,
                                    (time.time() - stime) / self.log_freq,
                                    ))
                stime = time.time()

        # save the model
        if self.multi_gpus:
            torch.save(self.model.module.base_megpoint.state_dict(), os.path.join(self.ckpt_dir, 'model_%02d.pt' % epoch_idx))
        else:
            torch.save(self.model.base_megpoint.state_dict(), os.path.join(self.ckpt_dir, 'model_%02d.pt' % epoch_idx))

        self.logger.info("Training epoch %2d done." % epoch_idx)
        self.logger.info("-----------------------------------------------------")

    def _validate_one_epoch(self, epoch_idx):
        self.logger.info("*****************************************************")
        self.logger.info("Validating epoch %2d begin:" % epoch_idx)

        self._test_model_general(epoch_idx)

        illum_repeat_moving_acc = self.illum_repeat_mov.average()
        view_repeat_moving_acc = self.view_repeat_mov.average()

        current_size = self.illum_repeat_mov.current_size()

        self.logger.info("---------------------------------------------")
        self.logger.info("Moving Average of %d models:" % current_size)
        self.logger.info("illum_repeat_moving_acc=%.4f, view_repeat_moving_acc=%.4f" %
                         (illum_repeat_moving_acc, view_repeat_moving_acc))
        self.logger.info("---------------------------------------------")
        self.logger.info("Validating epoch %2d done." % epoch_idx)
        self.logger.info("*****************************************************")

    def _test_model_general(self, epoch_idx):

        self.model.eval()
        # 重置测评算子参数
        self.illum_repeat.reset()
        self.view_repeat.reset()
        self.point_statistics.reset()

        start_time = time.time()
        count = 0
        skip = 0

        if epoch_idx == 0:
            detection_threshold = self.detection_threshold
            self.logger.info("Test, detection_threshold=%.4f" % detection_threshold)
        else:
            detection_threshold = self.detection_threshold * 2.0
            self.logger.info("Test, detection_threshold=%.4f" % detection_threshold)

        for i, data in enumerate(self.test_dataset):
            first_image = data['first_image']
            second_image = data['second_image']
            gt_homography = data['gt_homography']
            image_type = data['image_type']

            image_pair = np.stack((first_image, second_image), axis=0)
            image_pair = torch.from_numpy(image_pair).to(torch.float).to(self.device).unsqueeze(dim=1)
            image_pair = image_pair*2/255. - 1.
            # debug released mode use
            # image_pair /= 255.

            _, prob_pair = self.model(image_pair)
            prob_pair = f.pixel_shuffle(prob_pair, 8)
            prob_pair = spatial_nms(prob_pair, kernel_size=int(self.nms_threshold*2+1))

            prob_pair = prob_pair.detach().cpu().numpy()
            first_prob = prob_pair[0, 0]
            second_prob = prob_pair[1, 0]

            # 得到对应的预测点
            first_point, first_point_num = self._generate_predict_point(
                first_prob, detection_threshold, top_k=self.top_k)  # [n,2]
            second_point, second_point_num = self._generate_predict_point(
                second_prob, detection_threshold, top_k=self.top_k)  # [m,2]

            if first_point_num == 0 or second_point_num == 0:
                skip += 1
                continue
            # 对单样本进行测评
            if image_type == 'illumination':
                self.illum_repeat.update(first_point, second_point, gt_homography)
            elif image_type == 'viewpoint':
                self.view_repeat.update(first_point, second_point, gt_homography)
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

        # 统计最终的检测点数目的平均值和方差
        point_avg, point_std = self.point_statistics.average()

        self.logger.info("Skip point num: %d" % skip)
        self.logger.info("Repeatability: illumination: %.4f, viewpoint: %.4f, total: %.4f" %
                         (illum_repeat, view_repeat, total_repeat))
        self.logger.info("Detection point, average: %.4f, variance: %.4f" % (point_avg, point_std))

        # 将测试结果写入summaryWriter中，方便在tensorboard中查看
        self.summary_writer.add_scalar("illumination/Repeatability", illum_repeat, epoch_idx)
        self.summary_writer.add_scalar("viewpoint/Repeatability", view_repeat, epoch_idx)

    @staticmethod
    def _compute_total_metric(illum_metric, view_metric):
        illum_acc, illum_sum, illum_num = illum_metric.average()
        view_acc, view_sum, view_num = view_metric.average()
        return illum_acc, view_acc, (illum_sum+view_sum)/(illum_num+view_num+1e-4)

    def _generate_predict_point(self, prob, detection_threshold, scale=None, top_k=0):
        point_idx = np.where(prob > detection_threshold)
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

    def _sample_aug_homography(self, batch_size):
        sampled_homo = []
        for i in range(batch_size):
            homo = self.homography_sampler.sample()
            sampled_homo.append(homo)
        sampled_homo = torch.from_numpy(np.stack(sampled_homo, axis=0)).to(torch.float)
        return sampled_homo











