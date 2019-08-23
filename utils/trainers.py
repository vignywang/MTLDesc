#
# Created by ZhangYuyang on 2019/8/12
#
import os
import torch
import time
import torch.nn.functional as f
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

from data_utils.synthetic_dataset import SyntheticTrainDataset
from data_utils.synthetic_dataset import SyntheticValTestDataset
from data_utils.coco_dataset import COCOAdaptionTrainDataset
from data_utils.coco_dataset import COCOAdaptionValDataset
from data_utils.coco_dataset import COCOSuperPointTrainDataset
from nets.superpoint_net import SuperPointNet
from utils.evaluation_tools import mAPCalculator
from utils.utils import spatial_nms
from utils.utils import DescriptorHingeLoss


# 训练算子基类
class Trainer(object):

    def __init__(self, params):
        self.params = params
        self.batch_size = params.batch_size
        self.lr = params.lr
        self.epoch_num = params.epoch_num
        self.logger = params.logger
        self.ckpt_dir = params.ckpt_dir
        self.num_workers = params.num_workers
        self.log_freq = params.log_freq
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
        self.model = SuperPointNet()
        if self.multi_gpus:
            self.model = torch.nn.DataParallel(self.model)
        self.model.to(self.device)

        # 初始化优化器算子
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        # optimizer = torch.optim.AdamW(model.parameters(), lr=self.lr)

        # 初始化学习率调整算子
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=100, gamma=0.1)

        # 以下是需要在子类中初始化的类

        # 初始化训练数据的读入接口
        self.train_dataset = None
        self.train_dataloader = None

        # 初始化验证数据的读入接口
        self.val_dataset = None

        # 初始化验证算子
        self.validator = None

    def train(self):
        start_time = time.time()

        # start training
        for i in range(self.epoch_num):

            # train
            self.train_one_epoch(i)

            # validation
            self.validate_one_epoch(i)

            # adjust learning rate
            self.scheduler.step(i)

        end_time = time.time()
        self.logger.info("The whole training process takes %.3f h" % ((end_time - start_time)/3600))

    def train_one_epoch(self, epoch_idx):
        raise NotImplementedError

    def validate_one_epoch(self, epoch_idx):
        raise NotImplementedError


class MagicPointSyntheticTrainer(object):

    def __init__(self, params):
        self.params = params
        self.batch_size = params.batch_size
        self.lr = params.lr
        self.epoch_num = params.epoch_num
        self.logger = params.logger
        self.ckpt_dir = params.ckpt_dir
        self.num_workers = params.num_workers
        self.log_freq = params.log_freq
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

        # 初始化训练数据的读入接口
        train_dataset = SyntheticTrainDataset(params)
        train_dataloader = DataLoader(train_dataset, self.batch_size, shuffle=True, num_workers=self.num_workers,
                                      drop_last=self.drop_last)

        # 初始化验证数据的读入接口
        val_dataset = SyntheticValTestDataset(params, 'validation')

        # 初始化模型
        model = SuperPointNet()
        if self.multi_gpus:
            model = torch.nn.DataParallel(model)
        model.to(self.device)

        # 初始化优化器算子
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        # optimizer = torch.optim.AdamW(model.parameters(), lr=self.lr)

        # 初始化学习率调整算子
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=60, gamma=0.1)

        # 初始化loss算子
        cross_entropy_loss = torch.nn.CrossEntropyLoss(reduction='none')

        # 初始化验证算子
        validator = mAPCalculator()

        self.train_dataloader = train_dataloader
        self.val_dataset = val_dataset
        self.epoch_length = len(train_dataset) / self.batch_size
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.cross_entropy_loss = cross_entropy_loss
        self.validator = validator

    def train(self):
        start_time = time.time()

        # start training
        for i in range(self.epoch_num):

            # train
            self.train_one_epoch(i)

            # validation
            self.validate_one_epoch(i)

            # adjust learning rate
            self.scheduler.step(i)

        end_time = time.time()
        self.logger.info("The whole training process takes %.3f h" % ((end_time - start_time)/3600))

    def train_one_epoch(self, epoch_idx):

        self.model.train()

        self.logger.info("-----------------------------------------------------")
        self.logger.info("Training epoch %2d begin:" % epoch_idx)

        stime = time.time()
        for i, data in enumerate(self.train_dataloader):
            image = data['image'].to(self.device)
            label = data['label'].to(self.device)
            mask = data['mask'].to(self.device)

            logit, _, _ = self.model(image)
            unmasked_loss = self.cross_entropy_loss(logit, label)
            loss = self._compute_masked_loss(unmasked_loss, mask)

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

    def validate_one_epoch(self, epoch_idx):

        self.model.eval()
        self.validator.reset()
        self.logger.info("*****************************************************")
        self.logger.info("Validating epoch %2d begin:" % epoch_idx)

        start_time = time.time()

        for i, data in enumerate(self.val_dataset):
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
            self.validator.update(prob, gt_point)
            if i % 10 == 0:
                print("Having validated %d samples, which takes %.3fs" % (i, (time.time()-start_time)))
                start_time = time.time()

        # 计算一个epoch的mAP值
        mAP, _, _ = self.validator.compute_mAP()
        self.summary_writer.add_scalar("mAP", mAP)

        self.logger.info("[Epoch %2d] The mean Average Precision : %.4f of %d samples" % (epoch_idx, mAP,
                                                                                          len(self.val_dataset)))
        self.logger.info("Validating epoch %2d done." % epoch_idx)
        self.logger.info("*****************************************************")

    def _compute_masked_loss(self, unmasked_loss, mask):
        total_num = torch.sum(mask, dim=(1, 2))
        loss = torch.sum(mask*unmasked_loss, dim=(1, 2)) / total_num
        loss = torch.mean(loss)
        return loss


class MagicPointAdaptionTrainer(object):

    def __init__(self, params):
        self.params = params
        self.batch_size = params.batch_size
        self.lr = params.lr
        self.epoch_num = params.epoch_num
        self.logger = params.logger
        self.ckpt_dir = params.ckpt_dir
        self.num_workers = params.num_workers
        self.log_freq = params.log_freq
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

        # 初始化训练数据的读入接口
        train_dataset = COCOAdaptionTrainDataset(params)
        train_dataloader = DataLoader(train_dataset, self.batch_size, shuffle=True, num_workers=self.num_workers,
                                      drop_last=self.drop_last)

        # 初始化验证数据的读入接口
        val_dataset = COCOAdaptionValDataset(params, add_noise=False)

        # 初始化模型
        model = SuperPointNet()
        if self.multi_gpus:
            model = torch.nn.DataParallel(model)
        model.to(self.device)

        # 初始化优化器算子
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        # optimizer = torch.optim.AdamW(model.parameters(), lr=self.lr)

        # 初始化学习率调整算子
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)

        # 初始化loss算子
        cross_entropy_loss = torch.nn.CrossEntropyLoss(reduction='none')

        # 初始化验证算子
        validator = mAPCalculator()

        self.train_dataloader = train_dataloader
        self.val_dataset = val_dataset
        self.epoch_length = len(train_dataset) / self.batch_size
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.cross_entropy_loss = cross_entropy_loss
        self.validator = validator

    def train(self):
        start_time = time.time()

        # start training
        for i in range(self.epoch_num):

            # train
            self.train_one_epoch(i)

            # validation
            self.validate_one_epoch(i)

            # adjust learning rate
            self.scheduler.step(i)

        end_time = time.time()
        self.logger.info("The whole training process takes %.3f h" % ((end_time - start_time)/3600))

    def train_one_epoch(self, epoch_idx):

        self.model.train()

        self.logger.info("-----------------------------------------------------")
        self.logger.info("Training epoch %2d begin:" % epoch_idx)

        stime = time.time()
        for i, data in enumerate(self.train_dataloader):
            image = data['image'].to(self.device)
            label = data['label'].to(self.device)
            mask = data['mask'].to(self.device)

            logit, _, _ = self.model(image)
            unmasked_loss = self.cross_entropy_loss(logit, label)
            loss = self._compute_masked_loss(unmasked_loss, mask)

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

    def validate_one_epoch(self, epoch_idx):

        self.model.eval()
        self.validator.reset()
        self.logger.info("*****************************************************")
        self.logger.info("Validating epoch %2d begin:" % epoch_idx)

        start_time = time.time()

        for i, data in enumerate(self.val_dataset):
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
            self.validator.update(prob, gt_point)
            if i % 10 == 0:
                print("Having validated %d samples, which takes %.3fs" % (i, (time.time()-start_time)))
                start_time = time.time()

        # 计算一个epoch的mAP值
        mAP, _ = self.validator.compute_mAP()
        self.summary_writer.add_scalar("mAP", mAP)

        self.logger.info("[Epoch %2d] The mean Average Precision : %.4f of %d samples" % (epoch_idx, mAP,
                                                                                          len(self.val_dataset)))
        self.logger.info("Validating epoch %2d done." % epoch_idx)
        self.logger.info("*****************************************************")

    def _compute_masked_loss(self, unmasked_loss, mask):
        total_num = torch.sum(mask, dim=(1, 2))
        loss = torch.sum(mask*unmasked_loss, dim=(1, 2)) / total_num
        loss = torch.mean(loss)
        return loss


class SuperPointTrainer(Trainer):

    def __init__(self, params):
        super(SuperPointTrainer, self).__init__(params)
        # 初始化训练数据的读入接口
        self.train_dataset = COCOSuperPointTrainDataset(params)
        self.train_dataloader = DataLoader(self.train_dataset, self.batch_size, shuffle=True,
                                           num_workers=self.num_workers, drop_last=self.drop_last)
        self.epoch_length = len(self.train_dataset) / self.batch_size

        # 初始化验证数据的读入接口
        # self.val_dataset =

        # 初始化验证算子

        # 初始化loss算子
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss(reduction='none')
        self.descriptor_loss = DescriptorHingeLoss(device=self.device)
        self.descriptor_weight = params.descriptor_weight

    def train_one_epoch(self, epoch_idx):

        self.model.train()

        self.logger.info("-----------------------------------------------------")
        self.logger.info("Training epoch %2d begin:" % epoch_idx)

        stime = time.time()
        for i, data in enumerate(self.train_dataloader):
            image = data['image'].to(self.device)
            label = data['label'].to(self.device)
            mask = data['mask'].to(self.device)
            warped_image = data['warped_image'].to(self.device)
            warped_label = data['warped_label'].to(self.device)
            warped_mask = data['warped_mask'].to(self.device)
            descriptor_mask = data['descriptor_mask'].to(self.device)
            shape = image.shape

            image_pair = torch.cat((image, warped_image), dim=0)
            label_pair = torch.cat((label, warped_label), dim=0)
            mask_pair = torch.cat((mask, warped_mask), dim=0)
            logit_pair, desp_pair, _ = self.model(image_pair)

            self.summary_writer.add_histogram('descriptor', desp_pair)

            unmasked_point_loss = self.cross_entropy_loss(logit_pair, label_pair)
            point_loss = self._compute_masked_loss(unmasked_point_loss, mask_pair)

            desp_0, desp_1 = torch.split(desp_pair, shape[0], dim=0)
            desp_loss = self.descriptor_loss(desp_0, desp_1, descriptor_mask)

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
                self.summary_writer.add_scalar('loss', loss_val)
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

    def validate_one_epoch(self, epoch_idx):

        self.model.eval()
        self.logger.info("*****************************************************")
        self.logger.info("Validating epoch %2d begin:" % epoch_idx)
        self.logger.info("TODO: There has no validating step!")
        self.logger.info("Validating epoch %2d done." % epoch_idx)
        self.logger.info("*****************************************************")

    def _compute_masked_loss(self, unmasked_loss, mask):
        total_num = torch.sum(mask, dim=(1, 2))
        loss = torch.sum(mask*unmasked_loss, dim=(1, 2)) / total_num
        loss = torch.mean(loss)
        return loss



