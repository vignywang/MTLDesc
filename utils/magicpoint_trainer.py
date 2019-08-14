#
# Created by ZhangYuyang on 2019/8/12
#
import os
import torch
import time
import numpy as np
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

from data_utils.synthetic_dataset import SyntheticTrainDataset
from data_utils.synthetic_dataset import SyntheticValTestDataset
from nets.superpoint_net import SuperPointNet
from utils.evaluation_tools import mAPCalculator


class MagicPointTrainer(object):

    def __init__(self, params):
        self.params = params
        self.batch_size = params.batch_size
        self.lr = params.lr
        self.epoch_num = params.epoch_num
        self.logger = params.logger
        self.ckpt_dir = params.ckpt_dir
        self.num_workers = params.num_workers
        if torch.cuda.is_available():
            self.logger.info('gpu is available, set device to cuda !')
            self.device = torch.device('cuda:0')
        else:
            self.logger.info('gpu is not available, set device to cpu !')
            self.device = torch.device('cpu')
        self.multi_gpus = False
        if torch.cuda.device_count() > 1:
            self.batch_size *= torch.cuda.device_count()
            self.multi_gpus = True
            self.logger.info("Multi gpus is available, let's use %d GPUS" % torch.cuda.device_count())

        # 初始化summary writer
        self.summary_writer = SummaryWriter(self.ckpt_dir)

        # 初始化训练数据的读入接口
        train_dataset = SyntheticTrainDataset(params)
        train_dataloader = DataLoader(train_dataset, self.batch_size, shuffle=True, num_workers=8)

        # 初始化验证数据的读入接口
        val_dataset = SyntheticValTestDataset(params, 'validation')

        # 初始化模型
        model = SuperPointNet()
        if self.multi_gpus:
            model = torch.nn.DataParallel(model)
        model.to(self.device)

        # 初始化优化器算子
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)

        # 初始化loss算子
        cross_entropy_loss = torch.nn.CrossEntropyLoss(reduction='none')

        # 初始化验证算子
        validator = mAPCalculator()

        self.train_dataloader = train_dataloader
        self.val_dataset = val_dataset
        self.epoch_length = len(train_dataset) / self.batch_size
        self.model = model
        self.optimizer = optimizer
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

            # if i % self.params.sum_freq == 0:
            #     self.summary_writer.add_histogram("loss/positive", positive, global_step=i)
            #     self.summary_writer.add_histogram("loss/negtive", negtive, global_step=i)

            if torch.isnan(loss):
                self.logger.error('loss is nan!')

            self.optimizer.zero_grad()
            loss.backward()

            self.optimizer.step()

            if i % self.params.log_freq == 0:
                loss_val = loss.item()
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

            image = image.to(self.device).unsqueeze(dim=0)
            # 得到原始的经压缩的概率图，概率图每个通道64维，对应空间每个像素是否为关键点的概率
            _, _, prob = self.model(image)
            prob = prob.detach().cpu().numpy()[0]
            gt_point = gt_point.numpy()
            # 将概率图展开为原始图像大小
            prob = np.transpose(prob, (1, 2, 0))
            prob = np.reshape(prob, (30, 40, 8, 8))
            prob = np.transpose(prob, (0, 2, 1, 3))
            prob = np.reshape(prob, (240, 320))

            self.validator.update(prob, gt_point)
            if i % 10 == 0:
                print("Having validated %d samples, which takes %.3fs" % (i, (time.time()-start_time)))
                start_time = time.time()

        # 计算一个epoch的mAP值
        mAP, _, _ = self.validator.compute_mAP()

        self.logger.info("[Epoch %2d] The mean Average Precision : %.4f of %d samples" % (epoch_idx, mAP,
                                                                                          len(self.val_dataset)))
        self.logger.info("Validating epoch %2d done." % epoch_idx)
        self.logger.info("*****************************************************")

    def _compute_masked_loss(self, unmasked_loss, mask):
        total_num = torch.sum(mask, dim=(1, 2))
        loss = torch.sum(mask*unmasked_loss, dim=(1, 2)) / total_num
        loss = torch.mean(loss)
        return loss















