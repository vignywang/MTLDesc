#
# Created by ZhangYuyang on 2020/9/14
#
import os
import time
from pathlib import Path

import torch
import numpy as np
import cv2
from tqdm import tqdm
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tensorboardX import SummaryWriter

from data_utils import get_dataset
from nets import get_model
from .utils import PolynomialLR, DepthEvaluator
from utils.utils import JointLoss


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

        self._initialize_dataset()
        self._initialize_test_dataset()
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

    def _initialize_test_dataset(self, *args, **kwargs):
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


class ValidateTrainer(_BaseTrainer):
    '''
    Used for debug
    '''
    def __init__(self, **config):
        super(ValidateTrainer, self).__init__(**config)

    def _initialize_model(self):
        self.logger.info("Initialize network arch {}".format(self.config['model']['backbone']))
        self.model_name = self.config['model']['backbone'].split('.')[-1]
        model = get_model(self.config['model']['backbone'])()

        self.model = model.to(self.device)
        self.model = self._load_model_params(self.config['model']['pretrained_ckpt'], self.model)

    def _initialize_dataset(self, *args, **kwargs):
        pass

    def _initialize_loss(self, *args, **kwargs):
        pass

    def _initialize_optimizer(self, *args, **kwargs):
        pass

    def _initialize_scheduler(self, *args, **kwargs):
        pass

    def _initialize_test_dataset(self):
        self.logger.info('Initialize test {}'.format(self.config['test']['dataset']))
        self.test_dataset = get_dataset(self.config['test']['dataset'])(**self.config['test'])

    def train(self):
        self._validate_one_epoch(0)

    def _validate_one_epoch(self, epoch_idx):
        self.model.eval()

        self.logger.info("-----------------------------------------------------")
        self.logger.info("Validate epoch %2d begin:" % epoch_idx)

        output_root = Path(self.config['test']['output_root'], self.model_name)
        output_root.mkdir(exist_ok=True, parents=True)

        evaluator = DepthEvaluator()
        for i, data in enumerate(tqdm(self.test_dataset)):
            image = torch.from_numpy(data['image']).unsqueeze(dim=0).permute((0, 3, 1, 2))
            depth_gt = data['depth']

            # Forward propagation
            log_depth_pred = self.model(image.to(self.device))
            depth_pred = torch.exp(log_depth_pred)

            # debug use
            color_image = data['color_image']
            pred_inv_depth = 1 / depth_pred[0, 0, :, :]
            pred_inv_depth = pred_inv_depth.detach().cpu().numpy()
            pred_inv_depth = pred_inv_depth / np.amax(pred_inv_depth)
            pred_inv_depth = np.tile((pred_inv_depth * 255).astype(np.uint8)[:, :, np.newaxis], (1, 1, 3))
            image_depth = np.concatenate((color_image, pred_inv_depth), axis=0)
            cv2.imwrite(os.path.join(str(output_root), '%s_%03d.jpg' % (self.model_name, i)), image_depth)

            # Pixel-wise labeling
            H, W = depth_gt.shape
            depth_pred = F.interpolate(depth_pred, size=(H, W), mode="bilinear", align_corners=True)
            depth_pred = depth_pred.detach().cpu().numpy()
            evaluator.eval(depth_pred[0, 0, :, :], depth_gt)

        errors = evaluator.val()
        for k, v in errors.items():
            self.logger.info('{}: {:.5f}'.format(k, v))
            self.summary_writer.add_scalar("metric/{}".format('_'.join(k.split(' '))), v, epoch_idx)


class DepthTrainer(_BaseTrainer):

    def __init__(self, **config):
        super(DepthTrainer, self).__init__(**config)

    def _initialize_model(self):
        self.logger.info("Initialize network arch {}".format(self.config['model']['backbone']))
        model = get_model(self.config['model']['backbone'])()

        if self.multi_gpus:
            model = torch.nn.DataParallel(model)
        self.model = model.to(self.device)

        # debug use
        # self.model = self._load_model_params(self.config['model']['pretrained_ckpt'], self.model)

    def _initialize_dataset(self):
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

    def _initialize_test_dataset(self):
        self.logger.info('Initialize test {}'.format(self.config['test']['dataset']))
        self.test_dataset = get_dataset(self.config['test']['dataset'])(**self.config['test'])

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

    def _initialize_loss(self):
        # 初始化loss算子
        self.logger.info("Initialize JointLoss.")
        self.loss = JointLoss()

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
            depth_gt = data['depth'].to(self.device)
            mask = data['mask'].to(self.device)

            # forward
            log_depth_pred = self.model(image)

            # loss
            loss, data_loss, gradient_loss = self.loss(log_depth_pred[:, 0, :, :], depth_gt, mask)

            if torch.isnan(loss):
                self.logger.error('loss is nan!')

            self.optimizer.zero_grad()

            loss.backward()

            self.optimizer.step()

            self.scheduler.step(epoch=i+epoch_idx*self.epoch_length)

            if i % self.config['train']['log_freq'] == 0:
                loss = loss.item()
                data_loss = data_loss.item()
                gradient_loss = gradient_loss.item()
                steps = i + epoch_idx * self.epoch_length

                self.summary_writer.add_scalar("loss/total_loss", loss, steps)
                self.summary_writer.add_scalar("loss/data_loss", data_loss, steps)
                self.summary_writer.add_scalar("loss/gradient_loss", gradient_loss, steps)
                for k, o in enumerate(self.optimizer.param_groups):
                    self.summary_writer.add_scalar("lr/group_{}".format(k), o["lr"], steps)

                self.logger.info(
                    "[Epoch:%2d][Step:%5d:%5d]: loss = %.4f, data_loss = %.4f, gradient_loss = %.4f, "
                    "one step cost %.4fs. " % (
                        epoch_idx, i, self.epoch_length,
                        loss, data_loss, gradient_loss,
                        (time.time() - stime) / self.config['train']['log_freq'],
                    ))
                stime = time.time()

        # save the model
        if self.multi_gpus:
            torch.save(self.model.module.state_dict(), os.path.join(self.config['ckpt_path'], 'model.pt'))
        else:
            torch.save(self.model.state_dict(), os.path.join(self.config['ckpt_path'], 'model.pt'))

    def _validate_one_epoch(self, epoch_idx):
        self.model.eval()

        self.logger.info("-----------------------------------------------------")
        self.logger.info("Validate epoch %2d begin:" % epoch_idx)

        evaluator = DepthEvaluator()
        for i, data in enumerate(tqdm(self.test_dataset)):
            image = torch.from_numpy(data['image']).unsqueeze(dim=0).permute((0, 3, 1, 2))
            depth_gt = data['depth']

            # Forward propagation
            log_depth_pred = self.model(image.to(self.device))
            depth_pred = torch.exp(log_depth_pred)

            # Pixel-wise labeling
            H, W = depth_gt.shape
            depth_pred = F.interpolate(depth_pred, size=(H, W), mode="bilinear", align_corners=True)
            depth_pred = depth_pred.detach().cpu().numpy()
            evaluator.eval(depth_pred[0, 0, :, :], depth_gt)

        errors = evaluator.val()
        for k, v in errors.items():
            self.logger.info('{}: {:.5f}'.format(k, v))
            self.summary_writer.add_scalar("metric/{}".format('_'.join(k.split(' '))), v, epoch_idx)



