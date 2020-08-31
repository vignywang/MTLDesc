#
# Created by ZhangYuyang on 2020/8/31
#
import os
import time

import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tensorboardX import SummaryWriter

from data_utils import get_dataset
from .utils import resize_labels
from .utils import PolynomialLR
from .utils import scores
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

    def train(self):
        start_time = time.time()

        # start training
        for i in range(self.config['train']['epoch_num']):

            # train
            self._train_one_epoch(i)

            # todo: validation
            # if i >= int(self.config['train']['epoch_num'] * 2/ 3):
            self._validate_one_epoch(i)
            pass

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







