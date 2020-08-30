#
# Created by ZhangYuyang on 2020/8/30
#
import os
import time

import torch
import torch.nn.functional as f
from torch.utils.data import DataLoader

from nets import get_model
from data_utils import get_dataset
from trainers.base_trainer import BaseTrainer
from utils.utils import spatial_nms
from utils.utils import DescriptorGeneralTripletLoss
from utils.utils import PointHeatmapWeightedBCELoss


class BiSeNetV1Trainer(BaseTrainer):

    def __init__(self, **config):
        super(BiSeNetV1Trainer, self).__init__(**config)

    def _initialize_dataset(self):
        # 初始化数据集
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

    def _initialize_model(self):
        self.logger.info("Initialize network arch {}".format(self.config['model']['backbone']))
        model = get_model(self.config['model']['backbone'])()

        # load pre-trained parameters
        self.logger.info('Load pretrained parameters from {}'.format(self.config['model']['pretrained_ckpt']))
        model_dict = model.state_dict()
        pretrain_dict = torch.load(self.config['model']['pretrained_ckpt'], map_location=self.device)
        pretrain_dict = {k: v for k, v in pretrain_dict.items() if k in model_dict}
        model_dict.update(pretrain_dict)
        model.load_state_dict(model_dict)

        # freeze
        model.freeze()

        if self.multi_gpus:
            model = torch.nn.DataParallel(model)
        self.model = model.to(self.device)

    def _initialize_loss(self):
        # 初始化loss算子
        # 初始化heatmap loss
        self.logger.info("Initialize the PointHeatmapWeightedBCELoss.")
        self.point_loss = PointHeatmapWeightedBCELoss()

        # 初始化描述子loss
        self.logger.info("Initialize the DescriptorGeneralTripletLoss.")
        self.descriptor_loss = DescriptorGeneralTripletLoss(self.device)

    def _initialize_optimizer(self):
        # 只优化指定参数
        self.logger.info("Initialize Adam optimizer with weight_decay: {:.5f}.".format(self.config['train']['weight_decay']))
        self.optimizer = torch.optim.Adam(
            params=filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.config['train']['lr'],
            weight_decay=self.config['train']['weight_decay'])

    def _initialize_scheduler(self):
        # 初始化学习率调整算子
        milestones = [20, 30]
        self.logger.info("Initialize lr_scheduler of MultiStepLR: (%d, %d)" % (milestones[0], milestones[1]))
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=milestones, gamma=0.1)

    def train(self):
        start_time = time.time()

        # start training
        for i in range(self.config['train']['epoch_num']):

            # train
            self._train_one_epoch(i)
            # break  # todo

            # validation
            if i >= int(self.config['train']['epoch_num'] * self.config['train']['validate_after']):
                self._validate_one_epoch(i)

            if self.config['train']['adjust_lr']:
                # adjust learning rate
                self.scheduler.step(i)

        end_time = time.time()
        self.logger.info("The whole training process takes %.3f h" % ((end_time - start_time)/3600))

    def _train_one_epoch(self, epoch_idx):
        self.model.train()

        self.logger.info("-----------------------------------------------------")
        self.logger.info("Training epoch %2d begin:" % epoch_idx)

        self._train_func(epoch_idx)

        self.logger.info("Training epoch %2d done." % epoch_idx)
        self.logger.info("-----------------------------------------------------")

    def _train_func(self, epoch_idx):
        self.model.train()
        self.model.part_eval()

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

            if i % self.config['train']['log_freq'] == 0:

                point_loss_val = point_loss.item()
                desp_loss_val = desp_loss.item()
                loss_val = loss.item()

                self.summary_writer.add_histogram('descriptor', desp_pair)
                self.logger.info("[Epoch:%2d][Step:%5d:%5d]: loss = %.4f, point_loss = %.4f, desp_loss = %.4f"
                                 " one step cost %.4fs. "
                                 % (epoch_idx, i, self.epoch_length, loss_val,
                                    point_loss_val, desp_loss_val,
                                    (time.time() - stime) / self.config['train']['log_freq'],
                                    ))
                stime = time.time()

        # save the model
        if self.multi_gpus:
            torch.save(self.model.module.state_dict(), os.path.join(self.config['ckpt_path'], 'model_%02d.pt' % epoch_idx))
        else:
            torch.save(self.model.state_dict(), os.path.join(self.config['ckpt_path'], 'model_%02d.pt' % epoch_idx))

    def _inference_func(self, image_pair):
        """
        image_pair: [2,1,h,w]
        """
        self.model.eval()
        _, _, height, width = image_pair.shape
        heatmap_pair, desp_pair = self.model(image_pair)

        # 得到对应的关键点
        heatmap_pair = torch.sigmoid(heatmap_pair)
        prob_pair = spatial_nms(heatmap_pair)

        prob_pair = prob_pair.detach().cpu().numpy()
        first_prob = prob_pair[0, 0]
        second_prob = prob_pair[1, 0]

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
        first_desp, second_desp = torch.chunk(desp_pair, 2, dim=0)

        select_first_desp = self._generate_descriptor_for_single_head(first_point, first_desp, height, width)
        select_second_desp = self._generate_descriptor_for_single_head(second_point, second_desp, height, width)

        return first_point, first_point_num, second_point, second_point_num, select_first_desp, select_second_desp

    def _generate_descriptor_for_single_head(self, point, desp, height, width):
        """
        构建superpoint描述子端的描述子
        """
        point = torch.from_numpy(point[:, ::-1].copy()).to(torch.float).to(self.device)
        # 归一化采样坐标到[-1,1]
        point = point * 2. / torch.tensor((width - 1, height - 1), dtype=torch.float, device=self.device) - 1
        point = point.unsqueeze(dim=0).unsqueeze(dim=2)  # [1,n,1,2]

        desp = f.grid_sample(desp, point, mode="bilinear")[0, :, :, 0].transpose(0, 1)
        desp = desp / torch.norm(desp, dim=1, keepdim=True)

        desp = desp.detach().cpu().numpy()

        return desp






