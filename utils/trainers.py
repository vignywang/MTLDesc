#
# Created by ZhangYuyang on 2019/8/12
#
import os
import time
import torch
import torch.nn.functional as f
import numpy as np
import cv2 as cv
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

from data_utils.synthetic_dataset import SyntheticTrainDataset
from data_utils.synthetic_dataset import SyntheticValTestDataset
from data_utils.coco_dataset import COCOAdaptionTrainDataset
from data_utils.coco_dataset import COCOAdaptionValDataset
from data_utils.coco_dataset import COCOSuperPointTrainDataset
from data_utils.hpatch_dataset import HPatchDataset
from nets.superpoint_net import SuperPointNet
from utils.evaluation_tools import mAPCalculator
from utils.evaluation_tools import HomoAccuracyCalculator
from utils.evaluation_tools import RepeatabilityCalculator
from utils.evaluation_tools import MeanMatchingAccuracy
from utils.evaluation_tools import PointStatistics
from utils.evaluation_tools import MovingAverage
from utils.utils import spatial_nms, Matcher
from utils.utils import DescriptorHingeLoss
from utils.utils import DescriptorTripletLoss
from utils.utils import BinaryDescriptorTripletLoss


# 训练算子基类
class TrainerTester(object):

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
        self.output_type = params.output_type
        if self.output_type == 'float':
            self.model = SuperPointNet()
        elif self.output_type == 'binary':
            self.model = SuperPointNet('binary')
        else:
            self.logger.error('Incorrect output_type: %s' % self.output_type)

        if self.multi_gpus:
            self.model = torch.nn.DataParallel(self.model)
        self.model.to(self.device)

        # 初始化优化器算子
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        # optimizer = torch.optim.AdamW(model.parameters(), lr=self.lr)

        # 初始化学习率调整算子
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=100, gamma=0.1)

    def train(self):
        start_time = time.time()

        # start training
        for i in range(self.epoch_num):

            # train
            self._train_one_epoch(i)

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
        model_dict = self.model.state_dict()
        pretrain_dict = torch.load(ckpt_file, map_location=self.device)
        model_dict.update(pretrain_dict)
        self.model.load_state_dict(model_dict)

        return True

    @staticmethod
    def _compute_masked_loss(unmasked_loss, mask):
        total_num = torch.sum(mask, dim=(1, 2))
        loss = torch.sum(mask*unmasked_loss, dim=(1, 2)) / total_num
        loss = torch.mean(loss)
        return loss


class MagicPointSynthetic(TrainerTester):

    def __init__(self, params):
        super(MagicPointSynthetic, self).__init__(params)

        self.save_threshold_curve = True

        self.train_dataset, self.val_dataset, self.test_dataset = self._initialize_dataset()
        self.train_dataloader = DataLoader(self.train_dataset, self.batch_size, shuffle=True,
                                           num_workers=self.num_workers, drop_last=self.drop_last)
        self.epoch_length = len(self.train_dataset) / self.batch_size

        # 初始化loss算子
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss(reduction='none')

        # 初始化验证算子
        self.mAP_calculator = mAPCalculator()

    def test(self, ckpt_file):
        self.logger.info("*****************************************************")
        self.logger.info("Testing model %s" % ckpt_file)

        # 从预训练的模型中恢复参数
        if not self._load_model_params(ckpt_file):
            self.logger.error('Can not load model!')
            return

        curve_name = None
        curve_dir = None
        if self.save_threshold_curve:
            save_root = '/'.join(ckpt_file.split('/')[:-1])
            curve_name = (ckpt_file.split('/')[-1]).split('.')[0]
            curve_dir = os.path.join(save_root, curve_name + '.png')

        mAP, test_data, count = self._test_func(self.test_dataset)

        if self.save_threshold_curve:
            self.mAP_calculator.plot_threshold_curve(test_data, curve_name, curve_dir)

        self.logger.info("The mean Average Precision : %.4f of %d samples" % (mAP, count))
        self.logger.info("Testing done.")
        self.logger.info("*****************************************************")

    def test_single_image(self, ckpt_file, image_dir):
        self.logger.info("*****************************************************")
        self.logger.info("Testing image %s" % image_dir)
        self.logger.info("From model %s" % ckpt_file)
        # 从预训练的模型中恢复参数
        if not self._load_model_params(ckpt_file):
            self.logger.error('Can not load model!')
            return

        self.model.eval()

        cv_image = cv.imread(image_dir, cv.IMREAD_GRAYSCALE)
        image = np.expand_dims(np.expand_dims(cv_image, 0), 0)
        image = torch.from_numpy(image).to(torch.float).to(self.device)

        _, _, prob = self.model(image)
        prob = f.pixel_shuffle(prob, 8)
        # 进行非极大值抑制
        prob = spatial_nms(prob)
        prob = prob.detach().cpu().numpy()[0, 0]

        pred_pt = np.where(prob>0.1)

        cv_pt_list = []
        for i in range(len(pred_pt[0])):
            kpt = cv.KeyPoint()
            kpt.pt = (pred_pt[1][i], pred_pt[0][i])
            cv_pt_list.append(kpt)

        result_dir = os.path.join('/'.join(ckpt_file.split('/')[:-1]), 'test_image.jpg')
        cv_image = cv.drawKeypoints(cv_image, cv_pt_list, None, color=(0, 0, 255))
        cv.imwrite(result_dir, cv_image)
        self.logger.info("Result dir: %s" % result_dir)
        self.logger.info("*****************************************************")

    def _initialize_dataset(self):
        train_dataset = SyntheticTrainDataset(self.params)
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

    def _validate_one_epoch(self, epoch_idx):
        self.logger.info("*****************************************************")
        self.logger.info("Validating epoch %2d begin:" % epoch_idx)

        mAP, _, count = self._test_func(self.val_dataset)
        self.summary_writer.add_scalar("mAP", mAP)

        self.logger.info("[Epoch %2d] The mean Average Precision : %.4f of %d samples" % (epoch_idx, mAP,
                                                                                          count))
        self.logger.info("Validating epoch %2d done." % epoch_idx)
        self.logger.info("*****************************************************")

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
            _, _, prob = self.model(image)
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


class MagicPointAdaption(MagicPointSynthetic):

    def __init__(self, params):
        super(MagicPointAdaption, self).__init__(params)

    def _initialize_dataset(self):
        self.logger.info('Initialize COCO Adaption Dataset %s' % self.params.coco_pseudo_idx)
        train_dataset = COCOAdaptionTrainDataset(self.params)
        val_dataset = COCOAdaptionValDataset(self.params, add_noise=False)
        return train_dataset, val_dataset, None


class SuperPoint(TrainerTester):

    def __init__(self, params):
        super(SuperPoint, self).__init__(params)
        # 初始化训练数据的读入接口
        self.train_dataset = COCOSuperPointTrainDataset(params)
        self.train_dataloader = DataLoader(self.train_dataset, self.batch_size, shuffle=True,
                                           num_workers=self.num_workers, drop_last=self.drop_last)
        self.epoch_length = len(self.train_dataset) / self.batch_size

        self.test_dataset = HPatchDataset(params)
        self.test_length = len(self.test_dataset)

        # 初始化验证算子
        self.logger.info('Initialize the homography accuracy calculator, correct_epsilon: %d' % self.correct_epsilon)
        self.logger.info('Initialize the repeatability calculator, detection_threshold: %.4f, correct_epsilon: %d'
                         % (self.detection_threshold, self.correct_epsilon))
        self.logger.info('Top k: %d' % self.top_k)

        self.illumination_repeatability = RepeatabilityCalculator(params.correct_epsilon)
        self.illumination_repeatability_mov = MovingAverage()
        self.illumination_homo_accuracy = HomoAccuracyCalculator(params.correct_epsilon,
                                                                 params.hpatch_height, params.hpatch_width)
        self.illumination_homo_accuracy_mov = MovingAverage()
        self.illumination_mma = MeanMatchingAccuracy(params.correct_epsilon)
        self.illumination_mma_mov = MovingAverage()

        self.viewpoint_repeatability = RepeatabilityCalculator(params.correct_epsilon)
        self.viewpoint_repeatability_mov = MovingAverage()
        self.viewpoint_homo_accuracy = HomoAccuracyCalculator(params.correct_epsilon,
                                                              params.hpatch_height, params.hpatch_width)
        self.viewpoint_homo_accuracy_mov = MovingAverage()
        self.viewpoint_mma = MeanMatchingAccuracy(params.correct_epsilon)
        self.viewpoint_mma_mov = MovingAverage()

        self.point_statistics = PointStatistics()

        self.matcher = Matcher()

        # 得到指定的loss构造类型
        self.loss_type = params.loss_type
        self.logger.info('The loss type: %s' % self.loss_type)

        self.output_type = params.output_type
        self.logger.info('The outout type: %s' % self.output_type)

        # 根据输出类型和loss类型初始化相应算子和函数
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss(reduction='none')
        self.descriptor_weight = params.descriptor_weight
        self.quantization_weight = params.quantization_weight
        # self.descriptor_weight = 0.
        if self.output_type == 'float':
            if self.loss_type == 'pairwise':
                self.descriptor_loss = DescriptorHingeLoss(device=self.device)
                self._train_func = self._train_use_pairwise_loss
            elif self.loss_type == 'triplet':
                self.descriptor_loss = DescriptorTripletLoss(device=self.device)
                self._train_func = self._train_use_triplet_loss
            else:
                self.logger.error('incorrect loss type: %s' % self.loss_type)
                assert False
        elif self.output_type == 'binary':
            self.descriptor_loss = BinaryDescriptorTripletLoss()
            self._train_func = self._train_use_triplet_loss_binary
        else:
            self.logger.error('incorrect output type: %s' % self.output_type)
            assert False

        # debug use
        # ckpt_file = "/home/zhangyuyang/project/development/MegPoint/superpoint_ckpt/good_results/superpoint_triplet_0.0010_24/model_49.pt"
        # self._load_model_params(ckpt_file)

    def test_HPatch(self, ckpt_file):
        self.logger.info("*****************************************************")
        self.logger.info("Testing model %s in HPatch" % ckpt_file)
        self.logger.info("The detection threshold: %.4f" % self.detection_threshold)
        self.logger.info("The correct thereshold: %d" % self.correct_epsilon)
        self.logger.info("The top k: %d" % self.top_k)

        self._load_model_params(ckpt_file)
        self._test_model()

        self.logger.info("Testing HPatch done.")
        self.logger.info("*****************************************************")

    def _train_one_epoch(self, epoch_idx):
        self._train_func(epoch_idx)

    def _train_use_triplet_loss_binary(self, epoch_idx):
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

            matched_idx = data['matched_idx'].to(self.device)
            matched_valid = data['matched_valid'].to(self.device)
            not_search_mask = data['not_search_mask'].to(self.device)
            desp_1_valid_mask = data['warped_valid_mask'].to(self.device)

            shape = image.shape

            image_pair = torch.cat((image, warped_image), dim=0)
            label_pair = torch.cat((label, warped_label), dim=0)
            mask_pair = torch.cat((mask, warped_mask), dim=0)
            logit_pair, desp_pair, _ = self.model(image_pair)

            unmasked_point_loss = self.cross_entropy_loss(logit_pair, label_pair)
            point_loss = self._compute_masked_loss(unmasked_point_loss, mask_pair)

            desp_0, desp_1 = torch.split(desp_pair, shape[0], dim=0)
            desp_loss, quantization_loss = self.descriptor_loss(
                desp_0, desp_1, matched_idx, matched_valid, not_search_mask, desp_1_valid_mask)

            loss = point_loss + self.descriptor_weight*desp_loss + self.quantization_weight*quantization_loss

            if torch.isnan(loss):
                self.logger.error('loss is nan!')

            self.optimizer.zero_grad()
            loss.backward()

            self.optimizer.step()

            loss_val = loss.item()
            point_loss_val = point_loss.item()
            desp_loss_val = desp_loss.item()
            quantization_loss_val = quantization_loss.item()

            self.summary_writer.add_scalar('loss/total_loss', loss_val,
                                           global_step=int(i + epoch_idx * self.epoch_length))
            self.summary_writer.add_scalar('loss/point_loss', point_loss,
                                           global_step=int(i + epoch_idx * self.epoch_length))
            self.summary_writer.add_scalar('loss/desp_loss', desp_loss_val,
                                           global_step=int(i + epoch_idx * self.epoch_length))
            self.summary_writer.add_scalar('loss/quantization_loss', quantization_loss_val,
                                           global_step=int(i + epoch_idx * self.epoch_length))

            if i % self.log_freq == 0:

                self.summary_writer.add_histogram('descriptor', desp_pair,
                                                  global_step=int(i+epoch_idx*self.epoch_length))

                self.logger.info("[Epoch:%2d][Step:%5d:%5d]: "
                                 "loss = %.4f, "
                                 "point_loss = %.4f, "
                                 "desp_loss = %.4f, "
                                 "quantization_loss = %.4f, "
                                 " one step cost %.4fs. "
                                 % (epoch_idx, i, self.epoch_length,
                                    loss_val,
                                    point_loss_val,
                                    desp_loss_val,
                                    quantization_loss_val,
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

    def _train_use_triplet_loss(self, epoch_idx):

        self.model.train()

        self.logger.info("-----------------------------------------------------")
        self.logger.info("Training epoch %2d begin:" % epoch_idx)

        debug_use = False

        stime = time.time()
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

            # debug use
            warped_grid = data['warped_grid'].to(self.device)
            matched_grid = data['matched_grid'].to(self.device)

            shape = image.shape

            image_pair = torch.cat((image, warped_image), dim=0)
            label_pair = torch.cat((label, warped_label), dim=0)
            mask_pair = torch.cat((mask, warped_mask), dim=0)
            logit_pair, desp_pair, _ = self.model(image_pair)

            unmasked_point_loss = self.cross_entropy_loss(logit_pair, label_pair)
            point_loss = self._compute_masked_loss(unmasked_point_loss, mask_pair)

            desp_0, desp_1 = torch.split(desp_pair, shape[0], dim=0)

            if debug_use:
                desp_loss, positive_dist, negative_dist = self.descriptor_loss(
                    desp_0, desp_1, matched_idx, matched_valid, not_search_mask, warped_grid, matched_grid,
                    debug_use=True)
            else:
                desp_loss = self.descriptor_loss(
                    desp_0, desp_1, matched_idx, matched_valid, not_search_mask, warped_grid, matched_grid)

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

                if debug_use:
                    for name, params in self.model.named_parameters():
                        name_p = name.split('.')[0]
                        if name_p in ['convDa', 'convDb']:
                            self.summary_writer.add_histogram('params/'+name, params)
                            self.summary_writer.add_histogram('grad/'+name, params.grad)
                    self.summary_writer.add_scalar(
                        'dist/positive_dist', positive_dist, int(i+self.epoch_length*epoch_idx))
                    self.summary_writer.add_scalar(
                        'dist/negative_dist', negative_dist, int(i+self.epoch_length*epoch_idx))

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

    def _train_use_pairwise_loss(self, epoch_idx):

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
            valid_mask = data['warped_valid_mask'].to(self.device)

            shape = image.shape

            image_pair = torch.cat((image, warped_image), dim=0)
            label_pair = torch.cat((label, warped_label), dim=0)
            mask_pair = torch.cat((mask, warped_mask), dim=0)
            logit_pair, desp_pair, _ = self.model(image_pair)

            unmasked_point_loss = self.cross_entropy_loss(logit_pair, label_pair)
            point_loss = self._compute_masked_loss(unmasked_point_loss, mask_pair)

            desp_0, desp_1 = torch.split(desp_pair, shape[0], dim=0)
            desp_loss = self.descriptor_loss(desp_0, desp_1, descriptor_mask, valid_mask)

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

    def _validate_one_epoch(self, epoch_idx):
        self.logger.info("*****************************************************")
        self.logger.info("Validating epoch %2d begin:" % epoch_idx)

        self._test_model()

        illum_homo_moving_acc = self.illumination_homo_accuracy_mov.average()
        view_homo_moving_acc = self.viewpoint_homo_accuracy_mov.average()

        illum_mma_moving_acc = self.illumination_mma_mov.average()
        view_mma_moving_acc = self.viewpoint_mma_mov.average()

        illum_repeat_moving_acc = self.illumination_repeatability_mov.average()
        view_repeat_moving_acc = self.viewpoint_repeatability_mov.average()

        current_size = self.viewpoint_mma_mov.current_size()

        self.logger.info("Moving Average of %d models:" % current_size)
        self.logger.info("illum_homo_moving_acc=%.4f, view_homo_moving_acc=%.4f" %
                         (illum_homo_moving_acc, view_homo_moving_acc))
        self.logger.info("illum_mma_moving_acc=%.4f, view_mma_moving_acc=%.4f" %
                         (illum_mma_moving_acc, view_mma_moving_acc))
        self.logger.info("illum_repeat_moving_acc=%.4f, view_repeat_moving_acc=%.4f" %
                         (illum_repeat_moving_acc, view_repeat_moving_acc))
        self.logger.info("Validating epoch %2d done." % epoch_idx)
        self.logger.info("*****************************************************")

    def _test_model(self):

        self.model.eval()
        # 重置测评算子参数
        self.illumination_repeatability.reset()
        self.illumination_homo_accuracy.reset()
        self.illumination_mma.reset()
        self.viewpoint_repeatability.reset()
        self.viewpoint_homo_accuracy.reset()
        self.viewpoint_mma.reset()
        self.point_statistics.reset()

        start_time = time.time()
        count = 0

        for i, data in enumerate(self.test_dataset):
            first_image = data['first_image']
            second_image = data['second_image']
            gt_homography = data['gt_homography']
            image_type = data['image_type']

            image_pair = np.stack((first_image, second_image), axis=0)
            image_pair = torch.from_numpy(image_pair).to(torch.float).to(self.device).unsqueeze(dim=1)
            # debug released mode use
            # image_pair /= 255.

            _, desp_pair, prob_pair = self.model(image_pair)
            prob_pair = f.pixel_shuffle(prob_pair, 8)
            prob_pair = spatial_nms(prob_pair, kernel_size=int(self.nms_threshold*2+1))

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

            # 得到匹配点
            matched_point = self.matcher(first_point, select_first_desp,
                                         second_point, select_second_desp)

            # 计算得到单应变换
            pred_homography, _ = cv.findHomography(matched_point[0][:, np.newaxis, ::-1],
                                                   matched_point[1][:, np.newaxis, ::-1], cv.RANSAC)

            # 对单样本进行测评
            if image_type == 'illumination':
                self.illumination_repeatability.update(first_point, second_point, gt_homography)
                self.illumination_homo_accuracy.update(pred_homography, gt_homography)
                self.illumination_mma.update(gt_homography, matched_point)
            elif image_type == 'viewpoint':
                self.viewpoint_repeatability.update(first_point, second_point, gt_homography)
                self.viewpoint_homo_accuracy.update(pred_homography, gt_homography)
                self.viewpoint_mma.update(gt_homography, matched_point)
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
        illum_repeat, illum_repeat_sum, illum_num_sum = self.illumination_repeatability.average()
        view_repeat, view_repeat_sum, view_num_sum = self.viewpoint_repeatability.average()
        total_repeat = (illum_repeat_sum + view_repeat_sum) / (illum_num_sum + view_num_sum)
        self.illumination_repeatability_mov.push(illum_repeat)
        self.viewpoint_repeatability_mov.push(view_repeat)

        # 计算估计的单应变换准确度
        illum_homo_acc, illum_homo_sum, illum_homo_num = self.illumination_homo_accuracy.average()
        view_homo_acc, view_homo_sum, view_homo_num = self.viewpoint_homo_accuracy.average()
        total_homo_acc = (illum_homo_sum + view_homo_sum) / (illum_homo_num + view_homo_num)
        self.illumination_homo_accuracy_mov.push(illum_homo_acc)
        self.viewpoint_homo_accuracy_mov.push(view_homo_acc)

        # 计算匹配的准确度
        illum_match_acc, illum_match_sum, illum_match_num = self.illumination_mma.average()
        view_match_acc, view_match_sum, view_match_num = self.viewpoint_mma.average()
        total_match_acc = (illum_match_sum + view_match_sum) / (illum_match_num + view_match_num)
        self.illumination_mma_mov.push(illum_match_acc)
        self.viewpoint_mma_mov.push(view_match_acc)

        # 统计最终的检测点数目的平均值和方差
        point_avg, point_std = self.point_statistics.average()

        self.logger.info("Homography accuracy: illumination: %.4f, viewpoint: %.4f, total: %.4f" %
                         (illum_homo_acc, view_homo_acc, total_homo_acc))
        self.logger.info("Mean Matching Accuracy: illumination: %.4f, viewpoint: %.4f, total: %.4f " %
                         (illum_match_acc, view_match_acc, total_match_acc))
        self.logger.info("Repeatability: illumination: %.4f, viewpoint: %.4f, total: %.4f" %
                         (illum_repeat, view_repeat, total_repeat))
        self.logger.info("Detection point, average: %.4f, variance: %.4f" % (point_avg, point_std))

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



