#
# Created by ZhangYuyang on 2020/2/25
#
import cv2 as cv
import numpy as np
import torch
import torch.nn.functional as f

from nets.megpoint import resnet18_fast
from nets.megpoint import Extractor
from utils.utils import spatial_nms


class BasicMegPoint(object):

    def __init__(self, resnet_ckpt, extractor_ckpt, logger, detection_threshold=0.90, top_k=1000):
        self.logger = logger
        self.name = "megpoint"

        self.detection_threshold = detection_threshold
        self.top_k = top_k
        self.logger.info("detection_threshold=%.4f, top_k=%d" % (detection_threshold, top_k))

        if torch.cuda.is_available():
            self.logger.info('gpu is available, set device to cuda !')
            self.device = torch.device('cuda:0')
            self.gpu_count = 1
        else:
            self.logger.info('gpu is not available, set device to cpu !')
            self.device = torch.device('cpu')

        # 初始化resnet18_fast
        self.logger.info("Initialize resnet18_fast")
        self.model = resnet18_fast().to(self.device)
        self.model = self._load_model_params(resnet_ckpt, self.model)
        self.extractor = Extractor().to(self.device)
        self.extractor = self._load_model_params(extractor_ckpt, self.extractor)

    def _load_model_params(self, ckpt_file, previous_model):
        if ckpt_file is None:
            print("Please input correct checkpoint file dir!")
            return False

        self.logger.info("Load pretrained model %s " % ckpt_file)

        model_dict = previous_model.state_dict()
        pretrain_dict = torch.load(ckpt_file, map_location=self.device)
        model_dict.update(pretrain_dict)
        previous_model.load_state_dict(model_dict)
        return previous_model

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

    def generate_feature(self, img):
        """
        获取一幅BGR图像对应的特征点及其描述子
        Args:
            img_dir: 图像地址
        Returns:
            point: [n,2] 特征点
            descriptor: [n,128] 描述子
        """
        # switch to eval mode
        self.model.eval()
        self.extractor.eval()

        org_h, org_w = img.shape

        # to torch and scale to [-1,1]
        img = torch.from_numpy(img).to(torch.float).unsqueeze(dim=0).unsqueeze(dim=0).to(self.device)
        img = (img / 255.) * 2. - 1.

        # detector
        heatmap, c1, c2, c3, c4 = self.model(img)
        prob = torch.sigmoid(heatmap)
        prob = spatial_nms(prob)

        # 得到对应的预测点
        prob = prob.detach().cpu().numpy()
        prob = prob[0, 0]
        point, point_num = self._generate_predict_point(prob, top_k=self.top_k)  # [n,2]

        # descriptor
        desp = self._generate_combined_descriptor_fast(point, c1, c2, c3, c4, org_h, org_w)

        return point, desp, point_num

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

        feature = torch.cat((c1_feature, c2_feature, c3_feature, c4_feature), dim=2)
        desp = self.extractor(feature)[0]  # [n,128]

        desp = desp.detach().cpu().numpy()

        return desp

    def __call__(self, *args, **kwargs):
        raise NotImplementedError


class HPatchMegPoint(BasicMegPoint):
    """专用于hpatch测试的SuperPoint模型"""

    def __init__(self, resnet_ckpt, extractor_ckpt, logger):
        super(HPatchMegPoint, self).__init__(resnet_ckpt, extractor_ckpt, logger)

    def __call__(self, first_image, second_image, *args, **kwargs):
        first_point, first_desp, first_point_num = self.generate_feature(first_image)
        second_point, second_desp, second_point_num = self.generate_feature(second_image)

        return first_point, first_point_num, first_desp, second_point, second_point_num, second_desp





