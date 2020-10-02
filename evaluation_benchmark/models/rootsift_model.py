#
# Created by ZhangYuyang on 2020/2/25
#
import os
import time

import cv2 as cv
import numpy as np
import torch
import torch.nn.functional as f
# from thop import profile
from torchsummary import summary


class RootsiftModel(object):

    def __init__(self,  **config):
        self.name = 'Root-SIFT'
        self.config = {
            'top_k': 10000,
        }
        self.config.update(config)
        self.model = cv.xfeatures2d_SIFT.create(nfeatures=self.config['top_k'])

        self.time_collect = []

    def average_inference_time(self):
        average_time = sum(self.time_collect) / len(self.time_collect)
        info = ('MLIFeat average inference time: {}ms / {}fps'.format(
            round(average_time*1000), round(1/average_time))
        )
        print(info)
        return info

    def predict(self, img, keys="*"):
        """
        获取一幅灰度图像对应的特征点及其描述子
        Args:
            img: [h,w] 灰度图像,要求h,w能被16整除
        Returns:
            point: [n,2] 特征点,输出点以y,x为顺序
            descriptor: [n,128] 描述子
        """
        # detect and compute
        shape = img.shape
        keypoints_cv, desp = self.model.detectAndCompute(img, None)
        desp /= (desp.sum(axis=1, keepdims=True) + 1e-7)
        desp = np.sqrt(desp)

        # 将cv点转换为numpy格式的
        point = self._convert_cv2pt(keypoints_cv)

        predictions = {
            "keypoints": point,
            "descriptors": desp,
            'shape':shape,
        }

        if keys != '*':
            predictions = {k: predictions[k] for k in keys}

        return predictions

    @staticmethod
    def _convert_cv2pt(cv_point):
        point_list = []
        for i, cv_pt in enumerate(cv_point):
            pt = np.array((cv_pt.pt[0], cv_pt.pt[1]))  # x,y的顺序
            point_list.append(pt)
        point = np.stack(point_list, axis=0)
        return point

    def __call__(self, *args, **kwargs):
        raise NotImplementedError

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

