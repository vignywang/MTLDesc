#
# Created by ZhangYuyang on 2019/8/14
#
import cv2 as cv
import numpy as np
import torch
import torch.nn.functional as f


def spatial_nms(prob, kernel_size=5):
    """
    利用max_pooling对预测的特征点的概率图进行非极大值抑制
    Args:
        prob: shape为[h,w]的概率图
        kernel_size: 对每个点进行非极大值抑制时的窗口大小

    Returns:
        经非极大值抑制后的概率图
    """
    padding = int(kernel_size//2)
    pooled = f.max_pool2d(prob, kernel_size=kernel_size, stride=1, padding=padding)
    prob = torch.where(torch.eq(prob, pooled), prob, torch.zeros_like(prob))
    return prob







