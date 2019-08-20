#
# Created by ZhangYuyang on 2019/8/19
#
import torch
import numpy as np
from utils.adaption_maker import AdaptionMaker

np.random.seed(3242)


class Parameters:
    ckpt_file = '/home/zhangyuyang/project/development/MegPoint/magicpoint_ckpt/good_results/adam_0.0010_64/model_59.pt'
    out_dir = '/data/MegPoint/dataset/coco/train2014/pseudo_points'
    dataset_dir = '/data/MegPoint/dataset/coco/train2014/images'

    height = 240
    width = 320
    adaption_num = 0  # 50
    top_k = 600
    nms_ksize = 7
    detection_threshold = 0.001

    logger = None


params = Parameters()
adaption_maker = AdaptionMaker(params=params)
adaption_maker.run()




