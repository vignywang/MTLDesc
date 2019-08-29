#
# Created by ZhangYuyang on 2019/8/19
#
import torch
import numpy as np
from utils.adaption_maker import AdaptionMaker

# make a reproducible pseudo ground truth
np.random.seed(3242)


class AdaptionParameters:
    # # first round
    ckpt_file = '/home/zhangyuyang/project/development/MegPoint/magicpoint_ckpt/good_results/adam_0.0010_64/' \
                'model_59.pt'
    # # scond round
    # ckpt_file = '/home/zhangyuyang/project/development/MegPoint/magicpoint_ckpt/good_results/adam_adaption_0.0010_64/' \
    #             'model_99.pt'

    out_root = '/data/MegPoint/dataset/coco'
    coco_dataset_root = '/data/MegPoint/dataset/coco'

    height = 240
    width = 320
    adaption_num = 100
    top_k = 300
    # nms_threshold = 4  # 7
    nms_threshold = 8  # 7
    detection_threshold = 0.005


params = AdaptionParameters()
adaption_maker = AdaptionMaker(params=params)
adaption_maker.run()




