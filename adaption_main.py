#
# Created by ZhangYuyang on 2019/8/19
#
from utils.adaption_maker import AdaptionMaker


class Parameters:
    ckpt_file = '/home/project/development/MegPoint/magicpoint_ckpt/good_results/exp1_0.0010_64/model_59.pt'
    out_dir = '/data/MegPoint/dataset/coco/train2014/pseudo_points'
    dataset_dir = '/data/MegPoint/dataset/coco/train2014/images'

    height = 240
    width = 320
    adaption_num = 100
    top_k = 600
    nms_ksize = 7
    detection_threshold = 0.001

    logger = None


params = Parameters()
adaption_maker = AdaptionMaker(params=params)





