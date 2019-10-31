# 
# Created by ZhangYuyang on 2019/10/30
#
import torch
import numpy as np
from experiments.exp_heatmap_labeling import Labeler


def setup_seed():
    # make the result reproducible
    torch.manual_seed(3928)
    torch.cuda.manual_seed_all(2342)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(2933)


class HeatmapAdaptionParameters(object):

    def __init__(self):
        self.dataset_dir = "/data/MegPoint/dataset/coco"
        self.out_dir = "/data/MegPoint/dataset/coco/train2014/heatmap_image_pseudo_point_00"
        self.ckpt_file = "/home/zhangyuyang/project/development/MegPoint/magicpoint_ckpt/synthetic_heatmap_l2_0.0010_32/model_76.pt"
        self.height = 240
        self.width = 320
        self.batch_size = 24
        self.adaption_num = 100
        self.point_portion = 0.001
        self.detection_threshold = 0.5
        self.nms_threshold = 4
        self.top_k = 300
        self.point_threshold = None


if __name__ == "__main__":
    setup_seed()
    params = HeatmapAdaptionParameters()
    labeler = Labeler(params)
    labeler.label()




