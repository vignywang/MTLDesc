# 
# Created by ZhangYuyang on 2020/1/8
#
# 专用于评估Aachen Benchmark的数据集
import os
from glob import glob

import torch
import cv2 as cv
from torch.utils.data import Dataset


class AachenDataset(Dataset):
    """读入Aachen的图像并按原分辨率输出"""

    def __init__(self, dataset_root):
        self.dataset_root = dataset_root
        self.image_list = self._format_file_list()

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        img_dir = self.image_list[idx]
        img = cv.imread(self.image_list[idx], cv.IMREAD_GRAYSCALE)

        # debug use
        # cv.imshow("cur_img", img)
        # cv.waitKey()

        return {
            "img": img,
            "img_dir": img_dir,
        }

    def _format_file_list(self):
        db_image_list = glob(os.path.join(self.dataset_root, "db", "*.jpg"))
        # db_image_list = sorted(db_image_list, key=lambda x: int(x.split("/")[-1].split(".")[0]))
        night_image_list = glob(os.path.join(self.dataset_root, "query/night/nexus5x", "*.jpg"))
        night_image_list += glob(os.path.join(self.dataset_root, "query/night/milestone", "*.jpg"))
        # night_image_list = sorted(night_image_list, key=lambda x: int(x.split("/")[-1].split(".")[0].split("_")[-1]))

        day_image_list = glob(os.path.join(self.dataset_root, "query/day/milestone", "*.jpg"))
        day_image_list += glob(os.path.join(self.dataset_root, "query/day/nexus4", "*.jpg"))
        day_image_list += glob(os.path.join(self.dataset_root, "query/day/nexus5x", "*.jpg"))

        image_list = db_image_list + night_image_list + day_image_list
        return image_list


if __name__ == "__main__":
    dataset_root = "/home/zhangyuyang/data/aachen/Aachen-Day-Night/images/images_upright"

    dataset = AachenDataset(dataset_root)
    for i, data in enumerate(dataset):
        a = 3



