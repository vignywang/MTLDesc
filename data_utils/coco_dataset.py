#
# Created by ZhangYuyang on 2019/8/19
#
import os
import glob
import cv2 as cv

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


class COCOAdaptionDataset(Dataset):

    def __init__(self, params):
        self.params = params
        self.height = params.height
        self.width = params.width
        self.dataset_dir = params.dataset_dir
        self.image_list = self._format_file_list()

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image = cv.imread(self.image_list[idx], flags=cv.IMREAD_GRAYSCALE)
        image = cv.resize(image, (self.width, self.height), interpolation=cv.INTER_LINEAR)
        return image

    def _format_file_list(self):
        image_list = glob.glob(os.path.join(self.dataset_dir, "*.jpg"))
        image_list = sorted(image_list)
        return image_list


if __name__ == "__main__":
    class Parameters:
        dataset_dir = '/data/MegPoint/dataset/coco/train2014/images'
        height = 240
        width = 320

    params = Parameters()
    COCO_adaption_dataset = COCOAdaptionDataset(params)
    for i, image in enumerate(COCO_adaption_dataset):
        cv.imshow("image", image)
        cv.waitKey()











