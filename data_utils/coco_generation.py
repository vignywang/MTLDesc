# 
# Created by ZhangYuyang on 2019/9/19
#
import glob
import os
import cv2 as cv
import torch
from torch.utils.data import Dataset
# from data_utils.coco_dataset import COCORawDataset
# from coco_dataset import COCORawDataset


class COCORawDataset(Dataset):

    def __init__(self, dataset_dir):
        self.height = 240
        self.width = 320
        self.dataset_dir = os.path.join(dataset_dir, 'train2014', 'images')
        self.image_file_list = self._format_file_list()

    def __len__(self):
        return len(self.image_file_list)

    def __getitem__(self, idx):
        image = cv.imread(self.image_file_list[idx], flags=cv.IMREAD_GRAYSCALE)
        image = cv.resize(image, (self.width, self.height), interpolation=cv.INTER_LINEAR)

        return image

    def _format_file_list(self, num_limits=False):
        image_list = glob.glob(os.path.join(self.dataset_dir, "*.jpg"))
        image_list = sorted(image_list)
        if num_limits:
            length = 1000
        else:
            length = len(image_list)
        image_list = image_list[:length]
        return image_list

def generate_same_size_coco(coco_data_dir):
    dataset = COCORawDataset(coco_data_dir)
    out_dir = os.path.join(coco_data_dir, 'train2014', 'resized_images')
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    for i, image in enumerate(dataset):
        image_out_dir = os.path.join(out_dir, "image_%05d.jpg" % i)
        cv.imwrite(image_out_dir, image)
        if i % 1000 == 0:
            print("Having processed %d images" % i)


if __name__ == "__main__":
    data_dir = "/data/MegPoint/dataset/coco"
    generate_same_size_coco(coco_data_dir=data_dir)















