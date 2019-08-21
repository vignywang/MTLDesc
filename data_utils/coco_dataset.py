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

    def __init__(self, params, dataset_type):
        assert dataset_type in ['train2014', 'val2014']
        self.params = params
        self.height = params.height
        self.width = params.width
        self.dataset_dir = os.path.join(params.dataset_root, dataset_type, 'images')
        if dataset_type == 'train2014':
            num_limits = False
        else:
            num_limits = True
        self.image_list, self.image_name_list = self._format_file_list(num_limits)

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image = cv.imread(self.image_list[idx], flags=cv.IMREAD_GRAYSCALE)
        image = cv.resize(image, (self.width, self.height), interpolation=cv.INTER_LINEAR)
        name = self.image_name_list[idx]
        sample = {'image': image, 'name': name}
        return sample

    def _format_file_list(self, num_limits=False):
        image_list = glob.glob(os.path.join(self.dataset_dir, "*.jpg"))
        image_list = sorted(image_list)
        if num_limits:
            length = 1000
        else:
            length = len(image_list)
        image_name_list = []
        for image in image_list:
            image_name = (image.split('/')[-1]).split('.')[0]
            image_name_list.append(image_name)
        image_list = image_list[:length]
        image_name_list = image_name_list[:length]
        return image_list, image_name_list


if __name__ == "__main__":
    class Parameters:
        dataset_dir = '/data/MegPoint/dataset/coco/train2014/images'
        height = 240
        width = 320

    params = Parameters()
    COCO_adaption_dataset = COCOAdaptionDataset(params)
    for i, data in enumerate(COCO_adaption_dataset):
        image = data['image']
        name = data['name']
        print(name)
        cv.imshow("image", image)
        cv.waitKey()











