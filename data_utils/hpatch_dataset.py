#
# Created by ZhangYuyang on 2019/8/21
#
import os
import glob
import cv2 as cv
import numpy as np
from torch.utils.data import Dataset


def generate_hpatch_data_list(hpatch_dir):
    """
    输入HPatch数据的原始地址，存储以图像对（包括一对图像各自的地址以及对应的单应变换真值）为单位的list
    Args:
        hpatch_dir: HPatch数据的根目录
    """

    print('Generate HPatch data list:')
    illumination_folder = glob.glob(os.path.join(hpatch_dir, "i_*"))
    viewpoint_folder = glob.glob(os.path.join(hpatch_dir, 'v_*'))
    viewpoint_folder = sorted(viewpoint_folder)
    illumination_folder = sorted(illumination_folder)

    image_name = ['2.ppm', '3.ppm', '4.ppm', '5.ppm', '6.ppm']
    homograpy_name = ['H_1_2', 'H_1_3', 'H_1_4', 'H_1_5', 'H_1_6']

    # generate viewpoint relating data list
    viewpoint_list = []
    for folder in viewpoint_folder:
        first_image_dir = os.path.join(folder, '1.ppm')
        for i in range(5):
            # homograpy = np.loadtxt(os.path.join(folder, homograpy_name[i]), dtype=np.float)
            second_image_dir = os.path.join(folder, image_name[i])
            homograpy_dir = os.path.join(folder, homograpy_name[i])
            data_dir = ','.join([first_image_dir, second_image_dir, homograpy_dir])
            viewpoint_list.append(data_dir)

    # generate illumination relating data list
    illumination_list = []
    for folder in illumination_folder:
        first_image_dir = os.path.join(folder, '1.ppm')
        for i in range(5):
            second_image_dir = os.path.join(folder, image_name[i])
            homograpy_dir = os.path.join(folder, homograpy_name[i])
            data_dir = ','.join([first_image_dir, second_image_dir, homograpy_dir])
            illumination_list.append(data_dir)

    with open(os.path.join(hpatch_dir, 'viewpoint_list.txt'), 'w') as vf:
        for viewpoint_data_dir in viewpoint_list:
            vf.write(viewpoint_data_dir+'\n')

    with open(os.path.join(hpatch_dir, 'illumination_list.txt'), 'w') as ilf:
        for illumination_data_dir in illumination_list:
            ilf.write(illumination_data_dir+'\n')

    print('Generation Done!')


class HPatchDataset(Dataset):

    def __init__(self, params):
        self.params = params
        self.height = params.height
        self.width = params.width
        self.dataset_dir = params.hpatch_dataset_dir
        self.data_list = self._format_file_list()

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        first_image_dir = self.data_list[idx]['first']
        second_image_dir = self.data_list[idx]['second']
        homo_dir = self.data_list[idx]['homo_dir']
        image_type = self.data_list[idx]['type']

        first_image = cv.imread(first_image_dir, cv.IMREAD_GRAYSCALE)
        second_image = cv.imread(second_image_dir, cv.IMREAD_GRAYSCALE)

        org_first_shape = np.shape(first_image)
        org_second_shape = np.shape(second_image)
        resize_shape = np.array((self.height, self.width), dtype=np.float)
        # scale is used to recover the location in original image scale
        first_scale = resize_shape / org_first_shape
        second_scale = resize_shape / org_second_shape
        homo = np.loadtxt(homo_dir, dtype=np.float)
        homo = self._generate_adjust_homography(first_scale, second_scale, homo)

        first_image = cv.resize(first_image, (self.width, self.height), interpolation=cv.INTER_LINEAR)
        second_image = cv.resize(second_image, (self.width, self.height), interpolation=cv.INTER_LINEAR)
        # image_pair = np.stack((first_image, second_image), axis=0)

        sample = {'first_image': first_image, 'second_image': second_image,
                  'image_type': image_type, 'gt_homography': homo}
        return sample

    @staticmethod
    def _generate_adjust_homography(first_scale, second_scale, homography):
        first_inv_scale_mat = np.diag((1. / first_scale[1], 1. / first_scale[0], 1))
        second_scale_mat = np.diag((second_scale[1], second_scale[0], 1))
        adjust_homography = np.matmul(second_scale_mat, np.matmul(homography, first_inv_scale_mat))
        return adjust_homography

    def _format_file_list(self):
        data_list = []
        with open(os.path.join(self.dataset_dir, 'illumination_list.txt'), 'r') as ilf:
            illumination_lines = ilf.readlines()
            for line in illumination_lines:
                line = line[:-1]
                first_dir, second_dir, homo_dir = line.split(',')
                dir_slice = {'first': first_dir, 'second': second_dir, 'homo_dir': homo_dir, 'type': 'illumination'}
                data_list.append(dir_slice)

        with open(os.path.join(self.dataset_dir, 'viewpoint_list.txt'), 'r') as vf:
            viewpoint_lines = vf.readlines()
            for line in viewpoint_lines:
                line = line[:-1]
                first_dir, second_dir, homo_dir = line.split(',')
                dir_slice = {'first': first_dir, 'second': second_dir, 'homo_dir': homo_dir, 'type': 'viewpoint'}
                data_list.append(dir_slice)

        return data_list


if __name__ == "__main__":
    # # uncomment to generate the data list
    # hpatch_dir = '/data/MegPoint/dataset/hpatch'
    # generate_hpatch_data_list(hpatch_dir)

    pass



