import os
import argparse
import glob

from tqdm import tqdm
import numpy as np
import cv2 as cv
from scipy import io


def read_and_convert(file_dir):
    """
    Read the original make3d test image and corresponding depth,
    then crop the central part for testing
    :param file_dir: The original make3d file directory
    :param out_dir: The output file directory
    """
    img_file_dir = os.path.join(file_dir, 'Test134')
    depth_file_dir = os.path.join(file_dir, 'Gridlaserdata')

    img_file_list = glob.glob(os.path.join(img_file_dir, '*.jpg'))
    depth_file_list = glob.glob(os.path.join(depth_file_dir, '*.mat'))

    out_dir = os.path.join(file_dir, 'processed_dataset')
    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)

    file_list = open(os.path.join(out_dir, 'make3d.txt'), 'w')
    for i in tqdm(range(len(img_file_list))):
        img_file = img_file_list[i]
        depth_file = depth_file_list[i]

        img = cv.imread(img_file)
        img_final = cv.resize(img, (345, 460), interpolation=cv.INTER_LINEAR)
        img_name = img_file.split('/')[-1][4:]
        cv.imwrite(os.path.join(out_dir, img_name), img_final)

        depth_mat = io.loadmat(depth_file)
        depth_org = depth_mat['Position3DGrid'][:, :, 3]
        depth = cv.resize(depth_org, (345, 460), interpolation=cv.INTER_LINEAR)
        depth_name = depth_file.split('/')[-1][15:-4]
        np.save(os.path.join(out_dir, depth_name), depth)

        file_list.write(depth_name + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, required=True)
    args = parser.parse_args()

    read_and_convert(args.data_root)


