# 
# Created by ZhangYuyang on 2019/12/21
#
import os
from glob import glob

import numpy as np


def generate_satisfied_point_dataset(dataset_dir, required_point=100):
    org_image_list = glob(os.path.join(dataset_dir, "*.jpg"))
    org_image_list = sorted(org_image_list)

    count = 0
    with open(os.path.join(dataset_dir, "file_list.txt"), "w") as wf:
        for i, org_image_dir in enumerate(org_image_list):
            name = (org_image_dir.split('/')[-1]).split('.')[0]
            point_dir = os.path.join(dataset_dir, name + '.npy')

            point = np.load(point_dir)
            point_num = point.shape[0]
            if point_num >= required_point:
                wf.write(org_image_dir + "," + point_dir + "\n")
                count += 1

            if i % 1000 == 0:
                print("Having processed %d images, including %d satisfied image and point" % (i, count))

    print("Processing done. Totally %d satisfied image and point." % count)


if __name__ == "__main__":
    # dataset_dir = "/data/MegPoint/dataset/coco/train2014/pseudo_image_points_0"
    dataset_dir = "/data/MegPoint/dataset/coco/train2014/pseudo_image_points_1"
    generate_satisfied_point_dataset(dataset_dir)


