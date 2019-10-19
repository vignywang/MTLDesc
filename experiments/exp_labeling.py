# 
# Created by ZhangYuyang on 2019/10/16
#
import os
import time

import cv2 as cv
import numpy as np
import torch
import torch.nn.functional as f
from torch.utils.data import DataLoader

# from nets.megpoint_net import EncoderDecoderMegPoint
from nets.megpoint_net import STMegPointNet
from data_utils.coco_dataset import COCOMegPointRawDataset
from data_utils.coco_dataset import COCOMegPointDebugDataset
from data_utils.dataset_tools import draw_image_keypoints
from utils.utils import spatial_nms


class SimpleParams(object):

    def __init__(self):
        self.batch_size = 48
        self.height = 240
        self.width = 320
        self.patch_height = 30
        self.patch_width = 40
        self.coco_dataset_dir = '/data/MegPoint/dataset/coco'

        self.megpoint_ckpt = [
            "/home/zhangyuyang/project/development/MegPoint/magicpoint_ckpt/good_results/synthetic_new_0.0010_64/model_59.pt",
            "/home/zhangyuyang/project/development/MegPoint/megpoint_ckpt/only_detector_re_50_0.0010_48/model_49.pt",
            # "/home/zhangyuyang/project/development/MegPoint/megpoint_ckpt/st_setting_3_0.0010_16/model_04.pt",
            # "/home/zhangyuyang/project/development/MegPoint/megpoint_ckpt/st_setting_3_0.0010_16/model_09.pt"
        ]

        self.label_threshold = [
            [0.9961, 0.9444, 0.0106],
            [0.9997, 0.9487, 0.0032]
        ]

        self.colors = [
            # (255, 0, 0),
            (0, 255, 0),
            (0, 0, 255)
        ]


def generate_predict_point_by_threshold(prob, threshold):
    point_idx = np.where(prob > threshold)

    if point_idx[0].size == 0:
        point = np.empty((0, 2), dtype=np.int64)
        point_num = 0
    else:
        point = np.stack(point_idx, axis=1)  # [n,2]
        point_num = point.shape[0]

    return point, point_num


def statistic_labeling(idx, has_threshold=True):

    params = SimpleParams()
    raw_dataset = COCOMegPointRawDataset(params)
    epoch_length = len(raw_dataset) // params.batch_size
    data_loader = DataLoader(
        dataset=raw_dataset,
        batch_size=params.batch_size,
        shuffle=False,
        num_workers=8,
        drop_last=False
    )

    device = torch.device("cuda:0")
    # 初始化模型，并读取预训练的参数
    print("Load pretrained model: %s" %params.megpoint_ckpt[idx])
    model = STMegPointNet()
    # model = EncoderDecoderMegPoint(only_detector=True)
    model_dict = model.state_dict()
    pretrained_dict = torch.load(params.megpoint_ckpt[idx], map_location=device)
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    model = model.to(device)

    out_dir = "/data/MegPoint/dataset/coco/train2014/st_image_pseudo_point_%d" % idx
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    model.eval()
    nms_threshold = 4

    def compute_threshold(model, region_portion, point_portion):
        model.eval()
        # 计算每一张图像的关键点的概率图，并统计所有的概率
        print("Begin to compute threshold of region_portion: %.4f" % region_portion)
        print("Begin to compute threshold of point_portion: %.4f" % point_portion)
        region_partial_idx = int(params.batch_size*params.patch_height*params.patch_width*region_portion)
        point_partial_idx = int(params.batch_size*params.height*params.width*point_portion)
        start_time = time.time()

        all_point_region_prob = []
        all_point_prob = []

        for i, data in enumerate(data_loader):
            image = data["image"].to(device)
            results = model(image)  # [bt,2,h,w]
            point_prob = results[1]

            point_region_prob = point_prob.sum(dim=1)
            point_prob = f.pixel_shuffle(point_prob, 8)

            # 统计预测出有点的区域的个数，以及概率
            select_point_region_prob = point_region_prob.reshape((-1,))
            select_point_region_prob, _ = torch.sort(select_point_region_prob, descending=True)
            select_point_region_prob = select_point_region_prob[:region_partial_idx]
            all_point_region_prob.append(select_point_region_prob.detach().cpu().numpy())

            # 统计处点区域中预测点的概率，其个数应与有点区域数目相等
            select_point_prob = point_prob.squeeze().reshape((-1,))  # [bt*h*w]
            select_point_prob, _ = torch.sort(select_point_prob, descending=True)
            select_point_prob = select_point_prob[:point_partial_idx]
            all_point_prob.append(select_point_prob.detach().cpu().numpy())

            if i % 50 == 0:
                print(
                    "Batch %04d/%04d have been computed , %.3fs/step" %
                    (
                        i,
                        epoch_length,
                        (time.time()-start_time)/50
                    )
                )
                start_time = time.time()

        all_point_region_prob = np.concatenate(all_point_region_prob, axis=0)
        point_region_idx = int(all_point_region_prob.size * region_portion)
        point_region_idx = all_point_region_prob.size - point_region_idx - 1
        point_region_threshold = np.partition(all_point_region_prob, kth=point_region_idx)[point_region_idx]

        all_point_prob = np.concatenate(all_point_prob, axis=0)
        point_idx = int(all_point_prob.size * point_portion)
        point_idx = all_point_prob.size - point_idx - 1
        point_threshold = np.partition(all_point_prob, kth=point_idx)[point_idx]

        print("Computing Done!")
        print("The point_region_threshold is: %.4f" % point_region_threshold)
        print("The point_threshold is: %.4f" % point_threshold)

        return point_region_threshold, point_threshold

    if has_threshold:
        point_region_threshold, point_threshold = params.label_threshold[idx]
    else:
        point_region_threshold, point_threshold = compute_threshold(model, region_portion=0.5, point_portion=0.05)
    # 再次对数据集中每一张图像进行预测且标注
    print("Begin to label all target data by point_region_threshold, "
          "point_threshold")
    print("%.4f, %.4f" % (point_region_threshold, point_threshold))
    start_time = time.time()
    count = 0
    for i, data in enumerate(data_loader):
        image = data["image"].to(device)
        name = data["name"]
        results = model(image)
        point_prob = results[1]
        point_region_prob = point_prob.sum(dim=1)

        point_prob = f.pixel_shuffle(point_prob, 8)  # [bt,1,h,w]
        point_prob = spatial_nms(point_prob, kernel_size=int(nms_threshold * 2 + 1))

        point_region_mask = torch.where(
            torch.ge(point_region_prob, point_region_threshold),
            torch.ones_like(point_region_prob),
            torch.zeros_like(point_region_prob)
        ).unsqueeze(dim=1).repeat((1, 64, 1, 1))
        point_region_mask = f.pixel_shuffle(point_region_mask, 8)

        valid_mask = point_region_mask

        point_prob = (point_prob * valid_mask).squeeze().detach().cpu().numpy()
        valid_mask = valid_mask.squeeze().detach().cpu().numpy()

        image = ((image + 1) * 255. / 2.).to(torch.uint8).squeeze().cpu().numpy()

        batch_size = point_prob.shape[0]
        for j in range(batch_size):
            # 得到对应的预测点
            cur_point, cur_point_num = generate_predict_point_by_threshold(
                point_prob[j], threshold=point_threshold)
            cur_image = image[j]
            cur_name = name[j]
            cur_mask = valid_mask[j]

            cv.imwrite(os.path.join(out_dir, cur_name + '.jpg'), cur_image)
            np.save(os.path.join(out_dir, cur_name + "_mask"), cur_mask)
            np.save(os.path.join(out_dir, cur_name), cur_point)

            count += 1

        if count >= 100:
            break

        if i % 50 == 0:
            print(
                "Batch %04d/%04d have been labeled, %.3fs/step" %
                (
                    i,
                    params.batch_size,
                    (time.time() - start_time) / 50
                )
            )
            start_time = time.time()

    print("Labeling done.")


def read_image_point_and_show():
    params = SimpleParams()
    selftraining_dataset = COCOMegPointDebugDataset(params, read_mask=True, postfix="st_image_pseudo_point_0")
    superpoint_dataset = COCOMegPointDebugDataset(params, postfix="pseudo_image_points_0")

    st_iter = enumerate(selftraining_dataset)
    su_iter = enumerate(superpoint_dataset)

    for i in range(100):
        _, st_data = st_iter.__next__()
        _, su_data = su_iter.__next__()

        st_image, st_point, st_mask = st_data["image"], st_data["point"], st_data["mask"]
        su_image, su_point = su_data["image"], su_data["point"]

        st_image_point = draw_image_keypoints(image=st_image, points=st_point, show=False)
        st_zeros = np.zeros_like(st_mask)
        st_mask = (np.stack((st_zeros, st_zeros, st_mask), axis=2)*150)  # [h,w,3]
        st_image_point = np.clip((st_image_point.astype(np.float) + st_mask), 0, 255).astype(np.uint8)

        su_image_point = draw_image_keypoints(image=su_image, points=su_point, show=False)

        cat_image_point = np.concatenate((st_image_point, su_image_point), axis=1)

        cv.imshow("cat_image_point", cat_image_point)
        cv.waitKey()


def contrast():
    params = SimpleParams()
    dataset_num = len(params.megpoint_ckpt)
    selftraining_data_iter = []
    for i in range(dataset_num):
        dataset = COCOMegPointDebugDataset(params, read_mask=True, postfix="st_image_pseudo_point_%d" % i)
        selftraining_data_iter.append(enumerate(dataset))

    superpoint_dataset = COCOMegPointDebugDataset(params, postfix="pseudo_image_points_0")

    su_iter = enumerate(superpoint_dataset)

    for i in range(100):
        _, su_data = su_iter.__next__()
        su_image, su_point = su_data["image"], su_data["point"]
        su_image_point = draw_image_keypoints(image=su_image, points=su_point, show=False)

        st_image_point_list = []
        for j in range(dataset_num):
            _, st_data = selftraining_data_iter[j].__next__()
            st_image, st_point, st_mask = st_data["image"], st_data["point"], st_data["mask"]
            st_image_point = draw_image_keypoints(image=st_image, points=st_point, show=False)
            st_zeros = np.zeros_like(st_mask)
            st_mask = (np.stack((st_zeros, st_zeros, st_mask), axis=2)*150)  # [h,w,3]
            st_image_point = np.clip((st_image_point.astype(np.float) + st_mask), 0, 255).astype(np.uint8)
            st_image_point_list.append(st_image_point)

        st_image_point = np.concatenate(st_image_point_list, axis=1)
        cat_image_point = np.concatenate((st_image_point, su_image_point), axis=1)

        cv.imshow("cat_image_point", cat_image_point)
        cv.waitKey()

if __name__ == "__main__":
    for i in range(2):
    # a = 0.0965, 0.0377
        statistic_labeling(i, False)
    # image = cv.imread("/data/MegPoint/dataset/coco/train2014/st_image_pseudo_point_0/image_00003.jpg")
    # cv.imshow("image", image)
    # cv.waitKey()

    # read_image_point_and_show()

    contrast()




















