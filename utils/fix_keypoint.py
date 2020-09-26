#
# Created by ZhangYuyang on 2019/8/19
#
import os
import time
import numpy as np
import cv2 as cv
import torch
import torch.nn.functional as f
from tqdm import tqdm

from nets.superpoint_net import SuperPointNetFloat
from data_utils.dataset_tools import HomographyAugmentation
from data_utils.coco_dataset import COCOAdaptionDataset
from data_utils.megadepth_dataset import MegaDepthAdaptionDataset
from utils.utils import spatial_nms
from utils.utils import draw_image_keypoints


class FixpointMaker(object):

    def __init__(self, params, use_gpu=True, height=240, width=320):
        self.params = params
        self.ckpt_file = params.ckpt_file
        self.out_root = params.out_root
        self.height = height
        self.width = width
        self.adaption_num = params.adaption_num
        self.nms_threshold = params.nms_threshold
        self.top_k = params.top_k
        self.detection_threshold = params.detection_threshold
        if torch.cuda.is_available() and use_gpu:
            # self.logger.info('gpu is available, set device to cuda!')
            print('gpu is available, set device to cuda!')
            self.device = torch.device('cuda:0')
        else:
            # self.logger.info('gpu is not available, set device to cpu!')
            print('gpu is not available, set device to cpu!')
            self.device = torch.device('cpu')
        # self.device = torch.device('cpu')
        self.homography = HomographyAugmentation()

        if params.dataset_type == 'coco':
            # 初始化coco数据集（整个训练数据集和1000个验证数据集）, dataset_type只能是train2014或者val2014
            train_dataset = COCOAdaptionDataset(params, 'train2014')
            val_dataset = COCOAdaptionDataset(params, 'val2014')
        elif params.dataset_type == 'megadepth':
            # 初始化megadepth数据集，只生成训练集的标注
            train_dataset = MegaDepthAdaptionDataset(params.dataset_dir)
            val_dataset = None
        else:
            print("Unrecognized dataset_type: %s" % params.dataset_type)
            assert False

        # 初始化模型
        model = SuperPointNetFloat()
        # 从预训练的模型中恢复参数
        model_dict = model.state_dict()
        pretrain_dict = torch.load(self.ckpt_file, map_location=self.device)
        model_dict.update(pretrain_dict)
        model.load_state_dict(model_dict)
        model.to(self.device)

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.model = model
        self.train_out_dir = os.path.join(self.out_root, 'train')
        self.val_out_dir = os.path.join(self.out_root, 'val')

        # if params.dataset_type == 'coco':
        #     self.train_out_dir = self.train_out_dir + '_%d' % self.params.rounds
        #     self.val_out_dir = self.val_out_dir + '_%d' % self.params.rounds
        # else:
        #     pass

        if not os.path.exists(self.train_out_dir):
            os.mkdir(self.train_out_dir)
            os.mkdir(self.val_out_dir)
        print('The train_out_dir is: %s' % self.train_out_dir)
        print('The val_out_dir is: %s' % self.val_out_dir)

    def run(self):

        start_time = time.time()
        print("*****************************************************")
        print("Generating COCO pseudo-ground truth via model %s" % self.ckpt_file)

        if self.train_dataset is not None:
            # generate train image & pseudo ground truth
            if self.params.dataset_type == "coco":
                self._generate_pseudo_ground_truth(self.train_dataset, self.train_out_dir)
            else:
                self._generate_pseudo_ground_truth_megadepth(self.train_dataset, self.train_out_dir)

        if self.val_dataset is not None:
            # generate val image & pseudo ground truth
            if self.params.dataset_type == "coco":
                self._generate_pseudo_ground_truth(self.val_dataset, self.val_out_dir)
            else:
                self._generate_pseudo_ground_truth_megadepth(self.val_dataset, self.val_out_dir)

        print("*****************************************************")
        print("Generating COCO pseudo-ground truth done. Takes %.3f h" % ((time.time()-start_time)/3600))

    def _generate_pseudo_ground_truth(self, dataset, out_dir):
        self.model.eval()

        dataset_length = len(dataset)
        for i, data in enumerate(dataset):
            image = data['image']
            name = data['name']
            images = []
            inv_homographies = []
            images.append(image)
            inv_homographies.append(np.eye(3))
            for j in range(self.adaption_num):
                homography = self.homography.sample(height=self.height, width=self.width)
                inv_homography = np.linalg.inv(homography)
                transformed_image = cv.warpPerspective(image, homography, (self.width, self.height),
                                                       flags=cv.INTER_LINEAR)
                images.append(transformed_image)
                inv_homographies.append(inv_homography)

            # 送入网络做统一计算
            batched_image = torch.from_numpy(np.stack(images, axis=0)).unsqueeze(dim=1).to(torch.float)
            batched_image_list = [batched_image]
            if self.adaption_num > 50:
                batched_image_list = torch.split(batched_image, [50, self.adaption_num-49], dim=0)
            prob_list = []
            for img in batched_image_list:
                img = img.to(self.device)
                _, _, prob, _ = self.model(img)
                prob_list.append(prob)

            # 将概率图展开为原始图像大小
            prob = torch.cat(prob_list, dim=0)
            prob = f.pixel_shuffle(prob, 8)
            prob = prob.detach().cpu().numpy()[:, 0, :, :]  # [n+1, h, w]

            # 分别得到每个预测的概率图反变换的概率图
            count = np.ones_like(image)
            counts = []
            probs = []
            counts.append(count)
            for j in range(self.adaption_num+1):
                transformed_prob = cv.warpPerspective(prob[j], inv_homographies[j], (self.width, self.height),
                                                      flags=cv.INTER_LINEAR)
                transformed_count = cv.warpPerspective(count, inv_homographies[j], (self.width, self.height),
                                                       flags=cv.INTER_NEAREST)
                probs.append(transformed_prob)
                counts.append(transformed_count)

            probs = np.stack(probs, axis=2)  # [h,w,n+1]
            counts = np.stack(counts, axis=2)
            probs = np.sum(probs, axis=2)
            counts = np.sum(counts, axis=2)
            probs = probs/counts
            # todo:此处可改进为不用torch的方式，这样就不必要转换数据类型
            torch_probs = torch.from_numpy(probs).unsqueeze(dim=0).unsqueeze(dim=0)
            final_probs = spatial_nms(torch_probs, int(self.nms_threshold*2+1)).detach().cpu().numpy()[0, 0]

            satisfied_idx = np.where(final_probs > self.detection_threshold)
            ordered_satisfied_idx = np.argsort(final_probs[satisfied_idx])[::-1]  # 降序
            if len(ordered_satisfied_idx) < self.top_k:
                points = np.stack((satisfied_idx[0][ordered_satisfied_idx],
                                   satisfied_idx[1][ordered_satisfied_idx]), axis=1)
            else:
                points = np.stack((satisfied_idx[0][:self.top_k],
                                   satisfied_idx[1][:self.top_k]), axis=1)
            # debug hyper parameters use
            image_point = draw_image_keypoints(image, points)
            cv.imwrite('/home/yuyang/tmp/adaption/%03d.jpg' % i, image_point)
            np.save(os.path.join(out_dir, name), points)

            if i % 100 == 0:
                print("Having processed %dth/%d image" % (i, dataset_length))

    def _generate_pseudo_ground_truth_megadepth(self, dataset, out_dir):
        self.model.eval()

        dataset_length = len(dataset)
        for i, data in tqdm(enumerate(dataset)):
            image_pair = data['image']
            name = data['name']
            image_0, image_1 = np.split(image_pair, 2, axis=1)
            image_pair = [image_0, image_1]
            # image_point_pair = []
            points_pair = []
            for image in image_pair:
                images = []
                inv_homographies = []
                images.append(image)
                inv_homographies.append(np.eye(3))
                for j in range(self.adaption_num):
                    homography = self.homography.sample(height=self.height, width=self.width)
                    inv_homography = np.linalg.inv(homography)
                    transformed_image = cv.warpPerspective(image, homography, (self.width, self.height),
                                                           flags=cv.INTER_LINEAR)
                    images.append(transformed_image)
                    inv_homographies.append(inv_homography)

                # 送入网络做统一计算
                batched_image = torch.from_numpy(np.stack(images, axis=0)).unsqueeze(dim=1).to(torch.float)
                batched_image_list = [batched_image]
                if self.adaption_num > 50:
                    batched_image_list = torch.split(batched_image, [50, self.adaption_num-49], dim=0)
                prob_list = []
                for img in batched_image_list:
                    img = img.to(self.device)
                    _, _, prob, _ = self.model(img)
                    prob_list.append(prob)

                # 将概率图展开为原始图像大小
                prob = torch.cat(prob_list, dim=0)
                prob = f.pixel_shuffle(prob, 8)
                prob = prob.detach().cpu().numpy()[:, 0, :, :]  # [n+1, h, w]

                # 分别得到每个预测的概率图反变换的概率图
                count = np.ones_like(image)
                counts = []
                probs = []
                counts.append(count)
                for j in range(self.adaption_num+1):
                    transformed_prob = cv.warpPerspective(prob[j], inv_homographies[j], (self.width, self.height),
                                                          flags=cv.INTER_LINEAR)
                    transformed_count = cv.warpPerspective(count, inv_homographies[j], (self.width, self.height),
                                                       flags=cv.INTER_NEAREST)
                    probs.append(transformed_prob)
                    counts.append(transformed_count)

                probs = np.stack(probs, axis=2)  # [h,w,n+1]
                counts = np.stack(counts, axis=2)
                probs = np.sum(probs, axis=2)
                counts = np.sum(counts, axis=2)
                probs = probs/counts
                # todo:此处可改进为不用torch的方式，这样就不必要转换数据类型
                torch_probs = torch.from_numpy(probs).unsqueeze(dim=0).unsqueeze(dim=0)
                final_probs = spatial_nms(torch_probs, int(self.nms_threshold*2+1)).detach().cpu().numpy()[0, 0]

                satisfied_idx = np.where(final_probs > self.detection_threshold)
                ordered_satisfied_idx = np.argsort(final_probs[satisfied_idx])[::-1]  # 降序
                if len(ordered_satisfied_idx) < self.top_k:
                    points = np.stack((satisfied_idx[0][ordered_satisfied_idx],
                                       satisfied_idx[1][ordered_satisfied_idx]), axis=1)
                else:
                    points = np.stack((satisfied_idx[0][:self.top_k],
                                       satisfied_idx[1][:self.top_k]), axis=1)
                # debug hyper parameters use
                # image_point = draw_image_keypoints(image, points)
                # image_point_pair.append(image_point)
                points_pair.append(points)

            # image_point = np.concatenate(image_point_pair, axis=0)
            # cv.imwrite('/home/yuyang/tmp/adaption/%03d.jpg' % i, image_point)

            np.savez(os.path.join(out_dir, name), points_0=points_pair[0], points_1=points_pair[1])
            # tmp_file = np.load(os.path.join(out_dir, name))

            # if i % 100 == 0:
            #     print("Having processed %dth/%d image" % (i, dataset_length))


