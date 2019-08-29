#
# Created by ZhangYuyang on 2019/8/19
#
import os
import time
import numpy as np
import cv2 as cv
import torch
import torch.nn.functional as f

from nets.superpoint_net import SuperPointNet
from data_utils.dataset_tools import HomographyAugmentation
from data_utils.coco_dataset import COCOAdaptionDataset
from utils.utils import spatial_nms
from utils.utils import draw_image_keypoints


class AdaptionMaker(object):

    def __init__(self, params):
        self.params = params
        self.ckpt_file = params.ckpt_file
        self.out_root = params.out_root
        self.height = params.height
        self.width = params.width
        self.adaption_num = params.adaption_num
        self.nms_threshold = params.nms_threshold
        self.top_k = params.top_k
        self.detection_threshold = params.detection_threshold
        if torch.cuda.is_available():
            # self.logger.info('gpu is available, set device to cuda!')
            print('gpu is available, set device to cuda!')
            self.device = torch.device('cuda:0')
        else:
            # self.logger.info('gpu is not available, set device to cpu!')
            print('gpu is not available, set device to cpu!')
            self.device = torch.device('cpu')
        # self.device = torch.device('cpu')
        self.homography = HomographyAugmentation()

        # 初始化coco数据集（整个训练数据集和1000个验证数据集）, dataset_type只能是train2014或者val2014
        train_dataset = COCOAdaptionDataset(params, 'train2014')
        val_dataset = COCOAdaptionDataset(params, 'val2014')

        # 初始化模型
        model = SuperPointNet()
        # 从预训练的模型中恢复参数
        model_dict = model.state_dict()
        pretrain_dict = torch.load(self.ckpt_file)
        model_dict.update(pretrain_dict)
        model.load_state_dict(model_dict)
        model.to(self.device)

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.model = model
        self.train_out_dir = os.path.join(self.out_root, 'train2014', 'pseudo_image_points')
        self.val_out_dir = os.path.join(self.out_root, 'val2014', 'pseudo_image_points')
        count = 1
        while os.path.exists(self.train_out_dir):
            self.train_out_dir += '%d' % count
            self.val_out_dir += '%d' % count
            count += 1
        os.mkdir(self.train_out_dir)
        os.mkdir(self.val_out_dir)
        print('The train_out_dir is: %s' % self.train_out_dir)
        print('The val_out_dir is: %s' % self.train_out_dir)

    def run(self):

        start_time = time.time()
        print("*****************************************************")
        print("Generating COCO pseudo-ground truth via model %s" % self.ckpt_file)

        # generate train image & pseudo ground truth
        self._generate_pseudo_ground_truth(self.train_dataset, self.train_out_dir)

        # generate val image & pseudo ground truth
        self._generate_pseudo_ground_truth(self.val_dataset, self.val_out_dir)

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
                homography = self.homography.sample()
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
                _, _, prob = self.model(img)
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
                # 只取前100个点
                points = np.stack((satisfied_idx[0][:100],
                                   satisfied_idx[1][:100]), axis=1)
            # debug hyper parameters use
            # draw_image_keypoints(image, points)
            cv.imwrite(os.path.join(out_dir, name + '.jpg'), image)
            np.save(os.path.join(out_dir, name), points)

            if i % 100 == 0:
                print("Having processed %dth/%d image" % (i, dataset_length))




