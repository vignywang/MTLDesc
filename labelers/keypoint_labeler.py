#
# Created by ZhangYuyang on 2020/8/31
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
from data_utils import get_dataset
from utils.utils import spatial_nms
from utils.utils import draw_image_keypoints


class _BaseLabeler(object):

    def __init__(self, **config):
        self.config = config

        if torch.cuda.is_available():
            print('gpu is available, set device to cuda!')
            self.device = torch.device('cuda:0')
        else:
            print('gpu is not available, set device to cpu!')
            self.device = torch.device('cpu')
        self.homography = HomographyAugmentation()

        print('Generating dataset {}'.format(self.config['dataset']))
        self.dataset = get_dataset(self.config['dataset'])(**self.config)

        # 初始化模型
        model = SuperPointNetFloat()
        # 从预训练的模型中恢复参数
        model_dict = model.state_dict()
        pretrain_dict = torch.load(self.config['ckpt_path'], map_location=self.device)
        model_dict.update(pretrain_dict)
        model.load_state_dict(model_dict)
        model.to(self.device)
        self.model = model

        self.out_root = self.config['output_root']

        if not os.path.exists(self.out_root):
            os.mkdir(self.out_root)
        print('The train_out_dir is: %s' % self.out_root)

    def run(self):
        start_time = time.time()
        print("*****************************************************")
        print("Generating pseudo-ground truth via model {}".format(self.config['ckpt_path']))

        self._generate_pseudo_ground_truth()

        print("*****************************************************")
        print("Generating pseudo-ground truth done. Takes {:.3f} h".format((time.time()-start_time)/3600))

    def _generate_pseudo_ground_truth(self, *args, **kwargs):
        raise NotImplementedError

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


class Cocostuff10kLabeler(_BaseLabeler):

    def __init__(self, **config):
        super(Cocostuff10kLabeler, self).__init__(**config)

    def _generate_pseudo_ground_truth(self):
        self.model.eval()

        for i, data in tqdm(enumerate(self.dataset), total=len(self.dataset)):
            image_color = data['image_color']
            image = data['image_gray']
            label = data['label']
            image_id = data['image_id']
            h, w = image.shape

            # resize to 320x320
            l = 224
            min_size = min(h, w)
            image_color = cv.resize(image_color, None, None, fx=l/min_size, fy=l/min_size, interpolation=cv.INTER_LINEAR)
            image = cv.resize(image, None, None, fx=l/min_size, fy=l/min_size, interpolation=cv.INTER_LINEAR)
            label = cv.resize(label, None, None, fx=l/min_size, fy=l/min_size, interpolation=cv.INTER_LINEAR)
            h, w = image.shape

            if h > l:
                half = int((h-l)/2)
                image_color = image_color[half: half+l, :, :].copy()
                image = image[half: half+l, :].copy()
                label = label[half: half+l, :].copy()
            elif w > l:
                half = int((h - l) / 2)
                image_color = image_color[:, half: half + l, :].copy()
                image = image[:, half: half + l].copy()
                label = label[:, half: half + l].copy()

            images = []
            inv_homographies = []
            images.append(image)
            inv_homographies.append(np.eye(3))
            for j in range(self.config['adaption_num']):
                homography = self.homography.sample(height=l, width=l)
                inv_homography = np.linalg.inv(homography)
                transformed_image = cv.warpPerspective(image, homography, (l, l),
                                                       flags=cv.INTER_LINEAR)
                images.append(transformed_image)
                inv_homographies.append(inv_homography)

            # 送入网络做统一计算
            batched_image = torch.from_numpy(np.stack(images, axis=0)).unsqueeze(dim=1).to(torch.float)
            batched_image_list = [batched_image]
            if self.config['adaption_num'] > 50:
                batched_image_list = torch.split(batched_image, [50, self.config['adaption_num']-49], dim=0)

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
            for j in range(self.config['adaption_num']+1):
                transformed_prob = cv.warpPerspective(prob[j], inv_homographies[j], (l, l),
                                                      flags=cv.INTER_LINEAR)
                transformed_count = cv.warpPerspective(count, inv_homographies[j], (l, l),
                                                       flags=cv.INTER_NEAREST)
                probs.append(transformed_prob)
                counts.append(transformed_count)

            probs = np.stack(probs, axis=2)  # [h,w,n+1]
            counts = np.stack(counts, axis=2)
            probs = np.sum(probs, axis=2)
            counts = np.sum(counts, axis=2)
            probs = probs/counts

            torch_probs = torch.from_numpy(probs).unsqueeze(dim=0).unsqueeze(dim=0)
            final_probs = spatial_nms(torch_probs, int(self.config['nms_threshold']*2+1)).detach().cpu().numpy()[0, 0]

            satisfied_idx = np.where(final_probs > self.config['detection_threshold'])
            ordered_satisfied_idx = np.argsort(final_probs[satisfied_idx])[::-1]  # 降序
            if len(ordered_satisfied_idx) < self.config['top_k']:
                points = np.stack((satisfied_idx[0][ordered_satisfied_idx],
                                   satisfied_idx[1][ordered_satisfied_idx]), axis=1)
            else:
                points = np.stack((satisfied_idx[0][:self.config['top_k']],
                                   satisfied_idx[1][:self.config['top_k']]), axis=1)

            # debug hyper parameters use
            # image_point = draw_image_keypoints(image, points)
            # cv.imwrite('/home/yuyang/tmp/adaption/%03d.jpg' % i, image_point)

            cv.imwrite(os.path.join(self.out_root, image_id + '.jpg'), image_color)
            np.save(os.path.join(self.out_root, image_id + '_point'), points)
            np.save(os.path.join(self.out_root, image_id + '_label'), label)



