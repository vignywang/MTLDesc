# 
# Created by ZhangYuyang on 2019/10/22
#
import time
import random

import torch
from torch.utils.data import DataLoader
import numpy as np

from nets.superpoint_net import SuperPointNetFloat
from data_utils.coco_dataset import COCOSuperPointStatisticDataset
from data_utils.synthetic_dataset import SyntheticSuperPointStatisticDataset


def setup_seed():
    # make the result reproducible
    torch.manual_seed(3928)
    torch.cuda.manual_seed_all(2342)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(2933)
    random.seed(2312)


class PointStatistic(object):

    def __init__(self):
        self.total_segment_num = []  # 按10%一段进行分段统计各个段内的数目
        self._tmp_total_segment_num = []
        self._average = None
        self._std = None

    def reset(self):
        self.total_segment_num = []
        self._tmp_total_segment_num = []
        self._average = None
        self._std = None

    def update(self, dist):
        valid_dist = dist[dist <= 2.]
        max_dist = np.max(valid_dist)
        min_dist = np.min(valid_dist)

        inc_dist = (max_dist - min_dist) / 10
        segment_num = []
        for i in range(10):
            lower_bound = min_dist + i * inc_dist
            upper_bound = min_dist + (i + 1) * inc_dist
            cur_mask = (dist >= lower_bound) & (dist <= upper_bound)
            cur_num = np.sum(cur_mask)
            segment_num.append(cur_num)
        segment_num = np.stack(segment_num)
        self.total_segment_num.append(segment_num)

    @property
    def average(self):
        self._tmp_total_segment_num = np.stack(self.total_segment_num, axis=0)  # [m,10]
        self._average = np.mean(self._tmp_total_segment_num, axis=0)
        return self._average

    @property
    def std(self):
        if self._average is None:
            assert False
        self._std = np.sqrt(np.mean(np.square(self._tmp_total_segment_num - self._average[np.newaxis, :]), axis=0))
        return self._std


class StatisticParameters(object):

    def __init__(self):

        self.height = 240
        self.width = 320
        self.batch_size = 1
        self.ckpt_file = "/home/zhangyuyang/project/development/MegPoint/superpoint_ckpt/good_results/superpoint_triplet_0.0010_24_3_scale/model_49.pt"

        self.synthetic_dataset_dir = "/data/MegPoint/dataset/synthetic"
        self.coco_dataset_dir = "/data/MegPoint/dataset/coco"
        self.coco_pseudo_idx = "0"

        self.data_type = ["coco"]

        # homography & photometric relating params using in training
        self.homography_params = {
            'patch_ratio': 0.8,  # 0.8,  # 0.9,
            'perspective_amplitude_x': 0.3,  # 0.2,  # 0.1,
            'perspective_amplitude_y': 0.3,  # 0.2,  # 0.1,
            'scaling_sample_num': 5,
            'scaling_amplitude': 0.2,
            'translation_overflow': 0.05,
            'rotation_sample_num': 25,
            'rotation_max_angle': np.pi/3,  # np.pi/2.,  # np.pi/3,
            'do_perspective': True,
            'do_scaling': True,
            'do_rotation': True,
            'do_translation': True,
            'allow_artifacts': True
        }

        self.photometric_params = {
            'gaussian_noise_mean': 0,
            'gaussian_noise_std': 5,
            'speckle_noise_min_prob': 0,
            'speckle_noise_max_prob': 0.0035,
            'brightness_max_abs_change': 15,
            'contrast_min': 0.5,  # 0.7,
            'contrast_max': 1.5,  # 1.3,
            'shade_transparency_range': (-0.5, 0.5),
            'shade_kernel_size_range': (100, 150),
            'shade_nb_ellipese': 20,
            'motion_blur_max_kernel_size': 3,
            'do_gaussian_noise': True,
            'do_speckle_noise': True,
            'do_random_brightness': True,
            'do_random_contrast': True,
            'do_shade': True,
            'do_motion_blur': True
        }


class StatisticTools(object):

    def __init__(self, params):
        if torch.cuda.is_available():
            print('gpu is available, set device to cuda !')
            self.device = torch.device('cuda:0')
        else:
            print('gpu is not available, set device to cpu !')
            self.device = torch.device('cpu')

        self.model = SuperPointNetFloat()
        self.load_model_params(params.ckpt_file)
        self.model = self.model.to(self.device)
        self.data_type = params.data_type

        if self.data_type == "coco":
            print("Test COCO Dataset.")
            self.dataset = COCOSuperPointStatisticDataset(params)
        elif self.data_type == "synthetic":
            print("Tset Synthetic Dataset")
            self.dataset = SyntheticSuperPointStatisticDataset(params)
        else:
            assert False

        self.data_loader = DataLoader(
            self.dataset,
            batch_size=params.batch_size,
            # shuffle=False,
            shuffle=True,
            num_workers=4,
            drop_last=False
        )

    def load_model_params(self, ckpt_file):
        print("Load pretrained model %s" % ckpt_file)
        model_dict = self.model.state_dict()
        pretrained_dirc = torch.load(ckpt_file, map_location=self.device)
        model_dict.update(pretrained_dirc)
        self.model.load_state_dict(model_dict)

    def statistic(self, total_count=2000):
        self.model.eval()

        if total_count < 0:
            total_count = 90000
        point_st = PointStatistic()
        non_point_st = PointStatistic()
        start_time = time.time()
        count = 0
        for i, data in enumerate(self.data_loader):
            image = data["image"].to(self.device)
            point_mask = data["point_mask"].to(self.device)
            warped_image = data["warped_image"].to(self.device)
            matched_valid = data["matched_valid"].to(self.device)

            image_pair = torch.cat((image, warped_image), dim=0)
            _, desp_pair, *_ = self.model(image_pair)

            desp_0, desp_1 = torch.chunk(desp_pair, 2, dim=0)

            bt, dim, h, w = desp_0.shape

            desp_0 = torch.reshape(desp_0, (bt, dim, h * w)).transpose(1, 2)  # [bt,h*w,dim]
            desp_1 = torch.reshape(desp_1, (bt, dim, h * w))

            cos_similarity = torch.matmul(desp_0, desp_1)
            dist = torch.sqrt(2. * (1. - cos_similarity) + 1e-4)  # [bt,h*w,h*w]

            valid_interest_point_mask = point_mask * matched_valid  # [bt,h*w]
            valid_non_interest_point_mask = (1 - point_mask) * matched_valid

            for j in range(bt):
                valid_point_idx = torch.nonzero(valid_interest_point_mask[j])
                valid_non_point_idx = torch.nonzero(valid_non_interest_point_mask[j])

                point_dist = torch.gather(
                    dist[j], dim=0, index=valid_point_idx.repeat((1, h*w))).detach().cpu().numpy()
                non_point_dist = torch.gather(
                    dist[j], dim=0, index=valid_non_point_idx.repeat((1, h*w))).detach().cpu().numpy()

                point_num = point_dist.shape[0]
                non_point_num = non_point_dist.shape[0]

                for k in range(point_num):
                    point_st.update(point_dist[k])

                for k in range(non_point_num):
                    non_point_st.update(non_point_dist[k])

            count += bt
            if count % 100 == 0:
                print("Having computed %d samples, takes %.4f/step" % (count, (time.time()-start_time)/100))
                start_time = time.time()

            if count >= total_count:
                break

        print("Having computed total %d samples of %s" % (count, self.data_type))
        print(point_st.average)
        print(point_st.std)
        print("--------------------")
        print(non_point_st.average)
        print(non_point_st.std)


if __name__ == "__main__":
    # setup_seed()
    params = StatisticParameters()
    # params.data_type = "synthetic"
    params.data_type = "coco"
    statistic_tools = StatisticTools(params)
    statistic_tools.statistic(total_count=10)
































