# 
# Created by ZhangYuyang on 2019/11/13
#
import torch
import cv2 as cv
import numpy as np
import torch.nn.functional as f

from nets.megpoint_net import MegPointShuffleHeatmap
from nets.superpoint_net import SuperPointNetFloat
from utils.utils import Matcher
from utils.utils import spatial_nms
from data_utils.dataset_tools import draw_image_keypoints


class CvMatch(object):

    def __init__(self, query_idx, train_idx, distance):
        self.queryIdx = query_idx
        self.trainIdx = train_idx
        self.distance = distance


class BaseInference(object):

    def __init__(self, params):
        self.ckpt_file = params.ckpt_file
        self.top_k = params.top_k
        self.nms_threshold = params.nms_threshold
        self.detection_threshold = params.detection_threshold
        if torch.cuda.is_available():
            print('gpu is available, set device to cuda !')
            self.device = torch.device('cuda:0')
        else:
            print('gpu is not available, set device to cpu !')
            self.device = torch.device('cpu')

        self.model = None

        self._initialize_model()

    def _initialize_model(self):
        raise NotImplementedError

    def _load_model_params(self, ckpt_file):
        if ckpt_file is None:
            print("Please input correct checkpoint file dir!")
            assert False

        model_dict = self.model.state_dict()
        pretrain_dict = torch.load(ckpt_file, map_location=self.device)
        model_dict.update(pretrain_dict)
        self.model.load_state_dict(model_dict)

    def _generate_predict_point(self, prob, scale=None, top_k=0):
        point_idx = np.where(prob > self.detection_threshold)
        prob = prob[point_idx]
        sorted_idx = np.argsort(prob)[::-1]
        if sorted_idx.shape[0] >= top_k:
            sorted_idx = sorted_idx[:top_k]

        point = np.stack(point_idx, axis=1)  # [n,2]
        top_k_point = []
        for idx in sorted_idx:
            top_k_point.append(point[idx])
        point = np.stack(top_k_point, axis=0)
        point_num = point.shape[0]

        if scale is not None:
            point = point*scale
        return point, point_num

    def _generate_predict_descriptor(self, point, desp):
        point = torch.from_numpy(point).to(torch.float)  # 由于只有pytorch有gather的接口，因此将点调整为pytorch的格式
        desp = torch.from_numpy(desp)
        dim, h, w = desp.shape
        desp = torch.reshape(desp, (dim, -1))
        desp = torch.transpose(desp, dim0=1, dim1=0)  # [h*w,256]

        # 下采样
        scaled_point = point / 8
        point_y = scaled_point[:, 0:1]  # [n,1]
        point_x = scaled_point[:, 1:2]

        x0 = torch.floor(point_x)
        x1 = x0 + 1
        y0 = torch.floor(point_y)
        y1 = y0 + 1
        x_nearest = torch.round(point_x)
        y_nearest = torch.round(point_y)

        x0_safe = torch.clamp(x0, min=0, max=w-1)
        x1_safe = torch.clamp(x1, min=0, max=w-1)
        y0_safe = torch.clamp(y0, min=0, max=h-1)
        y1_safe = torch.clamp(y1, min=0, max=h-1)

        x_nearest_safe = torch.clamp(x_nearest, min=0, max=w-1)
        y_nearest_safe = torch.clamp(y_nearest, min=0, max=h-1)

        idx_00 = (x0_safe + y0_safe*w).to(torch.long).repeat((1, dim))
        idx_01 = (x0_safe + y1_safe*w).to(torch.long).repeat((1, dim))
        idx_10 = (x1_safe + y0_safe*w).to(torch.long).repeat((1, dim))
        idx_11 = (x1_safe + y1_safe*w).to(torch.long).repeat((1, dim))
        idx_nearest = (x_nearest_safe + y_nearest_safe*w).to(torch.long).repeat((1, dim))

        d_x = point_x - x0_safe
        d_y = point_y - y0_safe
        d_1_x = x1_safe - point_x
        d_1_y = y1_safe - point_y

        desp_00 = torch.gather(desp, dim=0, index=idx_00)
        desp_01 = torch.gather(desp, dim=0, index=idx_01)
        desp_10 = torch.gather(desp, dim=0, index=idx_10)
        desp_11 = torch.gather(desp, dim=0, index=idx_11)
        nearest_desp = torch.gather(desp, dim=0, index=idx_nearest)
        bilinear_desp = desp_00*d_1_x*d_1_y + desp_01*d_1_x*d_y + desp_10*d_x*d_1_y+desp_11*d_x*d_y

        # todo: 插值得到的描述子不再满足模值为1，强行归一化到模值为1，这里可能有问题
        condition = torch.eq(torch.norm(bilinear_desp, dim=1, keepdim=True), 0)
        interpolation_desp = torch.where(condition, nearest_desp, bilinear_desp)
        interpolation_norm = torch.norm(interpolation_desp, dim=1, keepdim=True)
        interpolation_desp = interpolation_desp/interpolation_norm

        return interpolation_desp.numpy()


class SuperPointInference(BaseInference):

    def __init__(self, params):
        super(SuperPointInference, self).__init__(params)

    def _initialize_model(self):
        print("Initialize the SuperPoint model with: %s" % self.ckpt_file)
        # self.model = SuperPointNetFloat()
        self.model = MegPointShuffleHeatmap()
        self._load_model_params(self.ckpt_file)
        self.model = self.model.to(self.device)

    def detect(self, image):
        """
        输入图像，输出图像检测的关键点及其对应描述子
        Args:
            image: [240,320]

        Returns:
            point: n个cv.KeyPoint
            desp: [n,256]
        """
        size = image.shape
        assert len(size) == 2

        self.model.eval()

        org_image = image
        image = torch.from_numpy(image).to(torch.float).to(self.device).unsqueeze(dim=0).unsqueeze(dim=0)
        image = image / 255. * 2. - 1

        result = self.model(image)
        heatmap = result[0]
        desp = result[1]

        prob = torch.sigmoid(heatmap)
        prob = spatial_nms(prob)

        desp = desp.detach().cpu().numpy()[0]
        prob = prob.detach().cpu().numpy()[0, 0]

        # 获取检测的点及数量
        point, point_num = self._generate_predict_point(prob, top_k=self.top_k)
        print("Having detected %d keypoint." % point_num)

        # debug use
        # draw_image_keypoints(org_image, point, show=True)

        # 获取对应的描述子
        desp = self._generate_predict_descriptor(point, desp)

        # 将点转换成opencv的格式
        cv_point = []
        for i in range(point_num):
            kpt = cv.KeyPoint()
            kpt.pt = (point[i, 1], point[i, 0])
            cv_point.append(kpt)

        return cv_point, desp

    def match(self, desp_0, desp_1, mode=0):
        """返回opencv格式的匹配结果"""
        dist_0_1 = self._compute_desp_dist(desp_0, desp_1)  # [n,m]
        dist_1_0 = dist_0_1.transpose((1, 0))  # [m,n]
        nearest_idx_0_1 = np.argmin(dist_0_1, axis=1)  # [n]
        nearest_idx_1_0 = np.argmin(dist_1_0, axis=1)  # [m]

        if mode == 0:
            second_nearest_idx_0_1 = np.argpartition(dist_0_1, kth=1, axis=1)[:, 1]
            matches = []
            for i, idx_0_1 in enumerate(nearest_idx_0_1):
                if i == nearest_idx_1_0[idx_0_1]:
                    s_idx_0_1 = second_nearest_idx_0_1[i]
                    m_dist = dist_0_1[i, idx_0_1]
                    sm_dist = dist_0_1[i, s_idx_0_1]
                    if m_dist / sm_dist >= 0.9:
                        continue
                    match = cv.DMatch()
                    match.queryIdx = i
                    match.trainIdx = int(idx_0_1)
                    match.distance = dist_0_1[i, idx_0_1]
                    matches.append(match)
            good_matches = matches

        if mode == 1:
            matches = []
            for i, idx_0_1 in enumerate(nearest_idx_0_1):
                if i == nearest_idx_1_0[idx_0_1]:
                    match = cv.DMatch()
                    match.queryIdx = i
                    match.trainIdx = int(idx_0_1)
                    match.distance = dist_0_1[i, idx_0_1]
                    matches.append(match)

            min_dist = 10000
            max_dist = 0
            for m in matches:
                dist = m.distance
                if dist < min_dist:
                    min_dist = dist
                if dist > max_dist:
                    max_dist = dist

            print("min_dist=%.4f, max_dist=%.4f" % (min_dist, max_dist))

            good_matches = []
            for m in matches:
                dist = m.distance
                if dist > 0.6:
                    continue
                else:
                    good_matches.append(m)

        return good_matches

    @staticmethod
    def _compute_desp_dist(desp_0, desp_1):
        # desp_0:[n,256], desp_1:[m,256]
        square_norm_0 = (np.linalg.norm(desp_0, axis=1, keepdims=True)) ** 2  # [n,1]
        square_norm_1 = (np.linalg.norm(desp_1, axis=1, keepdims=True).transpose((1, 0))) ** 2  # [1,m]
        xty = np.matmul(desp_0, desp_1.transpose((1, 0)))  # [n,m]
        dist = np.sqrt((square_norm_0 + square_norm_1 - 2 * xty + 1e-4))
        return dist


if __name__ == "__main__":
    class Parameters(object):

        def __init__(self):
            self.top_k = 1000
            self.nms_threshold = 4
            self.detection_threshold = 0.9

            # self.ckpt_file = "/home/zhangyuyang/project/development/MegPoint/magicpoint_ckpt/good_results/superpoint_magicleap.pth"
            self.ckpt_file = "/home/zhangyuyang/project/development/MegPoint/megpoint_ckpt/coco_weight_bce_01_0.0010_24/model_59.pt"


    params = Parameters()
    infer_model = SuperPointInference(params)

    # img_0 = cv.imread("/home/zhangyuyang/data/superpoint_test/0.jpg", cv.IMREAD_GRAYSCALE)
    img_0 = cv.imread("/home/zhangyuyang/data/superpoint_test/3.jpg", cv.IMREAD_GRAYSCALE)
    img_0 = cv.resize(img_0, (640, 480), cv.INTER_LINEAR)

    # img_1 = cv.imread("/home/zhangyuyang/data/superpoint_test/2.jpg", cv.IMREAD_GRAYSCALE)
    img_1 = cv.imread("/home/zhangyuyang/data/superpoint_test/4.jpg", cv.IMREAD_GRAYSCALE)
    img_1 = cv.resize(img_1, (640, 480), cv.INTER_LINEAR)

    point_0, desp_0 = infer_model.detect(img_0)
    point_1, desp_1 = infer_model.detect(img_1)

    matches = infer_model.match(desp_0, desp_1, 1)
    match_image = cv.drawMatches(img_0, point_0, img_1, point_1, matches, None, singlePointColor=(255, 255, 255))
    cv.imshow("match_image", match_image)
    cv.waitKey()
    cv.imwrite("/home/zhangyuyang/data/superpoint_test/matched_images/match_34.jpg", match_image)










