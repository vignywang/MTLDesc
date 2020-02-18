# 
# Created by ZhangYuyang on 2020/1/8
#
import os
import shutil
import argparse

import torch
import numpy as np
import cv2 as cv
import torch.nn.functional as f

from nets.megpoint_net import resnet18
from nets.megpoint_net import resnet34

from nets.megpoint_net import resnet18_s0s2s3s4
from nets.megpoint_net import resnet18_s0s2s3s4_256
from nets.megpoint_net import resnet18_s0s2s3s4_512
from nets.megpoint_net import resnet18_s0s2s3s4_auxiliary

from nets.megpoint_net import MegPointShuffleHeatmapOld
from nets.superpoint_net import SuperPointNetFloat
from evaluation.aachen_dataset import AachenDataset
from utils.utils import spatial_nms
from utils.utils import Matcher
from utils.utils import NearestNeighborRatioMatcher
from data_utils.dataset_tools import draw_image_keypoints


class FeatureGenerator(object):
    """
    用于提取特定数据集中图像的关键点以及对应的描述子
    """
    def __init__(self, params):
        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")
        else:
            self.device = torch.device("cpu:0")

        self.detection_threshold = params.detection_threshold
        self.top_k = params.top_k

        self._initialize_model(params.model_type, params.detector_ckpt_file, params.desp_ckpt_file)
        self._initialize_dataset(params.dataset_root)
        self._initialize_matcher()

    def _initialize_model(self, model_type, detector_ckpt_file, desp_ckpt_file):
        """
        初始化模型
        """
        if detector_ckpt_file is None or desp_ckpt_file is None:
            print("Please input correct ckpt_file instead of None.")
            assert False

        # print("Initialize model of resnet18_all")
        # model = resnet18_all().to(self.device)
        # print("Initialize model of half_resnet18_all")
        # model = half_resnet18_all().to(self.device)

        # 初始化检测模型
        print("Initialize model of magicleap superpoint")
        detector = SuperPointNetFloat().to(self.device)
        # print("Initialize model of MegPointShuffleHeatmapOld.")
        # detector = MegPointShuffleHeatmapOld().to(self.device)
        self.detector = self._restore_model_params(detector, detector_ckpt_file)

        # 初始化描述子模型
        if model_type == "resnet18":
            print("Initialize model of resnet18")
            descriptor = resnet18().to(self.device)
        elif model_type == "resnet34":
            print("Initialize model of resnet34")
            descriptor = resnet34().to(self.device)

        elif model_type == "resnet18_s0s2s3s4":
            print("Initialize model of resnet18_s0s2s3s4")
            descriptor = resnet18_s0s2s3s4().to(self.device)
        elif model_type == "resnet18_s0s2s3s4_256":
            print("Initialize model of resnet18_s0s2s3s4_256")
            descriptor = resnet18_s0s2s3s4_256().to(self.device)
        elif model_type == "resnet18_s0s2s3s4_512":
            print("Initialize model of resnet18_s0s2s3s4_512")
            descriptor = resnet18_s0s2s3s4_512().to(self.device)
        elif model_type == "resnet18_s0s2s3s4_auxiliary":
            print("Initialize model of resnet18_s0s2s3s4_auxiliary")
            descriptor = resnet18_s0s2s3s4_auxiliary().to(self.device)

        else:
            print("Unrecognized model_type: %s" % model_type)
            assert False
        self.descriptor = self._restore_model_params(descriptor, desp_ckpt_file)

    def _initialize_dataset(self, dataset_dir):
        """
        初始化数据集
        """
        print("Initialize dataset from %s" % dataset_dir)
        self.dataset = AachenDataset(dataset_dir)
        self.dataset_length = len(self.dataset)

    def _initialize_matcher(self):
        print("Initialize nearest matcher.")
        # self.matcher = Matcher()
        self.matcher = NearestNeighborRatioMatcher(0.9)

    def _restore_model_params(self, model, ckpt_file):
        # 读取参数
        print("Restore model params from %s" % ckpt_file)
        model_dict = model.state_dict()
        pretrain_dict = torch.load(ckpt_file, map_location=self.device)
        model_dict.update(pretrain_dict)
        model.load_state_dict(model_dict)
        return model

    def run(self):
        """
        启动模型，对数据集中每一张图像估计关键点和对应的描述子，
        关键点和对应描述子以npz的格式存在与图像相同的路径下，方法名作为后缀结尾，例如1.jpg -> 1.jpg.d2-net
        """

        for i, data in enumerate(self.dataset):
            img = data["img"]
            img_dir = data["img_dir"]

            height, width, _ = img.shape
            point, desp = self.generate_feature(img_dir)
            # point, desp = self.generate_feature_magicleap(img)
            point_num = point.shape[0]

            board_min = np.array((0, 0), dtype=np.float32)
            board_max = np.array((height-1, width-1), dtype=np.float32)
            point = np.clip(point, board_min, board_max)

            if point_num <= 4:
                print("skip this image because there's little point!")
                return None

            output_dir = img_dir + ".galaxy"
            np.savez(output_dir, keypoints=point[:, ::-1], descriptors=desp)

            shutil.move(output_dir + ".npz", output_dir)

            if i % 100 == 0:
                print("Having precessed %04d/%04d images." % (i, self.dataset_length))

            # debug use
            # debug_img = draw_image_keypoints(img, point, show=False)
            # debug_img = cv.resize(debug_img, None, None, fx=0.5, fy=0.5, interpolation=cv.INTER_LINEAR)
            # cv.imshow("debug_img", debug_img)
            # cv.waitKey()

    def match_image_pair(self, img_0_dir, img_1_dir, ratio, show=True):
        """
        匹配两幅图像，并选择是否将匹配结果显示出来
        Args:
            img_0: [h0,w0] 灰度图像
            img_1: [h1,w1] 灰度图像
            show: 选择是否即时显示出来
        Returns:
            matched_img
        """
        img_0 = cv.imread(img_0_dir)
        img_1 = cv.imread(img_1_dir)
        point_0, desp_0 = self.generate_feature(img_0_dir)
        # point_0, desp_0 = self.generate_feature_magicleap(img_0_dir)
        point_1, desp_1 = self.generate_feature(img_1_dir)
        # point_1, desp_1 = self.generate_feature_magicleap(img_1_dir)
        match_list = self.matcher(point_0, desp_0, point_1, desp_1)
        cv_point_0, cv_point_1, cv_match_list = self._convert_match2cv(match_list[0], match_list[1], ratio)

        # sift = cv.xfeatures2d_SIFT.create(2000)
        # cv_point_0, desp_0 = sift.detectAndCompute(img_0, None)
        # cv_point_1, desp_1 = sift.detectAndCompute(img_1, None)
        # matcher = cv.FlannBasedMatcher_create()
        # cv_match_list = matcher.match(desp_0, desp_1)

        match_img = cv.drawMatches(img_0, cv_point_0, img_1, cv_point_1, cv_match_list, None)
        match_img = cv.resize(match_img, None, None, fx=0.5, fy=0.5, interpolation=cv.INTER_LINEAR)

        cv.imwrite("/home/zhangyuyang/tmp_images/aachen_contrast/1.jpg", match_img)
        cv.imshow("match_img", match_img)
        cv.waitKey()

    def generate_feature(self, img_dir):
        """
        获取一幅图像的特征点及其对应的描述子
        Args:
            img_dir: 图像地址
        Returns:
            point: [n,2] 特征点
            descriptor: [n,128] 描述子
        """
        self.detector.eval()
        self.descriptor.eval()

        img = cv.imread(img_dir)[:, :, ::-1].copy()
        img_gray = cv.imread(img_dir, cv.IMREAD_GRAYSCALE)

        org_h, org_w, _ = img.shape
        do_scale = False
        if org_h > 1000 or org_w > 1000:
            h = int(org_h / 1.5)
            w = int(org_w / 1.5)
            # scale = np.array((org_h / h, org_w / w), dtype=np.float32)
            do_scale = True
        else:
            h = org_h
            w = org_w
            # scale = np.array((1, 1), dtype=np.float32)

        img = torch.from_numpy(img).to(torch.float).permute((2, 0, 1)).unsqueeze(dim=0).to(self.device)
        img = img * 2. / 255. - 1.

        img_gray = torch.from_numpy(img_gray).to(torch.float).unsqueeze(dim=0).unsqueeze(dim=0).to(self.device)
        img_gray = img_gray / 255.
        # img_gray = img_gray * 2. / 255. - 1.

        if do_scale:
            img = f.interpolate(img, size=(h, w), mode="bilinear", align_corners=True)

        # detector
        _, _, prob, _ = self.detector(img_gray)
        prob = f.pixel_shuffle(prob, 8)
        prob = spatial_nms(prob)
        # heatmap, _ = self.detector(img_gray)
        # heatmap = torch.sigmoid(heatmap)
        # prob = spatial_nms(heatmap)
        # 得到对应的预测点
        prob = prob.detach().cpu().numpy()
        prob = prob[0, 0]
        point, point_num = self._generate_predict_point(prob, self.detection_threshold, self.top_k)  # [n,2]

        desp_point = torch.from_numpy(point[:, ::-1].copy()).to(torch.float).to(self.device)
        # 归一化采样坐标到[-1,1]
        desp_point = desp_point * 2. / torch.tensor((org_w-1, org_h-1), dtype=torch.float, device=self.device) - 1
        desp_point = desp_point.unsqueeze(dim=0).unsqueeze(dim=2)  # [1,n,1,2]

        # descriptor
        desp = self.descriptor(img, desp_point)[0, :, :].detach().cpu().numpy()

        # 得到点对应的描述子
        # desp = self._generate_combined_descriptor(point, c1, c2, c3, c4, org_h, org_w)

        # if do_scale:
        #     point *= scale

        return point, desp

    def generate_feature_magicleap(self, img_dir):
        """
        用superpoint获取一幅图像的特征点及其对应的描述子
        Args:
            img: [h,w] 灰度图像
        Returns:
            point: [n,2] 特征点
            descriptor: [n,128] 描述子
        """
        self.detector.eval()

        img = cv.imread(img_dir, cv.IMREAD_GRAYSCALE)

        org_h, org_w = img.shape
        do_scale = False
        if org_h > 2000 or org_w > 2000:
            h = int(org_h // 2)
            w = int(org_w // 2)
            scale = np.array((org_h / h, org_w / w), dtype=np.float32)
            do_scale = True
        else:
            h = org_h
            w = org_w
            scale = np.array((1, 1), dtype=np.float32)

        img = torch.from_numpy(img).to(torch.float).unsqueeze(dim=0).unsqueeze(dim=0).to(self.device)
        img = img / 255.
        if do_scale:
            img = f.interpolate(img, size=(h, w), mode="bilinear", align_corners=True)

        self.detector.eval()
        _, desp, prob, _ = self.detector(img)
        prob = f.pixel_shuffle(prob, 8)
        prob = spatial_nms(prob)

        desp = desp.detach().cpu().numpy()[0]
        prob = prob.detach().cpu().numpy()[0, 0]

        # 得到对应的预测点
        point, point_num = self._generate_predict_point(prob, top_k=self.top_k, detection_threshold=0.005)  # [n,2]

        # 得到点对应的描述子
        desp = self._generate_predict_descriptor(point, desp)

        if do_scale:
            point *= scale

        return point, desp

    @staticmethod
    def _convert_match2cv(first_point_list, second_point_list, sample_ratio=1.0):
        cv_first_point = []
        cv_second_point = []
        cv_matched_list = []

        assert len(first_point_list) == len(second_point_list)

        inc = 1
        if sample_ratio < 1:
            inc = int(1.0 / sample_ratio)

        count = 0
        if len(first_point_list) > 0:
            for j in range(0, len(first_point_list), inc):
                cv_point = cv.KeyPoint()
                cv_point.pt = tuple(first_point_list[j][::-1])
                cv_first_point.append(cv_point)

                cv_point = cv.KeyPoint()
                cv_point.pt = tuple(second_point_list[j][::-1])
                cv_second_point.append(cv_point)

                cv_match = cv.DMatch()
                cv_match.queryIdx = count
                cv_match.trainIdx = count
                cv_matched_list.append(cv_match)

                count += 1

        return cv_first_point, cv_second_point, cv_matched_list

    @staticmethod
    def _generate_predict_point(prob, detection_threshold, top_k):

        point_idx = np.where(prob > detection_threshold)

        if len(point_idx[0]) == 0 or len(point_idx[1]) == 0:
            point = np.empty((0, 2))
            return point, 0

        prob = prob[point_idx]
        sorted_idx = np.argsort(prob)[::-1]
        if sorted_idx.shape[0] >= top_k:
            sorted_idx = sorted_idx[:top_k]

        point = np.stack(point_idx, axis=1)  # [n,2]
        top_k_point = []
        for idx in sorted_idx:
            top_k_point.append(point[idx])

        point = np.stack(top_k_point, axis=0).astype(np.float32)
        point_num = point.shape[0]

        return point, point_num

    def _generate_combined_descriptor(self, point, c1, c2, c3, c4, height, width):
        """
        用多层级的组合特征构造描述子
        Args:
            point: [n,2] 顺序是y,x
            c1,c2,c3,c4: 分别对应resnet4个block输出的特征,batchsize都是1
        Returns:
            desp: [n,dim]
        """
        point = torch.from_numpy(point[:, ::-1].copy()).to(torch.float).to(self.device)
        # 归一化采样坐标到[-1,1]
        point = point * 2. / torch.tensor((width-1, height-1), dtype=torch.float, device=self.device) - 1
        point = point.unsqueeze(dim=0).unsqueeze(dim=2)  # [1,n,1,2]

        c1_feature = f.grid_sample(c1, point, mode="bilinear")[0, :, :, 0].transpose(0, 1)
        c2_feature = f.grid_sample(c2, point, mode="bilinear")[0, :, :, 0].transpose(0, 1)
        c3_feature = f.grid_sample(c3, point, mode="bilinear")[0, :, :, 0].transpose(0, 1)
        c4_feature = f.grid_sample(c4, point, mode="bilinear")[0, :, :, 0].transpose(0, 1)
        desp = torch.cat((c1_feature, c2_feature, c3_feature, c4_feature), dim=1)
        # desp = torch.cat((c1_feature, c2_feature, c3_feature), dim=1)
        # desp = torch.cat((c1_feature, c2_feature), dim=1)
        desp = desp / torch.norm(desp, dim=1, keepdim=True)

        desp = desp.detach().cpu().numpy()

        return desp

    @staticmethod
    def _generate_predict_descriptor(point, desp):
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


if __name__ == "__main__":

    class Parameters:

        def __init__(self):
            self.dataset_root = "/home/zhangyuyang/data/aachen/Aachen-Day-Night/images/images_upright"

            self.detector_ckpt_file = "/home/zhangyuyang/project/development/MegPoint/magicpoint_ckpt/good_results/superpoint_magicleap.pth"

            # desp_ckpt_file = "/home/zhangyuyang/model_megadepth_07.pt"
            # desp_ckpt_file = "/home/zhangyuyang/model_megadepth_c4_14.pt"
            # desp_ckpt_file = "/home/zhangyuyang/model_megadepth_extract_00.pt"
            # desp_ckpt_file = "/home/zhangyuyang/model_megadepth_half_08.pt"
            # desp_ckpt_file = "/home/zhangyuyang/model_megadepth_11.pt"
            # desp_ckpt_file = "/home/zhangyuyang/remote_model/megadepth_only_descriptor_resnet_extractor_0.0010_8/model_59.pt"
            self.desp_ckpt_file = "/home/zhangyuyang/remote_model/megadepth_rank_0.0010_8/model_09.pt"

            self.desp_ckpt_root = "/home/zhangyuyang/remote_model"

            self.detection_threshold = 0.005  # for magicleap model
            self.top_k = 5000

            self.model_type = "resnet18"

    parse = argparse.ArgumentParser()
    parse.add_argument("--desp_ckpt_file", type=str, default="")
    parse.add_argument("--model_type", type=str, default="resnet18")
    args = parse.parse_args()

    params = Parameters()
    params.desp_ckpt_file = os.path.join(params.desp_ckpt_root, args.desp_ckpt_file)
    params.model_type = args.model_type

    feature_generator = FeatureGenerator(params)
    feature_generator.run()

    # debug use
    # img_0 = "/home/zhangyuyang/data/aachen/Aachen-Day-Night/images/images_upright/db/9.jpg"
    # img_1 = "/home/zhangyuyang/data/aachen/Aachen-Day-Night/images/images_upright/db/11.jpg"
    # img_0 = "/home/zhangyuyang/data/aachen/Aachen-Day-Night/images/images_upright/db/2125.jpg"
    # img_1 = "/home/zhangyuyang/data/aachen/Aachen-Day-Night/images/images_upright/query/night/nexus5x/IMG_20161227_192213.jpg"
    # img_0 = "/data/MegPoint/dataset/hpatch/v_dirtywall/1.ppm"
    # img_1 = "/data/MegPoint/dataset/hpatch/v_dirtywall/4.ppm"

    # feature_generator.match_image_pair(img_0, img_1, 0.25)





