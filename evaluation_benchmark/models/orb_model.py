#
# Created by ZhangYuyang on 2020/2/25
#
import cv2 as cv
import numpy as np


class BasicORB(object):

    def __init__(self, logger, top_k=1000):
        self.logger = logger
        self.name = "orb"

        self.top_k = top_k
        self.logger.info("top_k=%d" % top_k)

        # 初始化superpoint
        self.logger.info("Initialize ORB")
        self.orb = cv.ORB_create(nfeatures=1000)

    def generate_feature(self, img):
        """
        获取一幅灰度图像对应的特征点及其描述子
        Args:
            img_dir: 图像地址
        Returns:
            point: [n,2] 特征点
            descriptor: [n,128] 描述子
        """
        # detect and compute
        keypoints_cv, desp = self.orb.detectAndCompute(img, None)

        # 将cv点转换为numpy格式的
        # point = self._convert_cv2pt(keypoints_cv)
        # point_num = point.shape[0]
        point_num = len(keypoints_cv)

        return keypoints_cv, desp, point_num

    @staticmethod
    def _convert_cv2pt(cv_point):
        point_list = []
        for i, cv_pt in enumerate(cv_point):
            pt = np.array((cv_pt.pt[1], cv_pt.pt[0]))  # y,x的顺序
            point_list.append(pt)
        point = np.stack(point_list, axis=0)
        return point

    def __call__(self, *args, **kwargs):
        raise NotImplementedError


class HPatchORB(BasicORB):
    """专用于hpatch测试的SuperPoint模型"""

    def __init__(self, logger, top_k=1000):
        super(HPatchORB, self).__init__(logger, top_k=top_k)

    def __call__(self, first_image, second_image, *args, **kwargs):
        first_point, first_desp, first_point_num = self.generate_feature(first_image)
        second_point, second_desp, second_point_num = self.generate_feature(second_image)

        return first_point, first_point_num, first_desp, second_point, second_point_num, second_desp


