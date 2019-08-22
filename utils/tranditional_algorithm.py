# 
# Created by ZhangYuyang on 2019/8/22
#
import cv2 as cv
import numpy as np


class FAST(object):

    def __init__(self, threshold=10, top_k=300):
        self.top_k = top_k
        self.fast = cv.FastFeatureDetector_create(threshold=threshold, nonmaxSuppression=True)
        # self.fast = cv.FastFeatureDetector_create(nonmaxSuppression=True)

    def detect(self, image):
        cv_kp_set = self.fast.detect(image)
        if len(cv_kp_set) == 0:
            return np.empty((0, 2))
        response_set = []
        kp_set = []
        for kp in cv_kp_set:
            response_set.append(kp.response)
            kp_set.append(np.array((kp.pt[1], kp.pt[0])))
        response_array = np.stack(response_set, axis=0)
        sorted_idx = np.argsort(response_array)[::-1]
        if sorted_idx.shape[0] >= self.top_k:
            sorted_idx = sorted_idx[:self.top_k]
        final_kp_set = []
        for idx in sorted_idx:
            final_kp_set.append(kp_set[idx])
        kp_array = np.stack(final_kp_set, axis=0)

        return kp_array


if __name__ == "__main__":
    test_image_dir = '/data/MegPoint/dataset/hpatch/i_ajuntament/1.ppm'
    test_image = cv.imread(test_image_dir, cv.IMREAD_GRAYSCALE)
    fast = FAST()
    key_point = fast.detect(test_image)
    a = 3








