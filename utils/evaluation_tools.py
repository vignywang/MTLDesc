#
# Created by ZhangYuyang on 2019/8/13
#
import numpy as np
import matplotlib.pyplot as plt


class HomoAccuracyCalculator(object):

    def __init__(self, epsilon, height, width):
        self.height = height
        self.width = width
        self.epsilon = epsilon
        self.sum_accuracy = 0
        self.sum_sample_num = 0
        self.corner = self._generate_corner()

    def reset(self):
        self.sum_accuracy = 0
        self.sum_sample_num = 0

    def average(self):
        return self.sum_accuracy / self.sum_sample_num

    def update(self, pred_homography, gt_homography):
        warped_corner_by_pred = np.matmul(pred_homography, self.corner[:, :, np.newaxis])[:, :, 0]
        warped_corner_by_gt = np.matmul(gt_homography, self.corner[:, :, np.newaxis])[:, :, 0]
        warped_corner_by_pred = warped_corner_by_pred / warped_corner_by_pred[:, 2:3]
        warped_corner_by_gt = warped_corner_by_gt / warped_corner_by_gt[:, 2:3]
        diff = np.linalg.norm((warped_corner_by_pred-warped_corner_by_gt), axis=1, keepdims=False)
        diff = np.mean(diff)
        accuracy = (diff <= self.epsilon).astype(np.float)
        self.sum_accuracy += accuracy
        self.sum_sample_num += 1

    def _generate_corner(self):
        pt_00 = np.array((0, 0, 1), dtype=np.float)
        pt_01 = np.array((0, self.height-1, 1), dtype=np.float)
        pt_10 = np.array((self.width-1, 0, 1), dtype=np.float)
        pt_11 = np.array((self.width-1, self.height-1, 1), dtype=np.float)
        corner = np.stack((pt_00, pt_01, pt_10, pt_11), axis=0)
        return corner


class MeanMatchingAccuracy(object):

    def __init__(self, epsilon):
        self.epsilon = epsilon
        self.sum_accuracy = 0
        self.sum_sample_num = 0

    def reset(self):
        self.sum_accuracy = 0
        self.sum_sample_num = 0

    def update(self, gt_homography, matched_point):
        """
        计算单个样本对的匹配准确度
        Args:
            gt_homography: 该样本对的单应变换真值
            matched_point: List or Array.
            matched_point[0]是source image上的点,顺序为(x,y)
            matched_point[1]是target image上的点,顺序为(x,y),
        """
        inv_homography = np.linalg.inv(gt_homography)
        src_point, tgt_point = matched_point[0], matched_point[1]
        num_matched = np.shape(src_point)[0]
        ones = np.ones((num_matched, 1), dtype=np.float)

        homo_src_point = np.concatenate((src_point, ones), axis=1)
        homo_tgt_point = np.concatenate((tgt_point, ones), axis=1)

        project_src_point = np.matmul(gt_homography, homo_src_point[:, :, np.newaxis])[:, :, 0]
        project_tgt_point = np.matmul(inv_homography, homo_tgt_point[:, :, np.newaxis])[:, :, 0]

        project_src_point = project_src_point[:, :2] / project_src_point[:, 2:3]
        project_tgt_point = project_tgt_point[:, :2] / project_tgt_point[:, 2:3]

        dist_src = np.linalg.norm(tgt_point - project_src_point, axis=1)
        dist_tgt = np.linalg.norm(src_point - project_tgt_point, axis=1)

        correct_src = (dist_src <= self.epsilon)
        correct_tgt = (dist_tgt <= self.epsilon)
        correct = (correct_src & correct_tgt).astype(np.float)
        correct_ratio = np.mean(correct)
        self.sum_accuracy += correct_ratio
        self.sum_sample_num += 1

    def average(self):
        """
        Returns: 平均匹配准确度
        """
        return self.sum_accuracy/self.sum_sample_num


class RepeatabilityCalculator(object):

    def __init__(self, epsilon):
        self.epsilon = epsilon
        self.sum_repeatability = 0
        self.sum_sample_num = 0

    def reset(self):
        self.sum_repeatability = 0
        self.sum_sample_num = 0

    def update(self, point_0, point_1, homography):
        self.sum_repeatability += self.compute_one_sample_repeatability(point_0, point_1, homography)
        self.sum_sample_num += 1

    def average(self):
        average_repeatability = self.sum_repeatability/self.sum_sample_num
        return average_repeatability, self.sum_repeatability, self.sum_sample_num

    def compute_one_sample_repeatability(self, point_0, point_1, homography):
        inv_homography = np.linalg.inv(homography)

        num_0 = np.shape(point_0)[0]
        num_1 = np.shape(point_1)[0]
        one_0 = np.ones((num_0, 1), dtype=np.float)
        one_1 = np.ones((num_1, 1), dtype=np.float)

        # recover to the original size and flip the order (y,x) to (x,y)
        point_0 = point_0[:, ::-1]
        point_1 = point_1[:, ::-1]
        homo_point_0 = np.concatenate((point_0, one_0), axis=1)[:, :, np.newaxis]  # [n, 3, 1]
        homo_point_1 = np.concatenate((point_1, one_1), axis=1)[:, :, np.newaxis]

        # compute correctness from 0 to 1
        project_point_0 = np.matmul(homography, homo_point_0)
        project_point_0 = project_point_0[:, :2, 0] / project_point_0[:, 2:3, 0]
        correctness_0_1 = self.compute_correctness(project_point_0, point_1)

        # compute correctness from 1 to 0
        project_point_1 = np.matmul(inv_homography, homo_point_1)
        project_point_1 = project_point_1[:, :2, 0] / project_point_1[:, 2:3, 0]
        correctness_1_0 = self.compute_correctness(project_point_1, point_0)

        # compute repeatability
        total_point = np.shape(point_0)[0] + np.shape(point_1)[0]
        repeatability = (correctness_0_1 + correctness_1_0) / total_point
        return repeatability

    def compute_correctness(self, point_0, point_1):
        # compute the distance of two set of point
        # point_0: [n, 2], point_1: [m,2]
        point_0 = np.expand_dims(point_0, axis=1)  # [n, 1, 2]
        point_1 = np.expand_dims(point_1, axis=0)  # [1, m, 2]
        dist = np.linalg.norm(point_0 - point_1, axis=2)  # [n, m]

        min_dist = np.min(dist, axis=1, keepdims=False)  # [n]
        correctness = np.less_equal(min_dist, self.epsilon).astype(np.float)
        correctness = np.sum(correctness)

        return correctness


class mAPCalculator(object):

    def __init__(self):
        self.tp = []
        self.fp = []
        self.prob = []
        self.total_num = 0

    def reset(self):
        self.tp = []
        self.fp = []
        self.prob = []
        self.total_num = 0

    def update(self, org_prob, gt_point):
        tp, fp, prob, n_gt = self._compute_tp_fp(org_prob, gt_point)
        self.tp.append(tp)
        self.fp.append(fp)
        self.prob.append(prob)
        self.total_num += n_gt

    def compute_mAP(self):
        if len(self.tp) == 0:
            print("There has nothing to compute from! Please Check!")
            return
        tp = np.concatenate(self.tp)
        fp = np.concatenate(self.fp)
        prob = np.concatenate(self.prob)

        # 对整体进行排序
        sort_idx = np.argsort(prob)[::-1]
        tp = tp[sort_idx]
        fp = fp[sort_idx]
        prob = prob[sort_idx]

        # 进行累加计算
        tp_cum = np.cumsum(tp)
        fp_cum = np.cumsum(fp)
        recall = tp_cum / self.total_num
        precision = tp_cum / (tp_cum + fp_cum)
        prob = np.concatenate([[1], prob, [0]])
        recall = np.concatenate([[0], recall, [1]])
        precision = np.concatenate([[0], precision, [0]])
        mAP = np.sum(precision[1:] * (recall[1:] - recall[:-1]))

        test_data = np.stack((recall, precision, prob), axis=0)
        return mAP, test_data

    def plot_threshold_curve(self, test_data, curve_name, curve_dir):
        recall = test_data[0, 1:-1]
        precision = test_data[1, 1:-1]
        prob = test_data[2, 1:-1]

        tmp_idx = np.where(prob <= 0.15)
        recall = recall[tmp_idx]
        precision = precision[tmp_idx]
        prob = prob[tmp_idx]
        title = curve_name

        plt.figure(figsize=(10, 5))
        x_ticks = np.arange(0, 1, 0.01)
        y_ticks = np.arange(0, 1, 0.05)
        plt.title(title)
        plt.xticks(x_ticks)
        plt.yticks(y_ticks)
        plt.xlabel('probability threshold')
        plt.plot(prob, recall, label='recall')
        plt.plot(prob, precision, label='precision')
        plt.legend(loc='lower right')
        plt.grid()
        plt.savefig(curve_dir)

    @staticmethod
    def _compute_tp_fp(prob, gt_point, remove_zero=1e-4, distance_thresh=2):
        # 这里只能计算一个样本的tp以及fp，而不是一个batch
        assert len(np.shape(prob)) == 2

        mask = np.where(prob > remove_zero)
        # 留下满足满足要求的点
        prob = prob[mask]
        # 得到对应点的坐标, [n, 2]
        pred = np.array(mask).T

        sort_idx = np.argsort(prob)[::-1]
        prob = prob[sort_idx]
        pred = pred[sort_idx]

        # 得到每个点与真值点间的距离，最终得到[n,m]的距离表达式
        diff = np.expand_dims(pred, axis=1) - np.expand_dims(gt_point, axis=0)
        dist = np.linalg.norm(diff, axis=-1)
        matches = np.less_equal(dist, distance_thresh)

        tp = []
        matched = np.zeros(np.shape(gt_point)[0])
        for m in matches:
            correct = np.any(m)
            if correct:
                gt_idx = np.argmax(m)
                # 已匹配则为False
                tp.append(not matched[gt_idx])
                # 标记已匹配的点
                matched[gt_idx] = 1
            else:
                tp.append(False)
        tp = np.array(tp, bool)
        fp = np.logical_not(tp)
        n_gt = np.shape(gt_point)[0]

        return tp, fp, prob, n_gt
