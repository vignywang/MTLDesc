#
# Created by ZhangYuyang on 2019/8/13
#
import numpy as np
import matplotlib.pyplot as plt


class RepeatabilityCalculator(object):

    def __init__(self):
        self.sum_repeatability = 0
        self.sum_sample_num = 0

    def reset(self):
        self.sum_repeatability = 0
        self.sum_sample_num = 0

    def compute_one_sample_repeatability(self, prob_0, prob_1, threshold, epsilon, scale_0, scale_1, homography):
        inv_homography = np.linalg.inv(homography)

        point_0 = np.where(prob_0 > threshold)
        point_1 = np.where(prob_1 > threshold)
        one_0 = np.ones_like(point_0[0])[:, np.newaxis]
        one_1 = np.ones_like(point_1[0])[:, np.newaxis]
        point_0 = np.stack(point_0, axis=1)
        point_1 = np.stack(point_1, axis=1)

        # recover to the original size and flip the order (y,x) to (x,y)
        point_0 = (point_0 * scale_0)[:, ::-1]
        point_1 = (point_1 * scale_1)[:, ::-1]
        point_0 = np.concatenate((point_0, one_0), axis=1)
        point_1 = np.concatenate((point_1, one_1), axis=1)

        # compute correctness from 0 to 1

        # compute correctness from 1 to 0

        # compute average correctness

    def compute_distance(self, point_0, point_1):
        pass


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
        mAP = np.sum(precision[1:]*(recall[1:] - recall[:-1]))

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















