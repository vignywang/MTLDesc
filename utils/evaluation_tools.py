#
# Created by ZhangYuyang on 2019/8/13
#
import numpy as np


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

        # 进行累加计算
        tp_cum = np.cumsum(tp)
        fp_cum = np.cumsum(fp)
        recall = tp_cum / self.total_num
        precision = tp_cum / (tp_cum + fp_cum)
        recall = np.concatenate([[0], recall, [1]])
        precision = np.concatenate([[0], precision, [0]])
        mAP = np.sum(precision[1:]*(recall[1:] - recall[:-1]))
        return mAP, recall, precision

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















