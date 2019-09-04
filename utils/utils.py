#
# Created by ZhangYuyang on 2019/8/14
#
import cv2 as cv
import numpy as np
import torch
import torch.nn.functional as f


def compute_desp_dist(desp_0, desp_1):
    # desp_0:[n,256], desp_1:[m,256]
    square_norm_0 = (np.linalg.norm(desp_0, axis=1, keepdims=True))**2  # [n,1]
    square_norm_1 = (np.linalg.norm(desp_1, axis=1, keepdims=True).transpose((1, 0)))**2  # [1,m]
    xty = np.matmul(desp_0, desp_1.transpose((1, 0)))  # [n,m]
    dist = np.sqrt((square_norm_0+square_norm_1-2*xty+1e-4))
    return dist


def spatial_nms(prob, kernel_size=9):
    """
    利用max_pooling对预测的特征点的概率图进行非极大值抑制
    Args:
        prob: shape为[h,w]的概率图
        kernel_size: 对每个点进行非极大值抑制时的窗口大小

    Returns:
        经非极大值抑制后的概率图
    """
    padding = int(kernel_size//2)
    pooled = f.max_pool2d(prob, kernel_size=kernel_size, stride=1, padding=padding)
    prob = torch.where(torch.eq(prob, pooled), prob, torch.zeros_like(prob))
    return prob


def draw_image_keypoints(image, points, color=(0, 255, 0)):
    """
    将输入的关键点画到图像上并显示出来
    Args:
        image: 待画点的原始图像
        points: 图像对应的关键点组合，输入为np.array，shape为（n，2）, 点的第一维代表y轴，第二维代表x轴
        color: 待描关键点的颜色
    Returns:
        None
    """
    n, _ = points.shape
    cv_keypoints = []
    for i in range(n):
        keypt = cv.KeyPoint()
        keypt.pt = (points[i, 1], points[i, 0])
        cv_keypoints.append(keypt)
    image = cv.drawKeypoints(image.astype(np.uint8), cv_keypoints, None, color=color)
    cv.imshow("image&keypoints", image)
    cv.waitKey()
    return image


class DescriptorHingeLoss(object):
    """
    According the Paper of SuperPoint
    """

    def __init__(self, device, lambda_d=250, m_p=1, m_n=0.2, ):
        self.device = device
        self.lambda_d = torch.tensor(lambda_d, device=self.device)
        self.m_p = torch.tensor(m_p, device=self.device)
        self.m_n = torch.tensor(m_n, device=self.device)
        self.one = torch.tensor(1, device=self.device)

    def __call__(self, desp_0, desp_1, desp_mask, valid_mask):
        batch_size, dim, h, w = desp_0.shape
        desp_0 = torch.reshape(desp_0, (batch_size, dim, -1)).transpose(dim0=2, dim1=1)  # [bt, h*w, dim]
        desp_1 = torch.reshape(desp_1, (batch_size, dim, -1))

        cos_similarity = torch.matmul(desp_0, desp_1)
        positive_term = f.relu(self.m_p - cos_similarity) * self.lambda_d
        negative_term = f.relu(cos_similarity - self.m_n)

        positive_mask = desp_mask
        negative_mask = self.one - desp_mask

        loss = positive_mask*positive_term+negative_mask*negative_term

        # 考虑warp带来的某些区域没有图像,则其对应的描述子应当无效
        valid_mask = torch.unsqueeze(valid_mask, dim=1)  # [bt, 1, h*w]
        total_num = torch.sum(valid_mask, dim=(1, 2))*h*w
        loss = torch.sum(valid_mask*loss, dim=(1, 2))/total_num
        loss = torch.mean(loss)

        return loss


class DescriptorTripletLoss(object):

    def __init__(self, device):
        self.device = device

    def __call__(self, desp_0, desp_1, matched_idx, matched_valid, not_search_mask,
                 warped_grid=None, matched_grid=None):

        bt, dim, h, w = desp_0.shape
        desp_0 = torch.reshape(desp_0, (bt, dim, h*w)).transpose(1, 2)  # [bt,h*w,dim]
        desp_1 = torch.reshape(desp_1, (bt, dim, h*w))

        matched_idx = torch.unsqueeze(matched_idx, dim=1).repeat(1, dim, 1)  # [bt,dim,h*w]
        desp_1 = torch.gather(desp_1, dim=2, index=matched_idx)  # [bt,dim,h*w]

        cos_similarity = torch.matmul(desp_0, desp_1)
        dist = torch.sqrt(2.*(1.-cos_similarity)+1e-4)  # [bt,h*w,h*w]

        positive_pair = torch.diagonal(dist, dim1=1, dim2=2)  # [bt,h*w]
        dist = dist + not_search_mask

        hardest_negative_pair, hardest_negative_idx = torch.min(dist, dim=2)  # [bt,h*w]

        zeros = torch.zeros_like(positive_pair)
        loss_total, _ = torch.max(torch.stack((zeros, 1.+positive_pair-hardest_negative_pair), dim=2), dim=2)
        loss_total *= matched_valid

        valid_num = torch.sum(matched_valid, dim=1)

        # debug use
        # matched_dist = torch.norm((warped_grid-matched_grid), dim=2).detach().cpu().numpy()
        # negative_grid = torch.gather(matched_grid, dim=1, index=hardest_negative_idx.unsqueeze(dim=2).repeat(1, 1, 2))
        # negative_dist = torch.norm((warped_grid-negative_grid), dim=2).detach().cpu().numpy()
        # valid = matched_valid.detach().cpu().numpy()
        # debug_positive_pair = positive_pair.detach().cpu().numpy()
        # debug_negative_pair = hardest_negative_pair.detach().cpu().numpy()
        # debug_dist = (torch.sum((positive_pair-hardest_negative_pair)*matched_valid, dim=1)/valid_num)\
        #     .detach().cpu().numpy()

        loss = torch.mean(torch.sum(loss_total, dim=1)/valid_num)

        return loss


class Matcher(object):

    def __init__(self):
        pass

    def __call__(self, point_0, desp_0, point_1, desp_1):
        dist_0_1 = compute_desp_dist(desp_0, desp_1)  # [n,m]
        dist_1_0 = dist_0_1.transpose((1, 0))  # [m,n]
        nearest_idx_0_1 = np.argmin(dist_0_1, axis=1)  # [n]
        nearest_idx_1_0 = np.argmin(dist_1_0, axis=1)  # [m]
        matched_src = []
        matched_tgt = []
        for i, idx_0_1 in enumerate(nearest_idx_0_1):
            if i == nearest_idx_1_0[idx_0_1]:
                matched_src.append(point_0[i])
                matched_tgt.append(point_1[idx_0_1])
        if len(matched_src) != 0:
            matched_src = np.stack(matched_src, axis=0)
            matched_tgt = np.stack(matched_tgt, axis=0)
        return matched_src, matched_tgt



