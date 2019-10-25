#
# Created by ZhangYuyang on 2019/8/14
#
import cv2 as cv
import numpy as np
import torch
import torch.nn.functional as f


def compute_batched_dist(x, y, hamming=False):
    # x:[bt,256,n], y:[bt,256,n]
    cos_similarity = torch.matmul(x.transpose(1, 2), y)  # [bt,n,n]
    if hamming is False:
        square_norm_x = (torch.norm(x, dim=1, keepdim=True).transpose(1, 2))**2  # [bt,n,1]
        square_norm_y = (torch.norm(y, dim=1, keepdim=True))**2  # [bt,1,n]
        dist = torch.sqrt((square_norm_x + square_norm_y - 2 * cos_similarity + 1e-4))
        return dist
    else:
        dist = 0.5*(256-cos_similarity)
    return dist


def compute_cos_similarity_general(x, y):
    x_norm = torch.norm(x, p=2, dim=1, keepdim=True)
    y_norm = torch.norm(y, p=2, dim=1, keepdim=True)
    x = x.div(x_norm+1e-4)
    y = y.div(y_norm+1e-4)
    cos_similarity = torch.matmul(x.transpose(1, 2), y)  # [bt,h*w,h*w]
    return cos_similarity


def compute_cos_similarity_binary(x, y, k=256):
    x = x.div(np.sqrt(k))
    y = y.div(np.sqrt(k))
    cos_similarity = torch.matmul(x.transpose(1, 2), y)
    return cos_similarity


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


def prob_2_entropy(prob):
    """ convert probabilistic prediction maps to weighted self-information maps
    """
    n, c, h, w = prob.size()
    return -torch.mul(prob, torch.log2(prob + 1e-30)) / np.log2(c)


class DescriptorTripletLoss(object):

    def __init__(self, device):
        self.device = device

    def __call__(self, desp_0, desp_1, matched_idx, matched_valid, not_search_mask, debug_use=False):

        bt, dim, h, w = desp_0.shape
        desp_0 = torch.reshape(desp_0, (bt, dim, h*w)).transpose(1, 2)  # [bt,h*w,dim]
        desp_1 = torch.reshape(desp_1, (bt, dim, h*w))

        matched_idx = torch.unsqueeze(matched_idx, dim=1).repeat(1, dim, 1)  # [bt,dim,h*w]
        desp_1 = torch.gather(desp_1, dim=2, index=matched_idx)  # [bt,dim,h*w]

        cos_similarity = torch.matmul(desp_0, desp_1)
        dist = torch.sqrt(2.*(1.-cos_similarity)+1e-4)  # [bt,h*w,h*w]

        positive_pair = torch.diagonal(dist, dim1=1, dim2=2)  # [bt,h*w]
        dist = dist + 10*not_search_mask

        hardest_negative_pair, hardest_negative_idx = torch.min(dist, dim=2)  # [bt,h*w]

        zeros = torch.zeros_like(positive_pair)
        loss_total, _ = torch.max(torch.stack((zeros, 1.+positive_pair-hardest_negative_pair), dim=2), dim=2)
        loss_total *= matched_valid

        valid_num = torch.sum(matched_valid, dim=1)
        loss = torch.mean(torch.sum(loss_total, dim=1)/(valid_num + 1.))

        if debug_use:
            positive_dist = torch.mean(torch.sum(positive_pair*matched_valid, dim=1)/valid_num)
            negative_dist = torch.mean(torch.sum(hardest_negative_pair*matched_valid, dim=1)/valid_num)
            return loss, positive_dist, negative_dist
        else:
            return loss


class BinaryDescriptorPairwiseLoss(object):

    def __init__(self, device, lambda_d=250, m_p=1, m_n=0.2, ):
        self.device = device
        self.lambda_d = torch.tensor(lambda_d, device=self.device)
        self.m_p = torch.tensor(m_p, device=self.device)
        self.m_n = torch.tensor(m_n, device=self.device)
        self.one = torch.tensor(1, device=self.device)

    def __call__(self, desp_0, desp_1, feature_0, feature_1, desp_mask, valid_mask):
        batch_size, dim, h, w = desp_0.shape
        desp_0 = torch.reshape(desp_0, (batch_size, dim, -1))
        desp_1 = torch.reshape(desp_1, (batch_size, dim, -1))
        feature_0 = torch.reshape(feature_0, (batch_size, dim, -1))
        feature_1 = torch.reshape(feature_1, (batch_size, dim, -1))

        cos_similarity = compute_cos_similarity_binary(desp_0, desp_1)
        positive = -f.logsigmoid(cos_similarity) * self.lambda_d
        negative = -f.logsigmoid(1. - cos_similarity)

        positive_mask = desp_mask
        negative_mask = self.one - desp_mask

        pairwise_loss = positive_mask * positive + negative_mask * negative

        # 考虑warp带来的某些区域没有图像,则其对应的描述子应当无效
        total_num = torch.sum(valid_mask, dim=1)*h*w
        pairwise_loss = torch.sum(valid_mask.unsqueeze(dim=1)*pairwise_loss, dim=(1, 2))/total_num
        pairwise_loss = torch.mean(pairwise_loss)

        sqrt_1_k = 1./np.sqrt(dim)
        desp_1_valid_num = torch.sum(valid_mask, dim=1)

        ones_1_k = sqrt_1_k * torch.ones_like(feature_0)
        quantization_loss_0 = torch.norm((torch.abs(feature_0)-ones_1_k), p=1, dim=1)
        quantization_loss_1 = torch.norm((torch.abs(feature_1)-ones_1_k), p=1, dim=1)
        quantization_loss_0 = torch.mean(quantization_loss_0, dim=1)
        quantization_loss_1 = torch.sum(quantization_loss_1*valid_mask, dim=1)/desp_1_valid_num
        quantization_loss = torch.mean(torch.cat((quantization_loss_0, quantization_loss_1)))

        return pairwise_loss, quantization_loss


class BinaryDescriptorTripletLoss(object):

    def __init__(self):
        self.gamma = 10
        self.threshold = 1.0
        self.quantization_weight = 1.0
        self.dim = 256.

    def __call__(self, desp_0, desp_1, matched_idx, matched_valid, not_search_mask, valid_mask):
        bt, dim, h, w = desp_0.shape
        desp_0 = torch.reshape(desp_0, (bt, dim, h*w))  # [bt,dim,h*w]
        desp_1 = torch.reshape(desp_1, (bt, dim, h*w))

        matched_idx = torch.unsqueeze(matched_idx, dim=1).repeat(1, dim, 1)  # [bt,dim,h*w]
        desp_1 = torch.gather(desp_1, dim=2, index=matched_idx)  # [bt,dim,h*w]

        cos_similarity = torch.matmul(desp_0.transpose(1, 2), desp_1)  # [bt,h*w,h*w]

        dist = torch.sqrt(2.*(1.-cos_similarity)+1e-4)  # [bt,h*w,h*w]
        positive_pair = torch.diagonal(dist, dim1=1, dim2=2)  # [bt,h*w]
        dist = dist + 10*not_search_mask
        hardest_negative_pair, hardest_negative_idx = torch.min(dist, dim=2)  # [bt,h*w]
        triplet_metric = f.relu(1. + positive_pair - hardest_negative_pair)

        triplet_loss = triplet_metric*matched_valid
        match_valid_num = torch.sum(matched_valid, dim=1)
        triplet_loss = torch.mean(torch.sum(triplet_loss, dim=1)/match_valid_num)

        sqrt_1_k = 1./np.sqrt(dim)
        desp_1_valid_num = torch.sum(valid_mask, dim=1)

        ones_1_k = sqrt_1_k * torch.ones_like(desp_0)
        quantization_loss_0 = torch.norm((torch.abs(desp_0)-ones_1_k), p=1, dim=1)
        quantization_loss_1 = torch.norm((torch.abs(desp_1)-ones_1_k), p=1, dim=1)
        quantization_loss_0 = torch.mean(quantization_loss_0, dim=1)
        quantization_loss_1 = torch.sum(quantization_loss_1*valid_mask, dim=1)/desp_1_valid_num
        quantization_loss = torch.mean(torch.cat((quantization_loss_0, quantization_loss_1)))

        return triplet_loss, quantization_loss


class BinaryDescriptorTripletDirectLoss(object):

    def __init__(self):
        self.gamma = 10
        self.threshold = 1.0
        self.quantization_weight = 1.0
        self.dim = 256.

    def __call__(self, desp_0, desp_1, feature_0, feature_1, matched_idx, matched_valid, not_search_mask, valid_mask):
        bt, dim, h, w = desp_0.shape
        norm_factor = np.sqrt(dim)
        desp_0 = torch.reshape(desp_0, (bt, dim, h*w))  # [bt,dim,h*w]
        desp_1 = torch.reshape(desp_1, (bt, dim, h*w))
        feature_0 = torch.reshape(feature_0, (bt, dim, h*w))
        feature_1 = torch.reshape(feature_1, (bt, dim, h*w))

        matched_idx = torch.unsqueeze(matched_idx, dim=1).repeat(1, dim, 1)  # [bt,dim,h*w]
        desp_1 = torch.gather(desp_1, dim=2, index=matched_idx)  # [bt,dim,h*w]
        feature_1 = torch.gather(feature_1, dim=2, index=matched_idx)

        desp_0 = desp_0/norm_factor
        desp_1 = desp_1/norm_factor
        cos_similarity = torch.matmul(desp_0.transpose(1, 2), desp_1)

        positive_pair = torch.diagonal(cos_similarity, dim1=1, dim2=2)  # [bt,h*w]
        minus_cos_sim = 1.0 - cos_similarity + not_search_mask*10
        hardest_negative_pair, hardest_negative_idx = torch.min(minus_cos_sim, dim=2)  # [bt,h*w]

        triplet_metric = -f.logsigmoid(positive_pair)-f.logsigmoid(hardest_negative_pair)

        triplet_loss = triplet_metric*matched_valid
        match_valid_num = torch.sum(matched_valid, dim=1)
        triplet_loss = torch.mean(torch.sum(triplet_loss, dim=1)/match_valid_num)

        sqrt_1_k = 1./np.sqrt(dim)
        desp_1_valid_num = torch.sum(valid_mask, dim=1)

        ones_1_k = sqrt_1_k * torch.ones_like(feature_0)
        quantization_loss_0 = torch.norm((torch.abs(feature_0)-ones_1_k), p=1, dim=1)
        quantization_loss_1 = torch.norm((torch.abs(feature_1)-ones_1_k), p=1, dim=1)
        quantization_loss_0 = torch.mean(quantization_loss_0, dim=1)
        quantization_loss_1 = torch.sum(quantization_loss_1*valid_mask, dim=1)/desp_1_valid_num
        quantization_loss = torch.mean(torch.cat((quantization_loss_0, quantization_loss_1)))

        return triplet_loss, quantization_loss


class BinaryDescriptorTripletTanhLoss(object):

    def __init__(self):
        pass

    def __call__(self, desp_0, desp_1, matched_idx, matched_valid, not_search_mask, valid_mask):
        bt, dim, h, w = desp_0.shape
        desp_0 = torch.reshape(desp_0, (bt, dim, h*w))  # [bt,dim,h*w]
        desp_1 = torch.reshape(desp_1, (bt, dim, h*w))

        matched_idx = torch.unsqueeze(matched_idx, dim=1).repeat(1, dim, 1)  # [bt,dim,h*w]
        desp_1 = torch.gather(desp_1, dim=2, index=matched_idx)  # [bt,dim,h*w]

        cos_similarity = compute_cos_similarity_general(desp_0, desp_1)

        positive_pair = torch.diagonal(cos_similarity, dim1=1, dim2=2)  # [bt,h*w]
        minus_cos_sim = 1.0 - cos_similarity + not_search_mask*10
        hardest_negative_pair, hardest_negative_idx = torch.min(minus_cos_sim, dim=2)  # [bt,h*w]

        triplet_metric = -f.logsigmoid(positive_pair)-f.logsigmoid(hardest_negative_pair)

        triplet_loss = triplet_metric*matched_valid
        match_valid_num = torch.sum(matched_valid, dim=1)
        triplet_loss = torch.mean(torch.sum(triplet_loss, dim=1)/match_valid_num)

        return triplet_loss


class BinaryDescriptorTripletTanhSigmoidLoss(object):

    def __init__(self, logger):
        self.logger = logger
        logger.info("Initialize the Alpha Sigmoid Tanh Triplet loss.")

    def __call__(self, desp_0, desp_1, matched_idx, matched_valid, not_search_mask, valid_mask):
        bt, dim, h, w = desp_0.shape
        desp_0 = torch.reshape(desp_0, (bt, dim, h*w))  # [bt,dim,h*w]
        desp_1 = torch.reshape(desp_1, (bt, dim, h*w))
        sigmoid_params = 10./dim

        matched_idx = torch.unsqueeze(matched_idx, dim=1).repeat(1, dim, 1)  # [bt,dim,h*w]
        desp_1 = torch.gather(desp_1, dim=2, index=matched_idx)  # [bt,dim,h*w]

        inner_product = torch.matmul(desp_0.transpose(1, 2), desp_1)/dim

        positive_pair = torch.diagonal(inner_product, dim1=1, dim2=2)  # [bt,h*w]

        minus_cos_sim = 1. - inner_product + not_search_mask*10.
        hardest_negative_pair, _ = torch.min(minus_cos_sim, dim=2)  # [bt,h*w]
        triplet_metric = -f.logsigmoid(positive_pair)-f.logsigmoid(hardest_negative_pair)

        # masked_cos_sim = inner_product - not_search_mask*10.
        # hardest_negative_pair, _ = torch.max(masked_cos_sim, dim=2)
        # triplet_metric = -f.logsigmoid(positive_pair)-f.logsigmoid(-hardest_negative_pair)

        triplet_loss = triplet_metric*matched_valid
        match_valid_num = torch.sum(matched_valid, dim=1)
        triplet_loss = torch.mean(torch.sum(triplet_loss, dim=1)/match_valid_num)

        return triplet_loss


class BinaryDescriptorTripletTanhCauchyLoss(object):

    def __init__(self, logger):
        self.logger = logger
        self.lambda_f = 20.
        logger.info("Initialized the Cauchy Tanh Triplet loss, lambda: %.3f" % self.lambda_f)

    def __call__(self, desp_0, desp_1, matched_idx, matched_valid, not_search_mask, valid_mask):
        bt, dim, h, w = desp_0.shape
        desp_0 = torch.reshape(desp_0, (bt, dim, h*w))  # [bt,dim,h*w]
        desp_1 = torch.reshape(desp_1, (bt, dim, h*w))

        matched_idx = torch.unsqueeze(matched_idx, dim=1).repeat(1, dim, 1)  # [bt,dim,h*w]
        desp_1 = torch.gather(desp_1, dim=2, index=matched_idx)  # [bt,dim,h*w]

        inner_product = torch.matmul(desp_0.transpose(1, 2), desp_1)
        hamming_dist = 0.5*(dim - inner_product)

        positive_pair = torch.diagonal(hamming_dist, dim1=1, dim2=2)  # [bt,h*w]
        masked_hamming_dist = hamming_dist + not_search_mask*dim
        hardest_negative_pair, _ = torch.min(masked_hamming_dist, dim=2)  # [bt,h*w]

        triplet_metric = -torch.log(self._cauchy(positive_pair)) - torch.log(1.001 - self._cauchy(hardest_negative_pair))

        triplet_loss = triplet_metric*matched_valid
        match_valid_num = torch.sum(matched_valid, dim=1)
        triplet_loss = torch.mean(torch.sum(triplet_loss, dim=1)/match_valid_num)

        return triplet_loss

    def _cauchy(self, x):
        return self.lambda_f/(self.lambda_f + x)


class Matcher(object):

    def __init__(self, dtype='float'):
        if dtype == 'float':
            self.compute_desp_dist = self._compute_desp_dist
        elif dtype == 'binary':
            self.compute_desp_dist = self._compute_desp_dist_binary
        else:
            assert False

    def __call__(self, point_0, desp_0, point_1, desp_1):
        dist_0_1 = self.compute_desp_dist(desp_0, desp_1)  # [n,m]
        dist_1_0 = dist_0_1.transpose((1, 0))  # [m,n]
        nearest_idx_0_1 = np.argmin(dist_0_1, axis=1)  # [n]
        nearest_idx_1_0 = np.argmin(dist_1_0, axis=1)  # [m]
        matched_src = []
        matched_tgt = []
        for i, idx_0_1 in enumerate(nearest_idx_0_1):
            if i == nearest_idx_1_0[idx_0_1]:
                matched_src.append(point_0[i])
                matched_tgt.append(point_1[idx_0_1])
        if len(matched_src) <= 4:
            print("There exist too little matches")
            # assert False
            return None
        if len(matched_src) != 0:
            matched_src = np.stack(matched_src, axis=0)
            matched_tgt = np.stack(matched_tgt, axis=0)
        return matched_src, matched_tgt

    @staticmethod
    def _compute_desp_dist(desp_0, desp_1):
        # desp_0:[n,256], desp_1:[m,256]
        square_norm_0 = (np.linalg.norm(desp_0, axis=1, keepdims=True)) ** 2  # [n,1]
        square_norm_1 = (np.linalg.norm(desp_1, axis=1, keepdims=True).transpose((1, 0))) ** 2  # [1,m]
        xty = np.matmul(desp_0, desp_1.transpose((1, 0)))  # [n,m]
        dist = np.sqrt((square_norm_0 + square_norm_1 - 2 * xty + 1e-4))
        return dist

    @staticmethod
    def _compute_desp_dist_binary(desp_0, desp_1):
        # desp_0:[n,256], desp_1[m,256]
        dist_0_1 = np.logical_xor(desp_0[:, np.newaxis, :], desp_1[np.newaxis, :, :]).sum(axis=2)
        return dist_0_1


class DescriptorTripletLogSigmoidLoss(object):

    def __init__(self, device):
        self.device = device

    def __call__(self, desp_0, desp_1, matched_idx, matched_valid, not_search_mask, debug_use=False):
        bt, dim, h, w = desp_0.shape
        desp_0 = torch.reshape(desp_0, (bt, dim, h*w))  # [bt,dim,h*w]
        desp_1 = torch.reshape(desp_1, (bt, dim, h*w))

        matched_idx = torch.unsqueeze(matched_idx, dim=1).repeat(1, dim, 1)  # [bt,dim,h*w]
        desp_1 = torch.gather(desp_1, dim=2, index=matched_idx)  # [bt,dim,h*w]

        cos_similarity = torch.matmul(desp_0.transpose(1, 2), desp_1)  # [bt,h*w,h*w]

        positive_pair = torch.diagonal(cos_similarity, dim1=1, dim2=2)  # [bt,h*w]
        minus_cos_sim = 1.0 - cos_similarity + not_search_mask*10

        hardest_negative_pair, hardest_negative_idx = torch.min(minus_cos_sim, dim=2)  # [bt,h*w]

        triplet_metric = -f.logsigmoid(positive_pair)-f.logsigmoid(hardest_negative_pair)

        triplet_loss = triplet_metric*matched_valid
        match_valid_num = torch.sum(matched_valid, dim=1)
        triplet_loss = torch.mean(torch.sum(triplet_loss, dim=1)/match_valid_num)

        return triplet_loss
