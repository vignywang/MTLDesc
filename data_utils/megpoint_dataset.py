# 
# Created by ZhangYuyang on 2019/9/30
#
import cv2 as cv
import numpy as np
import torch
from torch import nn as nn
from torch.nn import functional as f
from torch.utils.data import Dataset

from data_utils.dataset_tools import HomographyAugmentation, PhotometricAugmentation
from data_utils.dataset_tools import space_to_depth
from nets.megpoint_net import BaseMegPointNet


class SpatialNonMaximumSuppression(nn.Module):

    def __init__(self, kernal_size=9):
        super(SpatialNonMaximumSuppression, self).__init__()
        padding = int(kernal_size//2)
        self.pool2d = nn.MaxPool2d(kernel_size=kernal_size, stride=1, padding=padding)

    def forward(self, x):
        pooled = self.pool2d(x)
        prob = torch.where(torch.eq(x, pooled), x, torch.zeros_like(x))
        return prob


class LabelGenerator(nn.Module):

    def __init__(self, params):
        super(LabelGenerator, self).__init__()
        self.sample_num = params.sample_num
        self.detection_threshold = params.detection_threshold
        self.train_top_k = params.train_top_k
        self.photo_sampler = PhotometricAugmentation(**params.photometric_params)
        self.spatial_nms = SpatialNonMaximumSuppression(int(params.nms_threshold*2+1))

        height = 240
        width = 320
        coords_x = torch.arange(0, width, dtype=torch.float, requires_grad=False)
        coords_y = torch.arange(0, height, dtype=torch.float, requires_grad=False)
        ones = torch.ones((height, width), dtype=torch.float, requires_grad=False)
        self.org_coords = torch.stack(
            (coords_x.unsqueeze(dim=0).repeat(height, 1),
             coords_y.unsqueeze(dim=1).repeat((1, width)),
             ones), dim=2
        ).reshape((height * width, 3))

        self.base_megpoint = BaseMegPointNet()

    def forward(self, image, sampled_homo, sampled_inv_homo, detection_threshold):

        self.base_megpoint.eval()
        shape = image.shape
        device = image.device

        # 用采样的单应变换合成图像
        org_image = image
        image = image.repeat((self.sample_num, 1, 1, 1))
        warped_image = interpolation(image, sampled_homo)

        # np_warped_image = warped_image.detach().cpu().numpy()

        # 得到所有图像的关键点概率预测图
        warped_images = torch.chunk(warped_image, self.sample_num, dim=0)
        warped_probs = []
        for j in range(self.sample_num):
            _, warped_prob = self.base_megpoint(warped_images[j])
            warped_probs.append(warped_prob.detach())
        warped_prob = torch.cat(warped_probs, dim=0)

        warped_prob = f.pixel_shuffle(warped_prob, 8)
        warped_count = torch.ones_like(warped_prob)

        prob = interpolation(warped_prob, sampled_inv_homo)
        count = interpolation(warped_count, sampled_inv_homo)

        probs = torch.split(prob, shape[0], dim=0)
        counts = torch.split(count, shape[0], dim=0)
        prob = torch.cat(probs, dim=1)  # [bt,10,h,w]
        count = torch.cat(counts, dim=1)

        final_prob = torch.sum(prob, dim=1, keepdim=True) / torch.sum(count, dim=1, keepdim=True)  # [bt,1,h,w]
        final_prob = self.spatial_nms(final_prob)

        # np_final_prob = final_prob.detach().cpu().numpy()

        # 取响应值的前top_k个
        sorted_prob, _ = torch.sort(final_prob.reshape((shape[0], shape[2]*shape[3])), dim=1, descending=True)
        threshold = sorted_prob[:, self.train_top_k-1:self.train_top_k]  # [bt, 1]
        final_prob = torch.where(
            torch.ge(final_prob, threshold.reshape((shape[0], 1, 1, 1))),
            final_prob,
            torch.zeros_like(final_prob)
        )
        # 在top_k中再取大于threshold的点
        final_prob = torch.where(
            torch.ge(final_prob, detection_threshold),
            final_prob,
            torch.zeros_like(final_prob)
        )  # [bt,1,h,w]

        final_prob = final_prob.reshape((shape[0], shape[2]*shape[3]))  # [bt,h*w]
        sorted_final_prob, sorted_idx = torch.sort(final_prob, dim=1, descending=True)
        point_mask = torch.where(
            sorted_final_prob > 0, torch.ones_like(sorted_final_prob), torch.zeros_like(sorted_final_prob))
        # 取前top k个点的idx，若该点不是关键点，那么idx统一为0

        point_mask = point_mask[:, :self.train_top_k]
        topk_idx = sorted_idx[:, :self.train_top_k].to(torch.float)  # [bt, top_k]
        point_x = topk_idx % shape[3]
        point_y = topk_idx // shape[3]
        ones = torch.ones_like(point_x)
        point = torch.stack((point_x, point_y, ones), dim=2)  # [bt, top_k, 3]

        # space_label = torch.where(torch.gt(final_prob, 0), torch.ones_like(final_prob), torch.zeros_like(final_prob))

        image = org_image.squeeze()

        return image, point, point_mask


class AdaptionDataset(Dataset):

    def __init__(self, length, homography_params, photometric_params):
        self.height = 240
        self.width = 320
        self.n_height = int(self.height/8)
        self.n_width = int(self.width/8)
        self.sample_list = []
        self.homography = HomographyAugmentation(**homography_params)
        self.photometric = PhotometricAugmentation(**photometric_params)
        self.cpu = torch.device("cpu:0")
        self.gpu = torch.device("cuda:0")
        self.dataset_length = length
        self.center_grid = self._generate_center_grid()

    def __len__(self):
        return self.dataset_length

    def reset(self):
        self.sample_list = []

    def append(self, image, point, point_mask):
        image = image.to(self.cpu)
        point = point.to(self.cpu)
        point_mask = point_mask.to(self.cpu)
        for i in range(image.shape[0]):
            single_sample = {
                "image": image[i], "point": point[i], "point_mask": point_mask[i]}
            self.sample_list.append(single_sample)

    def __getitem__(self, idx):
        sample = self.sample_list[idx]

        # 1.从内存中读取图像以及对应的点标签
        image = sample["image"].numpy()
        image = ((image + 1.) * 255./2.).astype(np.uint8)
        org_mask = np.ones_like(image)

        point = sample["point"].numpy()
        point_mask = sample["point_mask"].numpy()
        valid_point_idx = np.nonzero(point_mask)
        point = np.flip(point[valid_point_idx][:, :2], axis=1)  # [n,2] 顺序为y,x

        # 1.1 由随机采样的单应变换得到第二副图像及其对应的关键点位置、原始掩膜和该单应变换
        if torch.rand([]).item() < 0.5:
            warped_image = image.copy()
            warped_org_mask = org_mask.copy()
            warped_point = point.copy()
            homography = np.eye(3)
        else:
            warped_image, warped_org_mask, warped_point, homography = self.homography(image, point, return_homo=True)

        # 2.对图像光度进行增强
        if torch.rand([]).item() < 0.5:
            image = self.photometric(image)
        if torch.rand([]).item() < 0.5:
            warped_image = self.photometric(warped_image)

        image = torch.from_numpy(image).to(torch.float).unsqueeze(dim=0)
        warped_image = torch.from_numpy(warped_image).to(torch.float).unsqueeze(dim=0)
        image = image*2./255. - 1.
        warped_image = warped_image*2./255. - 1.

        # 3. 对点和标签的相关处理
        # 3.1 输入的点标签和掩膜的预处理
        point = np.abs(np.floor(point)).astype(np.int)
        warped_point = np.abs(np.floor(warped_point)).astype(np.int)
        point = torch.from_numpy(point)
        warped_point = torch.from_numpy(warped_point)
        org_mask = torch.from_numpy(org_mask)
        warped_org_mask = torch.from_numpy(warped_org_mask)

        # 3.2 得到第一副图点和标签的最终输出
        label = self._convert_points_to_label(point).to(torch.long)
        mask = space_to_depth(org_mask).to(torch.uint8)
        mask = torch.all(mask, dim=0).to(torch.float)

        # 3.3 得到第二副图点和标签的最终输出
        warped_label = self._convert_points_to_label(warped_point).to(torch.long)
        warped_mask = space_to_depth(warped_org_mask).to(torch.uint8)
        warped_mask = torch.all(warped_mask, dim=0).to(torch.float)

        # 4. 构造描述子loss有关关系的计算
        # 4.1 得到第二副图中有效描述子的掩膜
        warped_valid_mask = warped_mask.reshape((-1,))

        # 4.2 根据指定的loss类型计算不同的关系，
        #     triplet要计算匹配对应关系，匹配有效掩膜，匹配点与其他点的近邻关系
        matched_idx, matched_valid, not_search_mask, warped_grid, matched_grid, warped_valid_mask = \
            self.generate_corresponding_relationship(homography, warped_valid_mask)
        matched_idx = torch.from_numpy(matched_idx)
        matched_valid = torch.from_numpy(matched_valid).to(torch.float)
        not_search_mask = torch.from_numpy(not_search_mask)
        warped_grid = torch.from_numpy(warped_grid)
        matched_grid = torch.from_numpy(matched_grid)
        warped_valid_mask = torch.from_numpy(warped_valid_mask)

        return {'image': image, 'mask': mask, 'label': label,
                'warped_image': warped_image, 'warped_mask': warped_mask, 'warped_label': warped_label,
                'matched_idx': matched_idx, 'matched_valid': matched_valid,
                'not_search_mask': not_search_mask,
                'warped_grid': warped_grid, 'matched_grid': matched_grid, 'warped_valid_mask': warped_valid_mask}

    def _convert_points_to_label(self, points):

        height = self.height
        width = self.width
        n_height = int(height / 8)
        n_width = int(width / 8)
        assert n_height * 8 == height and n_width * 8 == width

        num_pt = points.shape[0]
        label = torch.zeros((height * width))
        if num_pt > 0:
            points_h, points_w = torch.split(points, 1, dim=1)
            points_idx = points_w + points_h * width
            label = label.scatter_(dim=0, index=points_idx[:, 0], value=1.0).reshape((height, width))
        else:
            label = label.reshape((height, width))

        dense_label = space_to_depth(label)
        dense_label = torch.cat((dense_label, 0.5 * torch.ones((1, n_height, n_width))), dim=0)  # [65, 30, 40]
        sparse_label = torch.argmax(dense_label, dim=0)  # [30,40]

        return sparse_label

    def _generate_center_grid(self, patch_height=8, patch_width=8):
        n_height = int(self.height/patch_height)
        n_width = int(self.width/patch_width)
        center_grid = []
        for i in range(n_height):
            for j in range(n_width):
                h = (patch_height-1.)/2. + i*patch_height
                w = (patch_width-1.)/2. + j*patch_width
                center_grid.append((w, h))
        center_grid = np.stack(center_grid, axis=0)
        return center_grid

    def _generate_descriptor_mask(self, homography):

        center_grid, warped_center_grid = self.__compute_warped_center_grid(homography)

        center_grid = np.expand_dims(center_grid, axis=0)  # [1,n,2]
        warped_center_grid = np.expand_dims(warped_center_grid, axis=1)  # [n,1,2]

        dist = np.linalg.norm((warped_center_grid-center_grid), axis=2)  # [n,n]
        mask = (dist < 8.).astype(np.float32)

        return mask

    def generate_corresponding_relationship(self, homography, valid_mask):

        center_grid, warped_center_grid = self.__compute_warped_center_grid(homography)

        dist = np.linalg.norm(warped_center_grid[:, np.newaxis, :]-center_grid[np.newaxis, :, :], axis=2)
        nearest_idx = np.argmin(dist, axis=1)
        nearest_dist = np.min(dist, axis=1)
        matched_valid = nearest_dist < 8.

        matched_grid = center_grid[nearest_idx, :]
        diff = np.linalg.norm(matched_grid[:, np.newaxis, :] - matched_grid[np.newaxis, :, :], axis=2)

        # nearest = diff < 8.
        nearest = diff < 16.
        valid_mask = valid_mask.numpy().astype(np.bool)
        valid_mask = valid_mask[nearest_idx]
        invalid = ~valid_mask[np.newaxis, :]

        not_search_mask = (nearest | invalid).astype(np.float32)
        matched_valid = matched_valid & valid_mask
        valid_mask = valid_mask.astype(np.float32)

        return nearest_idx, matched_valid, not_search_mask, warped_center_grid, matched_grid, valid_mask

    def __compute_warped_center_grid(self, homography, return_org_center_grid=True):

        center_grid = self.center_grid.copy()  # [n,2]
        num = center_grid.shape[0]
        ones = np.ones((num, 1), dtype=np.float)
        homo_center_grid = np.concatenate((center_grid, ones), axis=1)[:, :, np.newaxis]  # [n,3,1]
        warped_homo_center_grid = np.matmul(homography, homo_center_grid)
        warped_center_grid = warped_homo_center_grid[:, :2, 0] / warped_homo_center_grid[:, 2:, 0]  # [n,2]

        if return_org_center_grid:
            return center_grid, warped_center_grid
        else:
            return warped_center_grid


def interpolation(image, homo):
    bt, _, height, width = image.shape
    device = image.device
    inv_homo = torch.inverse(homo)

    height = 240
    width = 320
    coords_x = torch.arange(0, width, dtype=torch.float, requires_grad=False)
    coords_y = torch.arange(0, height, dtype=torch.float, requires_grad=False)
    ones = torch.ones((height, width), dtype=torch.float, requires_grad=False)
    org_coords = torch.stack(
        (coords_x.unsqueeze(dim=0).repeat(height, 1),
         coords_y.unsqueeze(dim=1).repeat((1, width)),
         ones), dim=2
    ).reshape((height * width, 3)).unsqueeze(dim=0).repeat((bt, 1, 1)).to(device)

    warped_coords = torch.bmm(inv_homo.unsqueeze(dim=1).repeat(1, height * width, 1, 1).reshape((-1, 3, 3)),
                              org_coords.reshape((-1, 3, 1))).squeeze().reshape((bt, height * width, 3))
    warped_coords_x = warped_coords[:, :, 0] / warped_coords[:, :, 2]
    warped_coords_y = warped_coords[:, :, 1] / warped_coords[:, :, 2]

    x0 = torch.floor(warped_coords_x)
    x1 = x0 + 1.
    y0 = torch.floor(warped_coords_y)
    y1 = y0 + 1.

    x0_safe = torch.clamp(x0, 0, width - 1)  # [bt*num,h*w]
    x1_safe = torch.clamp(x1, 0, width - 1)
    y0_safe = torch.clamp(y0, 0, height - 1)
    y1_safe = torch.clamp(y1, 0, height - 1)

    idx_00 = (x0_safe + y0_safe * width).to(torch.long)  # [bt*num,h*w]
    idx_01 = (x0_safe + y1_safe * width).to(torch.long)
    idx_10 = (x1_safe + y0_safe * width).to(torch.long)
    idx_11 = (x1_safe + y1_safe * width).to(torch.long)

    d_x = warped_coords_x - x0_safe
    d_y = warped_coords_y - y0_safe
    d_1_x = x1_safe - warped_coords_x
    d_1_y = y1_safe - warped_coords_y

    image = image.reshape((bt, height * width))  # [bt*num,h*w]
    img_00 = torch.gather(image, dim=1, index=idx_00)
    img_01 = torch.gather(image, dim=1, index=idx_01)
    img_10 = torch.gather(image, dim=1, index=idx_10)
    img_11 = torch.gather(image, dim=1, index=idx_11)

    bilinear_img = (img_00 * d_1_x * d_1_y + img_01 * d_1_x * d_y + img_10 * d_x * d_1_y + img_11 * d_x * d_y).reshape(
        (bt, 1, height, width)
    )

    return bilinear_img


def batch_space_to_depth(org_label, is_bool=False):
    batch_size, _, h, w = org_label.shape
    n_patch_h = h // 8
    n_patch_w = w // 8
    h_parts = torch.split(org_label, 8, dim=2)  # 每一个为[bt,1,8,w]
    all_parts = []
    for h_part in h_parts:
        w_parts = torch.split(h_part, 8, dim=3)  # 每一个为[bt,1,8,8]
        for w_part in w_parts:
            all_parts.append(w_part.reshape((batch_size, 64)))
    dense_label = torch.stack(all_parts, dim=2).reshape(
        (batch_size, 64, n_patch_h, n_patch_w))  # [bt,64,patch_h,patch_w]
    if not is_bool:
        extra_label = 0.5 * torch.ones_like(dense_label[:, :1, :, :])
        dense_label = torch.cat((dense_label, extra_label), dim=1)  # [bt,65,patch_h,patch_w]
        sparse_label = torch.argmax(dense_label, dim=1)
    else:
        sparse_label = torch.all(dense_label, dim=1)
    return sparse_label


def convert_point_to_label(point, point_mask):
    point = torch.where(point_mask.unsqueeze(dim=2).repeat((1, 1, 3)) > 0, point, torch.zeros_like(point))
    point = point.to(torch.long)
    point_y = point[:, :, 1]
    point_x = point[:, :, 0]
    batch_size, num_point, _ = point.shape

    height = 240
    width = 320
    label_list = []
    for i in range(batch_size):
        single_point_y = point_y[i]  # [topk]
        single_point_x = point_x[i]
        point_idx = torch.stack((single_point_y, single_point_x), dim=0)  # [2, topk]
        ones = torch.ones_like(single_point_y, dtype=torch.float)

        single_label = torch.sparse.FloatTensor(point_idx, ones, torch.Size((height, width))).to_dense()
        label_list.append(single_label)
    space_label = torch.stack(label_list, dim=0).unsqueeze(dim=1).reshape((batch_size, 1, height, width))
    space_label[:, :, 0, 0] = 0.
    return space_label


def projection(point, homo, height=240, width=320):
    device = point.device
    project_point = torch.matmul(point, homo.transpose(1, 2))
    # project_point = torch.matmul(point, torch.eye(3, device=device))
    project_point = project_point[:, :, :2] / project_point[:, :, 2:3]

    border_0 = torch.tensor((0., 0.), device=device)
    border_1 = torch.tensor((width-1., height-1.), device=device)

    mask_0 = torch.where(
        project_point > border_0, torch.ones_like(project_point), torch.zeros_like(project_point))
    mask_1 = torch.where(
        project_point < border_1, torch.ones_like(project_point), torch.zeros_like(project_point))

    # mask = torch.ones_like(project_point[:, :, 0])
    mask = torch.all(mask_0.to(torch.bool) & mask_1.to(torch.bool), dim=2).to(torch.float)
    ones = torch.ones_like(project_point[:, :, 0:1])
    project_point = torch.cat((project_point, ones), dim=2)

    return project_point, mask
