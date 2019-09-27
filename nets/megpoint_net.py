# 
# Created by ZhangYuyang on 2019/9/19
#
import torch
import torch.nn.functional as f
import torch.nn as nn
from torch.utils.data import Dataset
import numpy as np
import cv2 as cv

from data_utils.dataset_tools import HomographyAugmentation
from data_utils.dataset_tools import PhotometricAugmentation
from data_utils.dataset_tools import debug_draw_image_keypoints
from data_utils.dataset_tools import draw_image_keypoints
# from torchvision.models import ResNet


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


def space_to_depth(org_label, is_bool=False):
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


class AdaptionDataset(Dataset):

    def __init__(self, length, sampler_params):
        self.sample_list = []
        self.homography_sampler = HomographyAugmentation(**sampler_params)
        self.cpu = torch.device("cpu:0")
        self.gpu = torch.device("cuda:0")
        self.dataset_length = length

    def __len__(self):
        return self.dataset_length

    def __getitem__(self, idx):
        sample = self.sample_list[idx]
        image = sample["image"]
        image_mask = torch.ones_like(image, dtype=torch.bool)
        point = sample["point"]
        point_mask = sample["point_mask"]

        image, mask, point, point_mask = self.do_augmentation(image, image_mask, point, point_mask)
        sample = {"image": image, "image_mask": image_mask, "point": point, "point_mask": point_mask}

        return sample

    def do_augmentation(self, image, image_mask, point, point_mask):
        if np.random.randn() < 0.5:
            aug_homo = self.homography_sampler.sample()
        else:
            aug_homo = np.eye(3)

        # use opencv
        org_image = ((image[0] + 1) * 255./2.).numpy().astype(np.uint8)
        org_image_mask = image_mask[0].numpy().astype(np.float)
        warped_image = cv.warpPerspective(org_image, aug_homo, dsize=(320, 240), flags=cv.INTER_LINEAR)
        warped_image_mask = cv.warpPerspective(org_image_mask, aug_homo, dsize=(320, 240), flags=cv.INTER_LINEAR)
        valid_mask = warped_image_mask.astype(np.uint8).astype(np.bool)

        aug_image = torch.from_numpy(warped_image*2./255.-1).unsqueeze(dim=0).to(torch.float)
        aug_image_mask = torch.from_numpy(valid_mask).unsqueeze(dim=0)

        # use self-defined interpolation
        # org_image = ((image + 1) * 255 / 2).unsqueeze(dim=0)
        # org_image_mask = image_mask.to(torch.float).unsqueeze(dim=0)
        # warped_image = interpolation(org_image, torch.from_numpy(aug_homo).unsqueeze(dim=0).to(torch.float))[0]
        # warped_image_mask = interpolation(
        #     org_image_mask, torch.from_numpy(aug_homo).unsqueeze(dim=0).to(torch.float))[0]
        # valid_mask = warped_image_mask > 0.9

        # aug_image = warped_image*2/255-1
        # aug_image_mask = valid_mask

        aug_point, aug_point_mask = projection(
            point.unsqueeze(dim=0), torch.from_numpy(aug_homo).unsqueeze(dim=0).to(torch.float))
        aug_point_mask *= point_mask

        aug_point = aug_point[0]
        aug_point_mask = aug_point_mask[0]

        return aug_image, aug_image_mask, aug_point, aug_point_mask

    def append(self, image, point, point_mask):
        image = image.to(self.cpu)
        point = point.to(self.cpu)
        point_mask = point_mask.to(self.cpu)
        for i in range(image.shape[0]):
            single_sample = {
                "image": image[i], "point": point[i], "point_mask": point_mask[i]}
            self.sample_list.append(single_sample)

    def reset(self):
        self.sample_list = []

    def _sample_aug_homography(self, batch_size):
        sampled_homo = []
        for i in range(batch_size):
            # debug use
            # homo = np.eye(3)
            if np.random.randn() < 0.5:
                homo = self.homography_sampler.sample()
            else:
                homo = np.eye(3)
            sampled_homo.append(homo)
        sampled_homo = torch.from_numpy(np.stack(sampled_homo, axis=0)).to(torch.float)
        return sampled_homo


class BaseMegPointNet(nn.Module):

    def __init__(self):
        super(BaseMegPointNet, self).__init__()
        self.relu = torch.nn.ReLU(inplace=True)
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        c1, c2, c3, c4, c5, d1 = 64, 64, 128, 128, 256, 256
        # Shared Encoder.
        self.conv1a = nn.Conv2d(1, c1, kernel_size=3, stride=1, padding=1)
        self.conv1b = nn.Conv2d(c1, c1, kernel_size=3, stride=1, padding=1)
        self.conv2a = nn.Conv2d(c1, c2, kernel_size=3, stride=1, padding=1)
        self.conv2b = nn.Conv2d(c2, c2, kernel_size=3, stride=1, padding=1)
        self.conv3a = nn.Conv2d(c2, c3, kernel_size=3, stride=1, padding=1)
        self.conv3b = nn.Conv2d(c3, c3, kernel_size=3, stride=1, padding=1)
        self.conv4a = nn.Conv2d(c3, c4, kernel_size=3, stride=1, padding=1)
        self.conv4b = nn.Conv2d(c4, c4, kernel_size=3, stride=1, padding=1)
        # Detector Head.
        self.convPa = nn.Conv2d(c4, c5, kernel_size=3, stride=1, padding=1)
        self.convPb = nn.Conv2d(c5, 65, kernel_size=1, stride=1, padding=0)
        # Descriptor Head.
        self.convDa = nn.Conv2d(c4, c5, kernel_size=3, stride=1, padding=1)
        self.convDb = nn.Conv2d(c5, d1, kernel_size=1, stride=1, padding=0)

        # batch normalization
        self.bnPa = nn.BatchNorm2d(c5, affine=False)
        self.bnDa = nn.BatchNorm2d(c5, affine=False)

        self.softmax = nn.Softmax(dim=1)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.relu(self.conv1a(x))
        x = self.relu(self.conv1b(x))
        x = self.pool(x)
        x = self.relu(self.conv2a(x))
        x = self.relu(self.conv2b(x))
        x = self.pool(x)
        x = self.relu(self.conv3a(x))
        x = self.relu(self.conv3b(x))
        x = self.pool(x)
        x = self.relu(self.conv4a(x))
        x = self.relu(self.conv4b(x))

        cPa = self.relu(self.convPa(x))
        # cPa = self.bnPa(cPa)
        logit = self.convPb(cPa)
        prob = self.softmax(logit)[:, :-1, :, :]

        return logit, prob


class MegPointNet(nn.Module):

    def __init__(self):
        super(MegPointNet, self).__init__()
        self.base_megpoint = BaseMegPointNet()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        return self.base_megpoint(x)


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

        image = org_image

        return image, point, point_mask


class SpatialNonMaximumSuppression(nn.Module):

    def __init__(self, kernal_size=9):
        super(SpatialNonMaximumSuppression, self).__init__()
        padding = int(kernal_size//2)
        self.pool2d = nn.MaxPool2d(kernel_size=kernal_size, stride=1, padding=padding)

    def forward(self, x):
        pooled = self.pool2d(x)
        prob = torch.where(torch.eq(x, pooled), x, torch.zeros_like(x))
        return prob





