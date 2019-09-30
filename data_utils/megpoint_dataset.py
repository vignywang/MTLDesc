# 
# Created by ZhangYuyang on 2019/9/30
#
import cv2 as cv
import numpy as np
import torch
from torch.utils.data import Dataset

from data_utils.dataset_tools import HomographyAugmentation


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