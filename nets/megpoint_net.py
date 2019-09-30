# 
# Created by ZhangYuyang on 2019/9/19
#
import torch
import torch.nn.functional as f
import torch.nn as nn

from data_utils.dataset_tools import PhotometricAugmentation
from data_utils.megpoint_dataset import interpolation
# from torchvision.models import ResNet


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





