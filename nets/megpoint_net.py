# 
# Created by ZhangYuyang on 2019/9/19
#
import torch
import torch.nn as nn
import torch.nn.functional as f

from torchvision.models import ResNet
from torchvision.models import VGG


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
        self.bnPa = nn.BatchNorm2d(c5)
        self.bnDa = nn.BatchNorm2d(c5)

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


class MegPointNet(BaseMegPointNet):

    def __init__(self):
        super(MegPointNet, self).__init__()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

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
        feature = self.relu(self.conv4b(x))

        # detect head
        cPa = self.relu(self.convPa(feature))
        logit = self.convPb(cPa)
        prob = self.softmax(logit)[:, :-1, :, :]

        # descriptor head
        cDa = self.relu(self.convDa(feature))
        desc = self.convDb(cDa)
        dn = torch.norm(desc, p=2, dim=1, keepdim=True)
        desc = desc.div(dn)

        # return logit, prob, desc, feature
        return logit, prob, desc, feature


class MegPointShuffleHeatmap(nn.Module):
    """
    利用shuffle生成heatmap的输出的MegPoint网络类，其描述子生成方式与原网络相同，
    唯一不同点在于用shuffle操作生成全分辨率的概率图(heatmap)用于对每个点进行二分类。
    """

    def __init__(self):
        super(MegPointShuffleHeatmap, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.tanh = nn.Tanh()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        c1, c2, c3, c4, c5, d1 = 64, 64, 128, 128, 256, 256
        # Encoder.
        self.conv1a = nn.Conv2d(1, c1, kernel_size=3, stride=1, padding=1)
        self.conv1b = nn.Conv2d(c1, c1, kernel_size=3, stride=1, padding=1)
        self.conv2a = nn.Conv2d(c1, c2, kernel_size=3, stride=1, padding=1)
        self.conv2b = nn.Conv2d(c2, c2, kernel_size=3, stride=1, padding=1)
        self.conv3a = nn.Conv2d(c2, c3, kernel_size=3, stride=1, padding=1)
        self.conv3b = nn.Conv2d(c3, c3, kernel_size=3, stride=1, padding=1)
        self.conv4a = nn.Conv2d(c3, c4, kernel_size=3, stride=1, padding=1)
        self.conv4b = nn.Conv2d(c4, c4, kernel_size=3, stride=1, padding=1)

        # Detector head
        self.convPa = nn.Conv2d(c4, c5, kernel_size=3, stride=1, padding=1)
        self.convPb = nn.Conv2d(c5, 64, kernel_size=3, stride=1, padding=1)  # different from superpoint

        # Descriptor Head.
        self.convDa = nn.Conv2d(c4, c5, kernel_size=3, stride=1, padding=1)
        self.convDb = nn.Conv2d(c5, d1, kernel_size=1, stride=1, padding=0)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                if m.affine:
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):

        _, _, h, w = x.shape
        # encoder part
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
        feature = self.relu(self.conv4b(x))

        # detector head
        cPa = self.relu(self.convPa(feature))
        cPb = self.convPb(cPa)
        heatmap = f.pixel_shuffle(cPb, upscale_factor=8)

        # descriptor head
        cDa = self.relu(self.convDa(feature))
        cDb = self.convDb(cDa)

        dn = torch.norm(cDb, p=2, dim=1, keepdim=True)
        desp = cDb.div(dn)

        return heatmap, desp


class MegPointResidualShuffleHeatmap(nn.Module):
    """
    利用浅层特征生成浅层的区域描述子，深层网络生成浅层描述子的残差+浅层描述子得到深层次的区域描述子
    利用shuffle生成heatmap的输出的MegPoint网络类，其描述子生成方式与原网络相同，
    唯一不同点在于用shuffle操作生成全分辨率的概率图(heatmap)用于对每个点进行二分类。
    """

    def __init__(self):
        super(MegPointResidualShuffleHeatmap, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.tanh = nn.Tanh()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        c1, c2, c3, c4, c5, d1 = 64, 64, 128, 128, 256, 256
        # Encoder.
        self.conv1a = nn.Conv2d(1, c1, kernel_size=3, stride=1, padding=1)
        self.conv1b = nn.Conv2d(c1, c1, kernel_size=3, stride=1, padding=1)
        self.conv2a = nn.Conv2d(c1, c2, kernel_size=3, stride=1, padding=1)
        self.conv2b = nn.Conv2d(c2, c2, kernel_size=3, stride=1, padding=1)
        self.conv3a = nn.Conv2d(c2, c3, kernel_size=3, stride=1, padding=1)
        self.conv3b = nn.Conv2d(c3, c3, kernel_size=3, stride=1, padding=1)
        self.conv4a = nn.Conv2d(c3, c4, kernel_size=3, stride=1, padding=1)
        self.conv4b = nn.Conv2d(c4, c4, kernel_size=3, stride=1, padding=1)

        # 用于下采样浅层特征
        self.downsample = nn.Conv2d(c2, c4, kernel_size=4, stride=4, padding=0)

        # Detector head
        self.convPa = nn.Conv2d(c4, c5, kernel_size=3, stride=1, padding=1)
        self.convPb = nn.Conv2d(c5, 64, kernel_size=3, stride=1, padding=1)  # different from superpoint

        # Descriptor Head.
        self.convDa = nn.Conv2d(c4, c5, kernel_size=3, stride=1, padding=1)
        self.convDb = nn.Conv2d(c5, d1, kernel_size=1, stride=1, padding=0)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                if m.affine:
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):

        _, _, h, w = x.shape
        # encoder part
        x = self.relu(self.conv1a(x))
        x = self.relu(self.conv1b(x))

        x = self.pool(x)
        x = self.relu(self.conv2a(x))
        x = self.relu(self.conv2b(x))
        feature_shallow = x  # 浅层特征
        feature_shallow = self.downsample(feature_shallow)  # 经下采样后的浅层特征

        x = self.pool(x)
        x = self.relu(self.conv3a(x))
        x = self.relu(self.conv3b(x))

        x = self.pool(x)
        x = self.relu(self.conv4a(x))
        feature_deep = self.relu(self.conv4b(x) + feature_shallow)  # 经深度残差加和的深层特征

        # detector head
        cPa = self.relu(self.convPa(feature_deep))
        cPb = self.convPb(cPa)
        heatmap = f.pixel_shuffle(cPb, upscale_factor=8)

        # 浅层特征描述子端
        cDa_shallow = self.relu(self.convDa(feature_shallow))
        cDb_shallow = self.convDb(cDa_shallow)
        dn = torch.norm(cDb_shallow, p=2, dim=1, keepdim=True)
        desp_shallow = cDb_shallow.div(dn)

        # 深层特征描述子端
        cDa_deep = self.relu(self.convDa(feature_deep))
        cDb_deep = self.convDb(cDa_deep)
        dn = torch.norm(cDb_deep, p=2, dim=1, keepdim=True)
        desp_deep = cDb_deep.div(dn)

        return heatmap, desp_deep, desp_shallow




