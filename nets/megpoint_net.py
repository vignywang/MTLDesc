# 
# Created by ZhangYuyang on 2019/9/19
#
import torch
import torch.nn as nn
import torch.nn.functional as f

# from torchvision.models import ResNet
# from torchvision.models import VGG
# from torchvision.models.segmentation import deeplabv3_resnet50


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


class MegPointShuffleHeatmapOld(nn.Module):
    """
    用于恢复预训练的网络
    """

    def __init__(self):
        super(MegPointShuffleHeatmapOld, self).__init__()
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


class MegPointNew(nn.Module):
    """
    用于恢复预训练的网络
    """

    def __init__(self):
        super(MegPointNew, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.tanh = nn.Tanh()
        self.avg = nn.AvgPool2d(kernel_size=2, stride=2)
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

        self.compress1 = nn.Conv2d(c1, 32, kernel_size=1, stride=1)
        self.compress2 = nn.Conv2d(c2, 32, kernel_size=1, stride=1)
        self.compress3 = nn.Conv2d(c3, 32, kernel_size=1, stride=1)
        self.compress4 = nn.Conv2d(c4, 32, kernel_size=1, stride=1)

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
        c1 = self.relu(self.conv1b(x))

        c2 = self.avg(c1)
        c2 = self.relu(self.conv2a(c2))
        c2 = self.relu(self.conv2b(c2))

        c3 = self.avg(c2)
        c3 = self.relu(self.conv3a(c3))
        c3 = self.relu(self.conv3b(c3))

        c4 = self.avg(c3)
        c4 = self.relu(self.conv4a(c4))
        c4 = self.relu(self.conv4b(c4))

        compress1 = self.compress1(c1)
        compress2 = self.compress2(c2)
        compress3 = self.compress3(c3)
        compress4 = self.compress4(c4)

        return compress1, compress2, compress3, compress4


class MegPointSlidingHeatmap(nn.Module):
    """
    用于sliding训练的网络结构，只用于检测
    """

    def __init__(self):
        super(MegPointSlidingHeatmap, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.tanh = nn.Tanh()
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
        self.convPb = nn.Conv2d(c5, 1, kernel_size=3, stride=1, padding=1)  # different from superpoint

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

        x = self.relu(self.conv2a(x))
        x = self.relu(self.conv2b(x))

        x = self.relu(self.conv3a(x))
        x = self.relu(self.conv3b(x))

        x = self.relu(self.conv4a(x))
        feature = self.relu(self.conv4b(x))

        # detector head
        cPa = self.relu(self.convPa(feature))
        heatmap = self.convPb(cPa)

        return heatmap


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
        self.avgpool = nn.AvgPool2d(kernel_size=2, stride=2)
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

        # Feature pyramid network
        self.fpn_3 = nn.Conv2d(c4, c3, kernel_size=1, stride=1, padding=0)
        self.fpn_2 = nn.Conv2d(c3, c2, kernel_size=1, stride=1, padding=0)
        self.fpn_1 = nn.Conv2d(c2, c1, kernel_size=1, stride=1, padding=0)

        # Detector head
        self.convP = nn.Conv2d(c1, 1, kernel_size=3, stride=1, padding=1)  # different from superpoint

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
        c0 = self.relu(self.conv1b(x))

        c1 = self.avgpool(c0)
        c1 = self.relu(self.conv2a(c1))
        c1 = self.relu(self.conv2b(c1))

        c2 = self.avgpool(c1)
        c2 = self.relu(self.conv3a(c2))
        c2 = self.relu(self.conv3b(c2))

        c3 = self.avgpool(c2)
        c3 = self.relu(self.conv4a(c3))
        c3 = self.relu(self.conv4b(c3))

        # fpn head
        fpn_c3 = self.relu(self.fpn_3(c3))
        fpn_c3 = nn.functional.interpolate(fpn_c3, scale_factor=2, mode="bilinear", align_corners=True)
        fpn_c3 += c2

        fpn_c2 = self.relu(self.fpn_2(fpn_c3))
        fpn_c2 = nn.functional.interpolate(fpn_c2, scale_factor=2, mode="bilinear", align_corners=True)
        fpn_c2 += c1

        fpn_c1 = self.relu(self.fpn_1(fpn_c2))
        fpn_c1 = nn.functional.interpolate(fpn_c1, scale_factor=2, mode="bilinear", align_corners=True)
        fpn_c1 += c0

        # detector head
        heatmap = self.convP(fpn_c1)

        # descriptor head
        cDa = self.relu(self.convDa(c3))
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


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, combines=None, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNet, self).__init__()
        if combines is None:
            combines = [1, 1, 1, 1]
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=1, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        # self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # self.fc = nn.Linear(512 * block.expansion, num_classes)
        # self.compress1 = nn.Conv2d(64, 32, kernel_size=1, stride=1)
        # self.compress2 = nn.Conv2d(128, 32, kernel_size=1, stride=1)
        # self.compress3 = nn.Conv2d(256, 32, kernel_size=1, stride=1)
        # self.compress4 = nn.Conv2d(512, 32, kernel_size=1, stride=1)

        self.fc1 = nn.Linear((combines[0]*64+combines[1]*128+combines[2]*256+combines[3]*512)*block.expansion, 256)
        self.fc2 = nn.Linear(256, 128)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x, point):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        # x = self.maxpool(x)

        c1 = self.layer1(x)
        c2 = self.layer2(c1)
        c3 = self.layer3(c2)
        c4 = self.layer4(c3)

        # compress1 = self.compress1(c1)
        # compress2 = self.compress2(c2)
        # compress3 = self.compress3(c3)
        # compress4 = self.compress4(c4)

        c1_feature = f.grid_sample(
            c1, point, mode="bilinear", padding_mode="border")[:, :, :, 0].transpose(1, 2)
        c2_feature = f.grid_sample(
            c2, point, mode="bilinear", padding_mode="border")[:, :, :, 0].transpose(1, 2)
        c3_feature = f.grid_sample(
            c3, point, mode="bilinear", padding_mode="border")[:, :, :, 0].transpose(1, 2)
        c4_feature = f.grid_sample(
            c4, point, mode="bilinear", padding_mode="border")[:, :, :, 0].transpose(1, 2)

        feature = torch.cat((c1_feature, c2_feature, c3_feature, c4_feature), dim=2)
        feature = self.relu(self.fc1(self.relu(feature)))
        feature = self.fc2(feature)
        feature = feature / torch.norm(feature, dim=2, keepdim=True)

        return feature


class DescriptorExtractor(nn.Module):
    """
    与resnet配套使用的描述子生成器，其作用在于融合多层级的特征
    """
    def __init__(self):
        super(DescriptorExtractor, self).__init__()
        self.fc1 = nn.Linear(128, 128)
        self.fc2 = nn.Linear(128, 128)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(self.relu(x)))
        x = self.fc2(x)
        x = x / torch.norm(x, dim=2, keepdim=True)

        return x


def _resnet(arch, block, layers, pretrained, progress, **kwargs):
    model = ResNet(block, layers, **kwargs)
    return model


def resnet18(pretrained=False, progress=True, **kwargs):
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress,
                   **kwargs)


def resnet34(pretrained=False, progress=True, **kwargs):
    r"""ResNet-34 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet34', BasicBlock, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)


#### 大规模实验准备
###############测试不同stride带来的影响
# original stride resnet
class ResNetS0S2S3S4(nn.Module):

    def __init__(self, block, layers, combines=None, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNetS0S2S3S4, self).__init__()
        if combines is None:
            combines = [1, 1, 1, 1]
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])

        self.fc1 = nn.Linear((combines[0]*64+combines[1]*128+combines[2]*256+combines[3]*512)*block.expansion, 256)
        self.fc2 = nn.Linear(256, 128)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x, point):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        c1 = self.layer1(x)
        c2 = self.layer2(c1)
        c3 = self.layer3(c2)
        c4 = self.layer4(c3)

        c1_feature = f.grid_sample(
            c1, point, mode="bilinear", padding_mode="border")[:, :, :, 0].transpose(1, 2)
        c2_feature = f.grid_sample(
            c2, point, mode="bilinear", padding_mode="border")[:, :, :, 0].transpose(1, 2)
        c3_feature = f.grid_sample(
            c3, point, mode="bilinear", padding_mode="border")[:, :, :, 0].transpose(1, 2)
        c4_feature = f.grid_sample(
            c4, point, mode="bilinear", padding_mode="border")[:, :, :, 0].transpose(1, 2)

        feature = torch.cat((c1_feature, c2_feature, c3_feature, c4_feature), dim=2)
        feature = self.relu(self.fc1(self.relu(feature)))
        feature = self.fc2(feature)
        feature = feature / torch.norm(feature, dim=2, keepdim=True)

        return feature


def resnet18_s0s2s3s4():
    return ResNetS0S2S3S4(BasicBlock, [2, 2, 2, 2])

def resnet34_s0s2s3s4():
    return ResNetS0S2S3S4(BasicBlock, [3, 4, 6, 3])

def resnet50_s0s2s3s4():
    return ResNetS0S2S3S4(Bottleneck, [3, 4, 6, 3])


class ResNetS0S2S3S4Auxiliary256(nn.Module):

    def __init__(self, block, layers, combines=None, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNetS0S2S3S4Auxiliary256, self).__init__()
        if combines is None:
            combines = [1, 1, 1, 1]
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])

        # self.fc1 = nn.Linear((combines[0]*64+combines[1]*128+combines[2]*256+combines[3]*512)*block.expansion, 256)
        # self.fc2 = nn.Linear(256, 128)
        self.fc1 = nn.Linear(64*block.expansion, 64)
        self.fc2 = nn.Linear(128*block.expansion, 128)
        self.fc3 = nn.Linear(256*block.expansion, 256)
        self.fc4 = nn.Linear(512*block.expansion, 512)

        self.final_fc1 = nn.Linear((64+128+256+512)*block.expansion, 256)
        self.final_fc2 = nn.Linear(256, 256)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x, point):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        c1 = self.layer1(x)
        c2 = self.layer2(c1)
        c3 = self.layer3(c2)
        c4 = self.layer4(c3)

        c1_feature = f.grid_sample(
            c1, point, mode="bilinear", padding_mode="border")[:, :, :, 0].transpose(1, 2)
        c2_feature = f.grid_sample(
            c2, point, mode="bilinear", padding_mode="border")[:, :, :, 0].transpose(1, 2)
        c3_feature = f.grid_sample(
            c3, point, mode="bilinear", padding_mode="border")[:, :, :, 0].transpose(1, 2)
        c4_feature = f.grid_sample(
            c4, point, mode="bilinear", padding_mode="border")[:, :, :, 0].transpose(1, 2)

        c1_desp = self.fc1(c1_feature)
        c2_desp = self.fc2(c2_feature)
        c3_desp = self.fc3(c3_feature)
        c4_desp = self.fc4(c4_feature)

        c1_desp = c1_desp / torch.norm(c1_desp, dim=2, keepdim=True)
        c2_desp = c2_desp / torch.norm(c2_desp, dim=2, keepdim=True)
        c3_desp = c3_desp / torch.norm(c3_desp, dim=2, keepdim=True)
        c4_desp = c4_desp / torch.norm(c4_desp, dim=2, keepdim=True)

        feature = torch.cat((c1_feature, c2_feature, c3_feature, c4_feature), dim=2)
        feature = self.final_fc2(self.relu(self.final_fc1(feature)))
        desp = feature / torch.norm(feature, dim=2, keepdim=True)

        if self.training:
            return desp, c1_desp, c2_desp, c3_desp, c4_desp
        else:
            return desp


def resnet18_s0s2s3s4_auxiliary_256():
    return ResNetS0S2S3S4Auxiliary256(BasicBlock, [2, 2, 2, 2])


def resnet34_s0s2s3s4_auxiliary_256():
    return ResNetS0S2S3S4Auxiliary256(BasicBlock, [3, 4, 6, 3])


class ResNetS0S2S3S4Auxiliary(nn.Module):

    def __init__(self, block, layers, combines=None, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNetS0S2S3S4Auxiliary, self).__init__()
        if combines is None:
            combines = [1, 1, 1, 1]
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])

        # self.fc1 = nn.Linear((combines[0]*64+combines[1]*128+combines[2]*256+combines[3]*512)*block.expansion, 256)
        # self.fc2 = nn.Linear(256, 128)
        self.fc1 = nn.Linear(64*block.expansion, 64)
        self.fc2 = nn.Linear(128*block.expansion, 128)
        self.fc3 = nn.Linear(256*block.expansion, 256)
        self.fc4 = nn.Linear(512*block.expansion, 512)

        self.final_fc1 = nn.Linear((64+128+256+512)*block.expansion, 256)
        self.final_fc2 = nn.Linear(256, 128)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x, point):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        c1 = self.layer1(x)
        c2 = self.layer2(c1)
        c3 = self.layer3(c2)
        c4 = self.layer4(c3)

        c1_feature = f.grid_sample(
            c1, point, mode="bilinear", padding_mode="border")[:, :, :, 0].transpose(1, 2)
        c2_feature = f.grid_sample(
            c2, point, mode="bilinear", padding_mode="border")[:, :, :, 0].transpose(1, 2)
        c3_feature = f.grid_sample(
            c3, point, mode="bilinear", padding_mode="border")[:, :, :, 0].transpose(1, 2)
        c4_feature = f.grid_sample(
            c4, point, mode="bilinear", padding_mode="border")[:, :, :, 0].transpose(1, 2)

        c1_desp = self.fc1(c1_feature)
        c2_desp = self.fc2(c2_feature)
        c3_desp = self.fc3(c3_feature)
        c4_desp = self.fc4(c4_feature)

        c1_desp = c1_desp / torch.norm(c1_desp, dim=2, keepdim=True)
        c2_desp = c2_desp / torch.norm(c2_desp, dim=2, keepdim=True)
        c3_desp = c3_desp / torch.norm(c3_desp, dim=2, keepdim=True)
        c4_desp = c4_desp / torch.norm(c4_desp, dim=2, keepdim=True)

        feature = torch.cat((c1_feature, c2_feature, c3_feature, c4_feature), dim=2)
        feature = self.final_fc2(self.relu(self.final_fc1(feature)))
        desp = feature / torch.norm(feature, dim=2, keepdim=True)

        if self.training:
            return desp, c1_desp, c2_desp, c3_desp, c4_desp
        else:
            return desp


def resnet18_s0s2s3s4_auxiliary():
    return ResNetS0S2S3S4Auxiliary(BasicBlock, [2, 2, 2, 2])


def resnet34_s0s2s3s4_auxiliary():
    return ResNetS0S2S3S4Auxiliary(BasicBlock, [3, 4, 6, 3])


class ResNet256S0S2S3S4(nn.Module):

    def __init__(self, block, layers, combines=None, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNet256S0S2S3S4, self).__init__()
        if combines is None:
            combines = [1, 1, 1, 1]
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])

        self.fc1 = nn.Linear((combines[0]*64+combines[1]*128+combines[2]*256+combines[3]*512)*block.expansion, 256)
        self.fc2 = nn.Linear(256, 256)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x, point):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        c1 = self.layer1(x)
        c2 = self.layer2(c1)
        c3 = self.layer3(c2)
        c4 = self.layer4(c3)

        c1_feature = f.grid_sample(
            c1, point, mode="bilinear", padding_mode="border")[:, :, :, 0].transpose(1, 2)
        c2_feature = f.grid_sample(
            c2, point, mode="bilinear", padding_mode="border")[:, :, :, 0].transpose(1, 2)
        c3_feature = f.grid_sample(
            c3, point, mode="bilinear", padding_mode="border")[:, :, :, 0].transpose(1, 2)
        c4_feature = f.grid_sample(
            c4, point, mode="bilinear", padding_mode="border")[:, :, :, 0].transpose(1, 2)

        feature = torch.cat((c1_feature, c2_feature, c3_feature, c4_feature), dim=2)
        feature = self.relu(self.fc1(self.relu(feature)))
        feature = self.fc2(feature)
        feature = feature / torch.norm(feature, dim=2, keepdim=True)

        return feature


def resnet18_s0s2s3s4_256():
    return ResNet256S0S2S3S4(BasicBlock, [2, 2, 2, 2])


class ResNet512S0S2S3S4(nn.Module):

    def __init__(self, block, layers, combines=None, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNet512S0S2S3S4, self).__init__()
        if combines is None:
            combines = [1, 1, 1, 1]
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])

        self.fc1 = nn.Linear((combines[0]*64+combines[1]*128+combines[2]*256+combines[3]*512)*block.expansion, 512)
        self.fc2 = nn.Linear(512, 512)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x, point):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        c1 = self.layer1(x)
        c2 = self.layer2(c1)
        c3 = self.layer3(c2)
        c4 = self.layer4(c3)

        c1_feature = f.grid_sample(
            c1, point, mode="bilinear", padding_mode="border")[:, :, :, 0].transpose(1, 2)
        c2_feature = f.grid_sample(
            c2, point, mode="bilinear", padding_mode="border")[:, :, :, 0].transpose(1, 2)
        c3_feature = f.grid_sample(
            c3, point, mode="bilinear", padding_mode="border")[:, :, :, 0].transpose(1, 2)
        c4_feature = f.grid_sample(
            c4, point, mode="bilinear", padding_mode="border")[:, :, :, 0].transpose(1, 2)

        feature = torch.cat((c1_feature, c2_feature, c3_feature, c4_feature), dim=2)
        feature = self.relu(self.fc1(self.relu(feature)))
        feature = self.fc2(feature)
        feature = feature / torch.norm(feature, dim=2, keepdim=True)

        return feature


def resnet18_s0s2s3s4_512():
    return ResNet512S0S2S3S4(BasicBlock, [2, 2, 2, 2])


class ResNetS0S2S3S4C4(ResNetS0S2S3S4):

    def __init__(self, block, layers, combines=None, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNetS0S2S3S4C4, self).__init__(block, layers, combines=[0, 0, 0, 1])

    def forward(self, x, point):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        c1 = self.layer1(x)
        c2 = self.layer2(c1)
        c3 = self.layer3(c2)
        c4 = self.layer4(c3)

        c4_feature = f.grid_sample(c4, point, mode="bilinear", padding_mode="border")[:, :, :, 0].transpose(1, 2)

        feature = c4_feature
        feature = self.relu(self.fc1(self.relu(feature)))
        feature = self.fc2(feature)
        feature = feature / torch.norm(feature, dim=2, keepdim=True)

        return feature


def resnet18_s0s2s3s4_c4():
    return ResNetS0S2S3S4C4(BasicBlock, [2, 2, 2, 2])

def resnet34_s0s2s3s4_c4():
    return ResNetS0S2S3S4(BasicBlock, [3, 4, 6, 3])


class ResNetS0S2S3S4MaxPool(nn.Module):

    def __init__(self, block, layers, combines=None, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNetS0S2S3S4MaxPool, self).__init__()
        if combines is None:
            combines = [1, 1, 1, 1]
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])

        self.fc1 = nn.Linear((combines[0]*64+combines[1]*128+combines[2]*256+combines[3]*512)*block.expansion, 256)
        self.fc2 = nn.Linear(256, 128)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x, point):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        c1 = self.layer1(x)
        c2 = self.layer2(c1)
        c3 = self.layer3(c2)
        c4 = self.layer4(c3)

        c1_feature = f.grid_sample(
            c1, point, mode="bilinear", padding_mode="border")[:, :, :, 0].transpose(1, 2)
        c2_feature = f.grid_sample(
            c2, point, mode="bilinear", padding_mode="border")[:, :, :, 0].transpose(1, 2)
        c3_feature = f.grid_sample(
            c3, point, mode="bilinear", padding_mode="border")[:, :, :, 0].transpose(1, 2)
        c4_feature = f.grid_sample(
            c4, point, mode="bilinear", padding_mode="border")[:, :, :, 0].transpose(1, 2)

        feature = torch.cat((c1_feature, c2_feature, c3_feature, c4_feature), dim=2)
        feature = self.relu(self.fc1(self.relu(feature)))
        feature = self.fc2(feature)
        feature = feature / torch.norm(feature, dim=2, keepdim=True)

        return feature


def resnet18_s0s2s3s4_maxpool():
    return ResNetS0S2S3S4MaxPool(BasicBlock, [2, 2, 2, 2])


class ResNetS0S2S3S4AvgPool(nn.Module):

    def __init__(self, block, layers, combines=None, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNetS0S2S3S4AvgPool, self).__init__()
        if combines is None:
            combines = [1, 1, 1, 1]
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.avgpool = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])

        self.fc1 = nn.Linear((combines[0]*64+combines[1]*128+combines[2]*256+combines[3]*512)*block.expansion, 256)
        self.fc2 = nn.Linear(256, 128)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x, point):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.avgpool(x)

        c1 = self.layer1(x)
        c2 = self.layer2(c1)
        c3 = self.layer3(c2)
        c4 = self.layer4(c3)

        c1_feature = f.grid_sample(
            c1, point, mode="bilinear", padding_mode="border")[:, :, :, 0].transpose(1, 2)
        c2_feature = f.grid_sample(
            c2, point, mode="bilinear", padding_mode="border")[:, :, :, 0].transpose(1, 2)
        c3_feature = f.grid_sample(
            c3, point, mode="bilinear", padding_mode="border")[:, :, :, 0].transpose(1, 2)
        c4_feature = f.grid_sample(
            c4, point, mode="bilinear", padding_mode="border")[:, :, :, 0].transpose(1, 2)

        feature = torch.cat((c1_feature, c2_feature, c3_feature, c4_feature), dim=2)
        feature = self.relu(self.fc1(self.relu(feature)))
        feature = self.fc2(feature)
        feature = feature / torch.norm(feature, dim=2, keepdim=True)

        return feature


def resnet18_s0s2s3s4_avgpool():
    return ResNetS0S2S3S4AvgPool(BasicBlock, [2, 2, 2, 2])


# 第二次大规模实验
class ResNetC1C2C3C4(nn.Module):

    def __init__(self, block, layers, combines=None, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNetC1C2C3C4, self).__init__()
        if combines is None:
            combines = [1, 1, 1, 1]
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])

        self.fc1 = nn.Linear((combines[0]*64+combines[1]*128+combines[2]*256+combines[3]*512)*block.expansion, 256)
        self.fc2 = nn.Linear(256, 128)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x, point):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        c1 = self.layer1(x)
        c2 = self.layer2(c1)
        c3 = self.layer3(c2)
        c4 = self.layer4(c3)

        c1_feature = f.grid_sample(
            c1, point, mode="bilinear", padding_mode="border")[:, :, :, 0].transpose(1, 2)
        c2_feature = f.grid_sample(
            c2, point, mode="bilinear", padding_mode="border")[:, :, :, 0].transpose(1, 2)
        c3_feature = f.grid_sample(
            c3, point, mode="bilinear", padding_mode="border")[:, :, :, 0].transpose(1, 2)
        c4_feature = f.grid_sample(
            c4, point, mode="bilinear", padding_mode="border")[:, :, :, 0].transpose(1, 2)

        feature = torch.cat((c1_feature, c2_feature, c3_feature, c4_feature), dim=2)
        feature = self.relu(self.fc1(self.relu(feature)))
        feature = self.fc2(feature)
        feature = feature / torch.norm(feature, dim=2, keepdim=True)

        return feature


def resnet18_c1c2c3c4():
    return ResNetC1C2C3C4(BasicBlock, [2, 2, 2, 2])


class ResNetC2C3C4(ResNetC1C2C3C4):

    def __init__(self, block, layers):
        super(ResNetC2C3C4, self).__init__(block, layers, combines=[0, 1, 1, 1])

    def forward(self, x, point):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        c1 = self.layer1(x)
        c2 = self.layer2(c1)
        c3 = self.layer3(c2)
        c4 = self.layer4(c3)

        c2_feature = f.grid_sample(
            c2, point, mode="bilinear", padding_mode="border")[:, :, :, 0].transpose(1, 2)
        c3_feature = f.grid_sample(
            c3, point, mode="bilinear", padding_mode="border")[:, :, :, 0].transpose(1, 2)
        c4_feature = f.grid_sample(
            c4, point, mode="bilinear", padding_mode="border")[:, :, :, 0].transpose(1, 2)

        feature = torch.cat((c2_feature, c3_feature, c4_feature), dim=2)
        feature = self.relu(self.fc1(self.relu(feature)))
        feature = self.fc2(feature)
        feature = feature / torch.norm(feature, dim=2, keepdim=True)

        return feature


def resnet18_c2c3c4():
    return ResNetC2C3C4(BasicBlock, [2, 2, 2, 2])


class ResNetC3C4(ResNetC1C2C3C4):

    def __init__(self, block, layers):
        super(ResNetC3C4, self).__init__(block, layers, combines=[0, 0, 1, 1])

    def forward(self, x, point):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        c1 = self.layer1(x)
        c2 = self.layer2(c1)
        c3 = self.layer3(c2)
        c4 = self.layer4(c3)

        c3_feature = f.grid_sample(
            c3, point, mode="bilinear", padding_mode="border")[:, :, :, 0].transpose(1, 2)
        c4_feature = f.grid_sample(
            c4, point, mode="bilinear", padding_mode="border")[:, :, :, 0].transpose(1, 2)

        feature = torch.cat((c3_feature, c4_feature), dim=2)
        feature = self.relu(self.fc1(self.relu(feature)))
        feature = self.fc2(feature)
        feature = feature / torch.norm(feature, dim=2, keepdim=True)

        return feature


def resnet18_c3c4():
    return ResNetC3C4(BasicBlock, [2, 2, 2, 2])


class ResNetC4(ResNetC1C2C3C4):

    def __init__(self, block, layers):
        super(ResNetC4, self).__init__(block, layers, combines=[0, 0, 0, 1])

    def forward(self, x, point):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        c1 = self.layer1(x)
        c2 = self.layer2(c1)
        c3 = self.layer3(c2)
        c4 = self.layer4(c3)

        c4_feature = f.grid_sample(
            c4, point, mode="bilinear", padding_mode="border")[:, :, :, 0].transpose(1, 2)

        feature = c4_feature
        feature = self.relu(self.fc1(self.relu(feature)))
        feature = self.fc2(feature)
        feature = feature / torch.norm(feature, dim=2, keepdim=True)

        return feature


def resnet18_c4():
    return ResNetC4(BasicBlock, [2, 2, 2, 2])


class ResNetC1C2C3C4MaxPool(nn.Module):

    def __init__(self, block, layers, combines=None, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNetC1C2C3C4MaxPool, self).__init__()
        if combines is None:
            combines = [1, 1, 1, 1]
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])

        self.fc1 = nn.Linear((combines[0]*64+combines[1]*128+combines[2]*256+combines[3]*512)*block.expansion, 256)
        self.fc2 = nn.Linear(256, 128)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x, point):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        c1 = self.layer1(x)
        c2 = self.layer2(c1)
        c3 = self.layer3(c2)
        c4 = self.layer4(c3)

        c1_feature = f.grid_sample(
            c1, point, mode="bilinear", padding_mode="border")[:, :, :, 0].transpose(1, 2)
        c2_feature = f.grid_sample(
            c2, point, mode="bilinear", padding_mode="border")[:, :, :, 0].transpose(1, 2)
        c3_feature = f.grid_sample(
            c3, point, mode="bilinear", padding_mode="border")[:, :, :, 0].transpose(1, 2)
        c4_feature = f.grid_sample(
            c4, point, mode="bilinear", padding_mode="border")[:, :, :, 0].transpose(1, 2)

        feature = torch.cat((c1_feature, c2_feature, c3_feature, c4_feature), dim=2)
        feature = self.relu(self.fc1(self.relu(feature)))
        feature = self.fc2(feature)
        feature = feature / torch.norm(feature, dim=2, keepdim=True)

        return feature


def resnet18_c1c2c3c4_maxpool():
    return ResNetC1C2C3C4MaxPool(BasicBlock, [2, 2, 2, 2])


class ResNetC1C2C3C4AvgPool(nn.Module):

    def __init__(self, block, layers, combines=None, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNetC1C2C3C4AvgPool, self).__init__()
        if combines is None:
            combines = [1, 1, 1, 1]
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.avgpool = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])

        self.fc1 = nn.Linear((combines[0]*64+combines[1]*128+combines[2]*256+combines[3]*512)*block.expansion, 256)
        self.fc2 = nn.Linear(256, 128)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x, point):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.avgpool(x)

        c1 = self.layer1(x)
        c2 = self.layer2(c1)
        c3 = self.layer3(c2)
        c4 = self.layer4(c3)

        c1_feature = f.grid_sample(
            c1, point, mode="bilinear", padding_mode="border")[:, :, :, 0].transpose(1, 2)
        c2_feature = f.grid_sample(
            c2, point, mode="bilinear", padding_mode="border")[:, :, :, 0].transpose(1, 2)
        c3_feature = f.grid_sample(
            c3, point, mode="bilinear", padding_mode="border")[:, :, :, 0].transpose(1, 2)
        c4_feature = f.grid_sample(
            c4, point, mode="bilinear", padding_mode="border")[:, :, :, 0].transpose(1, 2)

        feature = torch.cat((c1_feature, c2_feature, c3_feature, c4_feature), dim=2)
        feature = self.relu(self.fc1(self.relu(feature)))
        feature = self.fc2(feature)
        feature = feature / torch.norm(feature, dim=2, keepdim=True)

        return feature


def resnet18_c1c2c3c4_avgpool():
    return ResNetC1C2C3C4AvgPool(BasicBlock, [2, 2, 2, 2])


