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


class STMegPointNet(BaseMegPointNet):

    def __init__(self):
        super(STMegPointNet, self).__init__()

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

    def reinitialize(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                m.reset_parameters()
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

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

        return logit, prob, desc


class STBNMegPointNet(nn.Module):

    def __init__(self):
        super(STBNMegPointNet, self).__init__()
        self.relu = torch.nn.ReLU(inplace=True)
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        c1, c2, c3, c4, c5, d1 = 64, 64, 128, 128, 256, 256
        # Shared Encoder.
        self.conv1a = nn.Conv2d(1, c1, kernel_size=3, stride=1, padding=1)
        self.bn1a = nn.BatchNorm2d(c1, affine=False)
        self.conv1b = nn.Conv2d(c1, c1, kernel_size=3, stride=1, padding=1)
        self.bn1b = nn.BatchNorm2d(c1, affine=False)
        self.conv2a = nn.Conv2d(c1, c2, kernel_size=3, stride=1, padding=1)
        self.bn2a = nn.BatchNorm2d(c2, affine=False)
        self.conv2b = nn.Conv2d(c2, c2, kernel_size=3, stride=1, padding=1)
        self.bn2b = nn.BatchNorm2d(c2, affine=False)
        self.conv3a = nn.Conv2d(c2, c3, kernel_size=3, stride=1, padding=1)
        self.bn3a = nn.BatchNorm2d(c3, affine=False)
        self.conv3b = nn.Conv2d(c3, c3, kernel_size=3, stride=1, padding=1)
        self.bn3b = nn.BatchNorm2d(c3, affine=False)
        self.conv4a = nn.Conv2d(c3, c4, kernel_size=3, stride=1, padding=1)
        self.bn4a = nn.BatchNorm2d(c4, affine=False)
        self.conv4b = nn.Conv2d(c4, c4, kernel_size=3, stride=1, padding=1)
        self.bn4b = nn.BatchNorm2d(c4, affine=False)
        # Detector Head.
        self.convPa = nn.Conv2d(c4, c5, kernel_size=3, stride=1, padding=1)
        self.bnPa = nn.BatchNorm2d(c5, affine=False)
        self.convPb = nn.Conv2d(c5, 65, kernel_size=1, stride=1, padding=0)
        # Descriptor Head.
        self.convDa = nn.Conv2d(c4, c5, kernel_size=3, stride=1, padding=1)
        self.bnDa = nn.BatchNorm2d(c5, affine=False)
        self.convDb = nn.Conv2d(c5, d1, kernel_size=1, stride=1, padding=0)

        self.softmax = nn.Softmax(dim=1)

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

    def reinitialize(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                m.reset_parameters()
            #     nn.init.constant_(m.weight, 1)
            #     nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.relu(self.bn1a(self.conv1a(x)))
        x = self.relu(self.bn1b(self.conv1b(x)))
        x = self.pool(x)
        x = self.relu(self.bn2a(self.conv2a(x)))
        x = self.relu(self.bn2b(self.conv2b(x)))
        x = self.pool(x)
        x = self.relu(self.bn3a(self.conv3a(x)))
        x = self.relu(self.bn3b(self.conv3b(x)))
        x = self.pool(x)
        x = self.relu(self.bn4a(self.conv4a(x)))
        feature = self.relu(self.bn4b(self.conv4b(x)))

        # detect head
        cPa = self.relu(self.bnPa(self.convPa(feature)))
        logit = self.convPb(cPa)
        prob = self.softmax(logit)[:, :-1, :, :]

        # descriptor head
        cDa = self.relu(self.bnDa(self.convDa(feature)))
        desc = self.convDb(cDa)
        dn = torch.norm(desc, p=2, dim=1, keepdim=True)
        desc = desc.div(dn)

        return logit, prob, desc


class Generator(nn.Module):

    def __init__(self):
        super(Generator, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.tanh = nn.Tanh()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        c1, c2, c3, c4 = 64, 64, 128, 128

        # Decoder.
        self.upconv3a = nn.Conv2d(c4, c3, kernel_size=3, stride=1, padding=1)
        self.upconv3b = nn.Conv2d(c3, c2, kernel_size=3, stride=1, padding=1)
        self.upconv2a = nn.Conv2d(c2, c2, kernel_size=3, stride=1, padding=1)
        self.upconv2b = nn.Conv2d(c2, c1, kernel_size=3, stride=1, padding=1)
        self.upconv1a = nn.Conv2d(c1, c1, kernel_size=3, stride=1, padding=1)
        self.upconv1b = nn.Conv2d(c1, 1, kernel_size=3, stride=1, padding=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, feature):
        # decoder part
        ux = f.interpolate(feature, scale_factor=2, mode='bilinear', align_corners=False)
        ux = self.relu(self.upconv3a(ux))
        ux = self.relu(self.upconv3b(ux))

        ux = f.interpolate(ux, scale_factor=2, mode='bilinear', align_corners=False)
        ux = self.relu(self.upconv2a(ux))
        ux = self.relu(self.upconv2b(ux))

        ux = f.interpolate(ux, scale_factor=2, mode='bilinear', align_corners=False)
        up_feature = self.relu(self.upconv1a(ux))
        recovered_image = self.tanh(self.upconv1b(up_feature))

        return recovered_image


class Discriminator(nn.Module):

    def __init__(self, input_dim=128):
        super(Discriminator, self).__init__()
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.conv1 = nn.Conv2d(input_dim, 128, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)
        self.pooling = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, 1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        x = self.leaky_relu(self.conv1(x))
        x = self.leaky_relu(self.conv2(x))
        x = self.leaky_relu(self.conv3(x))
        x = self.pooling(x).squeeze()
        logit = self.fc(x)
        return logit


class MegPointHeatmap(nn.Module):

    def __init__(self):
        super(MegPointHeatmap, self).__init__()
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

        # Decoder.
        self.upconv3a = nn.Conv2d(c4, c4, kernel_size=3, stride=1, padding=1)
        self.upconv3b = nn.Conv2d(c4, c3, kernel_size=3, stride=1, padding=1)
        self.upconv2a = nn.Conv2d(c3, c3, kernel_size=3, stride=1, padding=1)
        self.upconv2b = nn.Conv2d(c3, c2, kernel_size=3, stride=1, padding=1)
        self.upconv1a = nn.Conv2d(c2, c2, kernel_size=3, stride=1, padding=1)
        self.upconv1b = nn.Conv2d(c2, c1, kernel_size=3, stride=1, padding=1)

        # Detector head
        self.pred = nn.Conv2d(c1, 1, kernel_size=1, stride=1, padding=0)

        # Descriptor Head.
        self.convDa = nn.Conv2d(c4, c5, kernel_size=3, stride=1, padding=1)
        self.convDb = nn.Conv2d(c5, d1, kernel_size=1, stride=1, padding=0)

        self.soffmax = nn.Softmax(dim=1)

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

        # decoder part
        ux = f.interpolate(feature, scale_factor=2, mode='bilinear', align_corners=True)
        ux = self.relu(self.upconv3a(ux))
        ux = self.relu(self.upconv3b(ux))

        ux = f.interpolate(ux, scale_factor=2, mode="bilinear", align_corners=True)
        ux = self.relu(self.upconv2a(ux))
        ux = self.relu(self.upconv2b(ux))

        ux = f.interpolate(ux, scale_factor=2, mode="bilinear", align_corners=True)
        ux = self.relu(self.upconv1a(ux))
        ux = self.relu(self.upconv1b(ux))

        # detector head
        pred = self.pred(ux)

        # descriptor head
        cDa = self.relu(self.convDa(feature))
        cDb = self.convDb(cDa)

        dn = torch.norm(cDb, p=2, dim=1, keepdim=True)
        desp = cDb.div(dn)

        return pred, desp


class HardDetectionModule(nn.Module):
    def __init__(self, edge_threshold=5, nms_threshold=4):
        super(HardDetectionModule, self).__init__()

        self.edge_threshold = edge_threshold
        self.pooling_size = int(nms_threshold * 2 + 1)
        self.padding_size = int(self.pooling_size//2)

        self.dii_filter = torch.tensor(
            [[0, 1., 0], [0, -2., 0], [0, 1., 0]]
        ).reshape(1, 1, 3, 3)
        self.dij_filter = 0.25 * torch.tensor(
            [[1., 0, -1.], [0, 0., 0], [-1., 0, 1.]]
        ).reshape(1, 1, 3, 3)
        self.djj_filter = torch.tensor(
            [[0, 0, 0], [1., -2., 1.], [0, 0, 0]]
        ).reshape(1, 1, 3, 3)

    def forward(self, x):
        # device = x.device

        depth_wise_max = torch.max(x, dim=1, keepdim=True)[0]
        is_depth_wise_max = torch.where(
            torch.eq(x, depth_wise_max), torch.ones_like(x), torch.zeros_like(x)
        ).to(torch.bool)

        local_max = f.max_pool2d(x, self.pooling_size, stride=1, padding=self.padding_size)
        is_local_max = torch.where(
            torch.eq(x, local_max), torch.ones_like(x), torch.zeros_like(x)
        ).to(torch.bool)

        prob = torch.where(
            is_depth_wise_max & is_local_max, x, torch.zeros_like(x)
        ).sum(dim=1)

        # dii = f.conv2d(x, self.dii_filter.to(device), padding=1)
        # dij = f.conv2d(x, self.dij_filter.to(device), padding=1)
        # djj = f.conv2d(x, self.djj_filter.to(device), padding=1)

        # det = dii * djj - dij * dij
        # tr = dii + djj

        # threshold = (self.edge_threshold + 1) ** 2 / self.edge_threshold
        # is_not_edge = (tr * tr / det <= threshold) & (det > 0)

        # detected = is_depth_wise_max & is_local_max & is_not_edge

        return prob






