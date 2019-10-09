# 
# Created by ZhangYuyang on 2019/9/19
#
import torch
import torch.nn as nn
import torch.nn.functional as f

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
        x = self.relu(self.conv4b(x))

        # detect head
        cPa = self.relu(self.convPa(x))
        logit = self.convPb(cPa)
        prob = self.softmax(logit)[:, :-1, :, :]

        # descriptor head
        cDa = self.relu(self.convDa(x))
        feature = self.convDb(cDa)

        dn = torch.norm(feature, p=2, dim=1, keepdim=True)
        desc = feature.div(dn)

        return logit, prob, desc


class EncoderDecoderMegPoint(nn.Module):

    def __init__(self):
        super(EncoderDecoderMegPoint, self).__init__()
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

        # Descriptor head
        self.convDa = nn.Conv2d(c4, c5, kernel_size=3, stride=1, padding=1)
        self.convDb = nn.Conv2d(c5, d1, kernel_size=1, stride=1, padding=0)

        # Decoder.
        self.upconv3a = nn.Conv2d(c4, c3, kernel_size=3, stride=1, padding=1)
        self.upconv3b = nn.Conv2d(c3, c2, kernel_size=3, stride=1, padding=1)
        self.upconv2a = nn.Conv2d(c2, c2, kernel_size=3, stride=1, padding=1)
        self.upconv2b = nn.Conv2d(c2, c1, kernel_size=3, stride=1, padding=1)
        self.upconv1a = nn.Conv2d(c1, c1, kernel_size=3, stride=1, padding=1)
        self.upconv1b = nn.Conv2d(c1, 1, kernel_size=3, stride=1, padding=1)

    def forward(self, x):

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
        ux = f.interpolate(feature, scale_factor=2, mode='bilinear', align_corners=False)
        ux = self.relu(self.upconv3a(ux))
        ux = self.relu(self.upconv3b(ux))

        ux = f.interpolate(ux, scale_factor=2, mode='bilinear', align_corners=False)
        ux = self.relu(self.upconv2a(ux))
        ux = self.relu(self.upconv2b(ux))

        ux = f.interpolate(ux, scale_factor=2, mode='bilinear', align_corners=False)
        up_feature = self.relu(self.upconv1a(ux))
        recovered_image = self.tanh(self.upconv1b(up_feature))

        # descriptor head
        cDa = self.relu(self.convDa(feature))
        descriptor = self.convDb(cDa)
        dn = torch.norm(descriptor, p=2, dim=1, keepdim=True)
        descriptor = descriptor.div(dn)

        return recovered_image, descriptor, up_feature






