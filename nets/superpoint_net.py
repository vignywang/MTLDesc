#
# Created by ZhangYuyang on 2019/8/12
#
import torch
import torch.nn as nn
from torch.autograd import Function
import torch.nn.functional as f

# from torchvision.models import ResNet


class Hash(Function):

    @staticmethod
    def forward(ctx, input):
        return torch.sign(input)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


def hash_layer(input):
    return Hash.apply(input)


class BasicSuperPointNet(nn.Module):

    def __init__(self):
        super(BasicSuperPointNet, self).__init__()
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

        self.softmax = nn.Softmax(dim=1)
        self.tanh = nn.Tanh()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')


class MagicPointNet(BasicSuperPointNet):

    def __init__(self):
        super(MagicPointNet, self).__init__()

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
        logit = self.convPb(cPa)
        prob = self.softmax(logit)[:, :-1, :, :]

        return logit, prob


class SuperPointNetFloat(BasicSuperPointNet):

    def __init__(self):
        super(SuperPointNetFloat, self).__init__()

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
        # desc = hash_layer(feature)

        return logit, desc, prob, None


class SuperPointNetVisualize(BasicSuperPointNet):

    def __init__(self):
        super(SuperPointNetVisualize, self).__init__()

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
        prob = self.softmax(logit)[:, :-1, :, :]  # todo

        # descriptor head
        cDa = self.relu(self.convDa(x))
        feature = self.convDb(cDa)

        dn = torch.norm(feature, p=2, dim=1, keepdim=True)
        desc = feature.div(dn)
        # desc = hash_layer(feature)

        return logit, desc, prob


class SuperPointNetBinary(BasicSuperPointNet):

    def __init__(self):
        super(SuperPointNetBinary, self).__init__()

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
        feature = feature.div(dn)
        desc = hash_layer(feature)

        return logit, desc, prob, feature


class SuperPointNetTanh(BasicSuperPointNet):

    def __init__(self):
        super(SuperPointNetTanh, self).__init__()

    def forward(self, x, do_scale=False, epoch_idx=-1):
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
        cDa = self.bnDa(cDa)  # bn能够使之得到有效的输出
        feature = self.convDb(cDa)
        if do_scale and epoch_idx >= 0:
            scale = 1.0 + epoch_idx // 10  # 每10个epoch增加1
        else:
            scale = 1.0
        desc = self.tanh(scale*feature)

        return logit, desc, prob, feature


class SuperPointNetBackbone(nn.Module):

    def __init__(self):
        super(SuperPointNetBackbone, self).__init__()
        self.relu = torch.nn.ReLU(inplace=True)
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        # Shared Encoder.
        self.conv1a = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)
        self.conv1b = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.conv2a = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.conv2b = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.conv3a = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv3b = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.conv4a = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.conv4b = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)

        self.softmax = nn.Softmax(dim=1)

        self.heatmap = nn.Conv2d((64 + 16 + 8 + 2), 1, kernel_size=3, stride=1, padding=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        x = self.relu(self.conv1a(x))
        c1 = self.relu(self.conv1b(x))  # 64

        c2 = self.pool(c1)
        c2 = self.relu(self.conv2a(c2))
        c2 = self.relu(self.conv2b(c2))  # 64

        c3 = self.pool(c2)
        c3 = self.relu(self.conv3a(c3))
        c3 = self.relu(self.conv3b(c3))  # 128

        c4 = self.pool(c3)
        c4 = self.relu(self.conv4a(c4))
        c4 = self.relu(self.conv4b(c4))  # 128

        heatmap1 = c1  # [h,w,64]
        heatmap2 = f.pixel_shuffle(c2, 2)  # [h,w,16]
        heatmap3 = f.pixel_shuffle(c3, 4)  # [h,w,8]
        heatmap4 = f.pixel_shuffle(c4, 8)  # [h,w,2]
        heatmap = self.heatmap(torch.cat((heatmap1, heatmap2, heatmap3, heatmap4), dim=1))

        return heatmap, c1, c2, c3, c4


class SuperPointNetBackbone3(nn.Module):

    def __init__(self):
        super(SuperPointNetBackbone3, self).__init__()
        self.relu = torch.nn.ReLU(inplace=True)
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        # Shared Encoder.
        self.conv1a = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.conv1b = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.conv2a = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.conv2b = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.conv3a = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv3b = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.conv4a = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.conv4b = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)

        self.softmax = nn.Softmax(dim=1)

        self.heatmap = nn.Conv2d((64 + 16 + 8 + 2), 1, kernel_size=3, stride=1, padding=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        x = self.relu(self.conv1a(x))
        c1 = self.relu(self.conv1b(x))  # 64

        c2 = self.pool(c1)
        c2 = self.relu(self.conv2a(c2))
        c2 = self.relu(self.conv2b(c2))  # 64

        c3 = self.pool(c2)
        c3 = self.relu(self.conv3a(c3))
        c3 = self.relu(self.conv3b(c3))  # 128

        c4 = self.pool(c3)
        c4 = self.relu(self.conv4a(c4))
        c4 = self.relu(self.conv4b(c4))  # 128

        heatmap1 = c1  # [h,w,64]
        heatmap2 = f.pixel_shuffle(c2, 2)  # [h,w,16]
        heatmap3 = f.pixel_shuffle(c3, 4)  # [h,w,8]
        heatmap4 = f.pixel_shuffle(c4, 8)  # [h,w,2]
        heatmap = self.heatmap(torch.cat((heatmap1, heatmap2, heatmap3, heatmap4), dim=1))

        return heatmap, c1, c2, c3, c4


class SuperPointNetBackboneSeg(nn.Module):

    def __init__(self):
        super(SuperPointNetBackboneSeg, self).__init__()
        self.relu = torch.nn.ReLU(inplace=True)
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        # Shared Encoder.
        self.conv1a = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.conv1b = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.conv2a = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.conv2b = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.conv3a = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv3b = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.conv4a = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.conv4b = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)

        self.softmax = nn.Softmax(dim=1)

        self.heatmap = nn.Conv2d((64 + 16 + 8 + 2), 1, kernel_size=3, stride=1, padding=1)

        self.seg1 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.seg2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.seg3 = nn.Conv2d(256, 182, kernel_size=3, stride=1, padding=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        x = self.relu(self.conv1a(x))
        c1 = self.relu(self.conv1b(x))  # 64

        c2 = self.pool(c1)
        c2 = self.relu(self.conv2a(c2))
        c2 = self.relu(self.conv2b(c2))  # 64

        c3 = self.pool(c2)
        c3 = self.relu(self.conv3a(c3))
        c3 = self.relu(self.conv3b(c3))  # 128

        c4 = self.pool(c3)
        c4 = self.relu(self.conv4a(c4))
        c4 = self.relu(self.conv4b(c4))  # 128

        seg1 = self.relu(self.seg1(c4))
        seg2 = self.seg2(seg1)
        seg_logit = self.seg3(self.relu(seg2))

        heatmap1 = c1 # [h,w,64]
        heatmap2 = f.pixel_shuffle(c2, 2)  # [h,w,16]
        heatmap3 = f.pixel_shuffle(c3, 4)  # [h,w,8]
        heatmap4 = f.pixel_shuffle(c4, 8)  # [h,w,2]
        heatmap = self.heatmap(torch.cat((heatmap1, heatmap2, heatmap3, heatmap4), dim=1))

        if self.training:
            return heatmap, c1, c2, c3, c4, seg_logit
        else:
            seg_feature = seg2 / torch.norm(seg2, 2, dim=1, keepdim=True)
            return heatmap, c1, c2, c3, c4, seg_feature


class SuperPointExtractor(nn.Module):

    def __init__(self, combines=None):
        super(SuperPointExtractor, self).__init__()
        if combines is None:
            combines = [1, 1, 1, 1]
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear((64*combines[0]+64*combines[1]+128*combines[2]+128*combines[3]), 256)
        self.fc2 = nn.Linear(256, 128)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")

    def __call__(self, feature):
        feature = self.relu(self.fc1(self.relu(feature)))
        feature = self.fc2(feature)
        desp = feature / torch.norm(feature, dim=2, keepdim=True)

        return desp


class SuperPointExtractor256(nn.Module):

    def __init__(self, combines=None):
        super(SuperPointExtractor256, self).__init__()
        if combines is None:
            combines = [1, 1, 1, 1]
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear((64*combines[0]+64*combines[1]+128*combines[2]+128*combines[3]), 256)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")

    def __call__(self, feature):
        feature = self.fc1(self.relu(feature))
        desp = feature / torch.norm(feature, dim=2, keepdim=True)

        return desp


class SuperPointExtractor128(nn.Module):

    def __init__(self, combines=None):
        super(SuperPointExtractor128, self).__init__()
        if combines is None:
            combines = [1, 1, 1, 1]
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear((64*combines[0]+64*combines[1]+128*combines[2]+128*combines[3]), 128)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")

    def __call__(self, feature):
        feature = self.fc1(self.relu(feature))
        desp = feature / torch.norm(feature, dim=2, keepdim=True)

        return desp


class SegExtractor384(nn.Module):

    def __init__(self):
        super(SegExtractor384, self).__init__()
        self.relu = nn.ReLU()
        self.desp = nn.Linear(384, 128)
        self.seg_logit = nn.Linear(384, 182)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")

    def __call__(self, feature):
        desp = self.desp(self.relu(feature))
        desp = desp / torch.norm(desp, dim=2, keepdim=True)

        if self.training:
            seg_logit = self.seg_logit(self.relu(feature))
            return desp, seg_logit
        else:
            return desp


class Extractor384(nn.Module):

    def __init__(self):
        super(Extractor384, self).__init__()
        self.relu = nn.ReLU()
        self.desp = nn.Linear(384, 128)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")

    def __call__(self, feature):
        desp = self.desp(self.relu(feature))
        desp = desp / torch.norm(desp, dim=2, keepdim=True)

        return desp


class Extractor640(nn.Module):

    def __init__(self):
        super(Extractor640, self).__init__()
        self.relu = nn.ReLU()
        self.fc = nn.Linear(640, 128)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")

    def __call__(self, feature):
        feature = self.fc(self.relu(feature))
        desp = feature / torch.norm(feature, dim=2, keepdim=True)

        return desp


class SuperPointNet(nn.Module):

    def __init__(self):
        super(SuperPointNet, self).__init__()
        self.relu = torch.nn.ReLU(inplace=True)
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        c1, c2, c3, c4, c5, d1 = 64, 64, 128, 128, 256, 256
        # Shared Encoder.
        self.conv1a = nn.Conv2d(3, c1, kernel_size=3, stride=1, padding=1)
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

        self.softmax = nn.Softmax(dim=1)
        self.tanh = nn.Tanh()

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

        return logit, desc, prob


class SuperPointDetector(nn.Module):

    def __init__(self):
        super(SuperPointDetector, self).__init__()
        self.relu = torch.nn.ReLU(inplace=True)
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        # Shared Encoder.
        self.conv1a = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.conv1b = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.conv2a = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.conv2b = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.conv3a = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv3b = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.conv4a = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.conv4b = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)

        # Detector Head.
        self.convPa = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.convPb = nn.Conv2d(256, 65, kernel_size=1, stride=1, padding=0)

        self.softmax = nn.Softmax(dim=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        x = self.relu(self.conv1a(x))
        c1 = self.relu(self.conv1b(x))  # 64

        c2 = self.pool(c1)
        c2 = self.relu(self.conv2a(c2))
        c2 = self.relu(self.conv2b(c2))  # 64

        c3 = self.pool(c2)
        c3 = self.relu(self.conv3a(c3))
        c3 = self.relu(self.conv3b(c3))  # 128

        c4 = self.pool(c3)
        c4 = self.relu(self.conv4a(c4))
        c4 = self.relu(self.conv4b(c4))  # 128

        # detect head
        cPa = self.relu(self.convPa(c4))
        logit = self.convPb(cPa)
        prob = self.softmax(logit)[:, :-1, :, :]

        return logit, prob, c1, c2, c3, c4


class SuperPointDescriptor(nn.Module):

    def __init__(self):
        super(SuperPointDescriptor, self).__init__()
        self.relu = torch.nn.ReLU(inplace=True)
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        # Shared Encoder.
        self.conv1a = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.conv1b = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.conv2a = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.conv2b = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.conv3a = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv3b = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.conv4a = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.conv4b = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)

        self.softmax = nn.Softmax(dim=1)

        # Descriptor Head.
        self.convDa = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.convDb = nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0)

        self.heatmap = nn.Conv2d((64 + 16 + 8 + 2), 1, kernel_size=3, stride=1, padding=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        x = self.relu(self.conv1a(x))
        c1 = self.relu(self.conv1b(x))  # 64

        c2 = self.pool(c1)
        c2 = self.relu(self.conv2a(c2))
        c2 = self.relu(self.conv2b(c2))  # 64

        c3 = self.pool(c2)
        c3 = self.relu(self.conv3a(c3))
        c3 = self.relu(self.conv3b(c3))  # 128

        c4 = self.pool(c3)
        c4 = self.relu(self.conv4a(c4))
        c4 = self.relu(self.conv4b(c4))  # 128

        heatmap1 = c1  # [h,w,64]
        heatmap2 = f.pixel_shuffle(c2, 2)  # [h,w,16]
        heatmap3 = f.pixel_shuffle(c3, 4)  # [h,w,8]
        heatmap4 = f.pixel_shuffle(c4, 8)  # [h,w,2]
        heatmap = self.heatmap(torch.cat((heatmap1, heatmap2, heatmap3, heatmap4), dim=1))

        # descriptor head
        cDa = self.relu(self.convDa(c4))
        feature = self.convDb(cDa)

        dn = torch.norm(feature, p=2, dim=1, keepdim=True)
        desc = feature.div(dn)

        return heatmap, desc


class SuperPointFSMSegmentation(nn.Module):

    def __init__(self, *args, **kwargs):
        super(SuperPointFSMSegmentation, self).__init__()
        self.relu = torch.nn.ReLU(inplace=True)
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        # Shared Encoder.
        self.conv1a = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.conv1b = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.conv2a = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.conv2b = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.conv3a = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv3b = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.conv4a = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.conv4b = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)

        self.softmax = nn.Softmax(dim=1)

        # segmentation head
        self.segment1 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.segment2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.segment3 = nn.Conv2d(256, 182, kernel_size=3, stride=1, padding=1)

        # Descriptor Head.
        self.convDa = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.convDb = nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0)

        # detector head
        self.heatmap = nn.Conv2d((64 + 16 + 8 + 2), 1, kernel_size=3, stride=1, padding=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        x = self.relu(self.conv1a(x))
        c1 = self.relu(self.conv1b(x))  # 64

        c2 = self.pool(c1)
        c2 = self.relu(self.conv2a(c2))
        c2 = self.relu(self.conv2b(c2))  # 64

        c3 = self.pool(c2)
        c3 = self.relu(self.conv3a(c3))
        c3 = self.relu(self.conv3b(c3))  # 128

        c4 = self.pool(c3)
        c4 = self.relu(self.conv4a(c4))
        c4 = self.relu(self.conv4b(c4))  # 128

        # detector head
        heatmap1 = c1  # [h,w,64]
        heatmap2 = f.pixel_shuffle(c2, 2)  # [h,w,16]
        heatmap3 = f.pixel_shuffle(c3, 4)  # [h,w,8]
        heatmap4 = f.pixel_shuffle(c4, 8)  # [h,w,2]
        heatmap = self.heatmap(torch.cat((heatmap1, heatmap2, heatmap3, heatmap4), dim=1))

        # segmentation head
        s = self.relu(self.segment1(c4))
        s = self.relu(self.segment2(s))
        logit = self.segment2(s)

        # descriptor head
        cDa = self.relu(self.convDa(c4))
        feature = self.convDb(cDa+s)

        dn = torch.norm(feature, p=2, dim=1, keepdim=True)
        desc = feature.div(dn)

        return heatmap, desc, logit


class SuperPointSegmentation(nn.Module):

    def __init__(self, *args, **kwargs):
        super(SuperPointSegmentation, self).__init__()
        self.relu = torch.nn.ReLU(inplace=True)
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        # Shared Encoder.
        self.conv1a = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.conv1b = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.conv2a = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.conv2b = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.conv3a = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv3b = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.conv4a = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.conv4b = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)

        self.softmax = nn.Softmax(dim=1)

        # segmentation head
        self.segment1 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.segment2 = nn.Conv2d(256, 182, kernel_size=3, stride=1, padding=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        x = self.relu(self.conv1a(x))
        c1 = self.relu(self.conv1b(x))  # 64

        c2 = self.pool(c1)
        c2 = self.relu(self.conv2a(c2))
        c2 = self.relu(self.conv2b(c2))  # 64

        c3 = self.pool(c2)
        c3 = self.relu(self.conv3a(c3))
        c3 = self.relu(self.conv3b(c3))  # 128

        c4 = self.pool(c3)
        c4 = self.relu(self.conv4a(c4))
        c4 = self.relu(self.conv4b(c4))  # 128

        # segmentation head
        s = self.relu(self.segment1(c4))
        logit = self.segment2(s)

        if self.training:
            return [logit]
        else:
            return logit



if __name__ == "__main__":
    random_input = torch.randn((1, 1, 240, 320))
    model = SuperPointNetFloat()
    result = model(random_input)








