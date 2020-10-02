#
# Created by ZhangYuyang on 2020/9/1
#
import torch
import torch.nn as nn

# from torchvision.models import VGG


def conv3x3_relu(inplanes, planes, rate=1):
    conv3x3_relu = nn.Sequential(nn.Conv2d(inplanes, planes, kernel_size=3,
                                    stride=1, padding=rate, dilation=rate),
                                 nn.ReLU())
    return conv3x3_relu


class _VggLayer(nn.Sequential):

    def __init__(self, n_layers, in_ch, out_ch, stride, dilation):
        super(_VggLayer, self).__init__()

        for i in range(n_layers):
            self.add_module(
                'block{}'.format(i+1),
                conv3x3_relu(
                    inplanes=in_ch if i == 0 else out_ch,
                    planes=out_ch,
                    rate=dilation,
                )
            )
        self.add_module(
            'maxpooling',
            nn.MaxPool2d(3, stride=stride, padding=1)
        )


class _ASPP(nn.Module):
    """
    Atrous spatial pyramid pooling (ASPP)
    """

    def __init__(self, in_ch, out_ch, rates):
        super(_ASPP, self).__init__()
        for i, rate in enumerate(rates):
            self.add_module(
                "c{}".format(i),
                nn.Conv2d(in_ch, out_ch, 3, 1, padding=rate, dilation=rate, bias=True),
            )

        for m in self.children():
            nn.init.normal_(m.weight, mean=0, std=0.01)
            nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return sum([stage(x) for stage in self.children()])


class DeeplabVgg16(nn.Module):

    def __init__(self, **config):
        super(DeeplabVgg16, self).__init__()
        self.block1 = nn.Sequential(
            conv3x3_relu(3, 64),
            conv3x3_relu(64, 64),
            nn.MaxPool2d(2, stride=2))
        self.block2 = nn.Sequential(
            conv3x3_relu(64, 128),
            conv3x3_relu(128, 128),
            nn.MaxPool2d(2, stride=2))
        self.block3 = nn.Sequential(
            conv3x3_relu(128, 256),
            conv3x3_relu(256, 256),
            conv3x3_relu(256, 256),
            nn.MaxPool2d(2, stride=2))
        self.block4 = nn.Sequential(
            conv3x3_relu(256, 512, rate=2),
            conv3x3_relu(512, 512, rate=2),
            conv3x3_relu(512, 512, rate=2))
        self.block5 = nn.Sequential(
            conv3x3_relu(512, 512, rate=4),
            conv3x3_relu(512, 512, rate=4),
            conv3x3_relu(512, 512, rate=4))
        # self.aspp = _ASPP(512, config['n_classes'], rates=[6, 12, 18, 24])
        self.logit = nn.Conv2d(512, config['n_classes'], 3, stride=1, padding=1)

        self._initialize_weights()

    def forward(self, x):
        f1 = self.block1(x)
        f2 = self.block2(f1)
        f3 = self.block3(f2)
        f4 = self.block4(f3)
        f5 = self.block5(f4)

        logit = self.logit(f5)

        if self.training:
            return [logit]
        else:
            return logit

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


class _ConvBnReLU(nn.Sequential):
    """
    Cascade of 2D convolution, batch norm, and ReLU.
    """

    def __init__(
        self, in_ch, out_ch, kernel_size, stride, padding, dilation, relu=True
    ):
        super(_ConvBnReLU, self).__init__()
        self.add_module(
            "conv",
            nn.Conv2d(
                in_ch, out_ch, kernel_size, stride, padding, dilation, bias=False
            ),
        )
        self.add_module("bn", nn.BatchNorm2d(out_ch, eps=1e-5, momentum=1 - 0.999))

        if relu:
            self.add_module("relu", nn.ReLU())


class _Bottleneck(nn.Module):
    """
    Bottleneck block of MSRA ResNet.
    """

    def __init__(self, in_ch, out_ch, stride, dilation, downsample):
        super(_Bottleneck, self).__init__()
        mid_ch = out_ch // 4
        self.reduce = _ConvBnReLU(in_ch, mid_ch, 1, stride, 0, 1, True)
        self.conv3x3 = _ConvBnReLU(mid_ch, mid_ch, 3, 1, dilation, dilation, True)
        self.increase = _ConvBnReLU(mid_ch, out_ch, 1, 1, 0, 1, False)
        self.shortcut = (
            _ConvBnReLU(in_ch, out_ch, 1, stride, 0, 1, False)
            if downsample
            else nn.Identity()
        )

    def forward(self, x):
        h = self.reduce(x)
        h = self.conv3x3(h)
        h = self.increase(h)
        h += self.shortcut(x)
        return torch.relu(h)


class _ResLayer(nn.Sequential):
    """
    Residual layer with multi grids
    """

    def __init__(self, n_layers, in_ch, out_ch, stride, dilation, multi_grids=None):
        super(_ResLayer, self).__init__()

        if multi_grids is None:
            multi_grids = [1 for _ in range(n_layers)]
        else:
            assert n_layers == len(multi_grids)

        # Downsampling is only in the first block
        for i in range(n_layers):
            self.add_module(
                "block{}".format(i + 1),
                _Bottleneck(
                    in_ch=(in_ch if i == 0 else out_ch),
                    out_ch=out_ch,
                    stride=(stride if i == 0 else 1),
                    dilation=dilation * multi_grids[i],
                    downsample=(True if i == 0 else False),
                ),
            )


class _Stem(nn.Sequential):
    """
    The 1st conv layer.
    Note that the max pooling is different from both MSRA and FAIR ResNet.
    """

    def __init__(self, out_ch):
        super(_Stem, self).__init__()
        self.add_module("conv1", _ConvBnReLU(3, out_ch, 7, 2, 3, 1))
        self.add_module("pool", nn.MaxPool2d(3, 2, 1, ceil_mode=True))


class DeeplabV2Base(nn.Sequential):
    """
    DeepLab v2: Dilated ResNet + ASPP
    Output stride is fixed at 8
    """

    def __init__(self, **config):
        super(DeeplabV2Base, self).__init__()
        ch = [64 * 2 ** p for p in range(6)]
        self.add_module("layer1", _Stem(ch[0]))
        self.add_module("layer2", _ResLayer(config['n_blocks'][0], ch[0], ch[2], 1, 1))
        self.add_module("layer3", _ResLayer(config['n_blocks'][1], ch[2], ch[3], 2, 1))
        self.add_module("layer4", _ResLayer(config['n_blocks'][2], ch[3], ch[4], 1, 2))
        self.add_module("layer5", _ResLayer(config['n_blocks'][3], ch[4], ch[5], 1, 4))

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, _ConvBnReLU.BATCH_NORM):
                m.eval()


class DeeplabV2(nn.Module):
    """
    DeepLab v2: Dilated ResNet + ASPP
    Output stride is fixed at 8
    """

    def __init__(self, **config):
        super(DeeplabV2, self).__init__()
        self.base = DeeplabV2Base(**config)

    def forward(self, x):
        feature = self.base(x)

        return feature

    def freeze(self):
        for p in self.parameters():
            p.requires_grad = False


if __name__ == "__main__":
    config = {'n_classes': 21}
    model = DeeplabVgg16(**config)
    model.eval()
    image = torch.randn(1, 3, 513, 513)

    print(model)
    print("input:", image.shape)
    print("output:", model(image).shape)





