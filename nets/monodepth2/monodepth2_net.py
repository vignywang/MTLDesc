#
# Created by ZhangYuyang on 2020/9/17
#
from torch.nn.modules import Module

from .resnet_encoder import ResnetEncoder
from .depth_decoder import DepthDecoder


class Monodepth2Net(Module):

    def __init__(self):
        super(Monodepth2Net, self).__init__()
        self.encoder = ResnetEncoder(18, pretrained=True)
        self.decoder = DepthDecoder(self.encoder.num_ch_enc)

    def forward(self, x):
        features = self.encoder(x)
        outputs = self.decoder(features)

        return outputs[('disp', 0)]





