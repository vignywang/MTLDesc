import torch
import torch.nn as nn
import torch.nn.functional as f
class ScaleBackbone(nn.Module):
    def __init__(self):
        super(ScaleBackbone, self).__init__()
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

        self.heatmap = nn.Conv2d((64 + 16 + 8 + 2), 1, kernel_size=3, stride=1, padding=1)
        self.scalemap = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1)
        self.active=f.softplus
        self.conv_avg = nn.Conv2d(128, 384, kernel_size=3, stride=1, padding=1)
        self.conv_des = nn.Conv2d(384, 128, kernel_size=1, stride=1, padding=0)
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

        #KeyPoint Map
        heatmap1 = c1  # [h,w,64]
        heatmap2 = f.pixel_shuffle(c2, 2)  # [h,w,16]
        heatmap3 = f.pixel_shuffle(c3, 4)  # [h,w,8]
        heatmap4 = f.pixel_shuffle(c4, 8)  # [h,w,2]
        heatmap = self.heatmap(torch.cat((heatmap1, heatmap2, heatmap3, heatmap4), dim=1))


        #Descriptor
        des_size = c2.shape[2:]  #1/2 HxW
        c1 = f.interpolate(c1, des_size, mode='bilinear')
        c2 = c2
        c3 = f.interpolate(c3, des_size, mode='bilinear')
        c4 = f.interpolate(c4, des_size, mode='bilinear')
        feature=torch.cat((c1, c2, c3, c4), dim=1)

        # Scale Map
        weightmap = torch.mean(feature, dim=1, keepdim=True)
        scalemap = self.scalemap(weightmap)
        scalemap = self.active(scalemap)

        #Global Context
        avg = f.avg_pool2d(c4, c4.size()[2:])
        avg =self.relu( self.conv_avg(avg))
        descriptor=feature+avg
        descriptor=self.conv_des(descriptor)


        return heatmap,descriptor, scalemap
