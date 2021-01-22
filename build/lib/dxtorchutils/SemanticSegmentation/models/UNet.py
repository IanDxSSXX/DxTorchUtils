""" UNet """
""" 不检测边缘而得到同原来图一样大的图 """
import torch
from torch.nn import *
import torch.nn.functional as F


class UNet(Module):
    def __init__(self, n_classed=2, in_features=1, edge_opt=False, is_bn=True):
        super(UNet, self).__init__()
        self.pool = MaxPool2d(2)
        self.en_block1 = _ConvBlock(in_features, 64, edge_opt, is_bn)
        self.en_block2 = _ConvBlock(64, 128, edge_opt, is_bn)
        self.en_block3 = _ConvBlock(128, 256, edge_opt, is_bn)
        self.en_block4 = _ConvBlock(256, 512, edge_opt, is_bn)
        self.en_block5 = _ConvBlock(512, 1024, edge_opt, is_bn)

        self.up_sample1 = _UpSampleBlock(1024, edge_opt, is_bn)
        self.up_sample2 = _UpSampleBlock(512, edge_opt, is_bn)
        self.up_sample3 = _UpSampleBlock(256, edge_opt, is_bn)
        self.up_sample4 = _UpSampleBlock(128, edge_opt, is_bn)

        self.de_block1 = _ConvBlock(1024, 512, edge_opt, is_bn)
        self.de_block2 = _ConvBlock(512, 256, edge_opt, is_bn)
        self.de_block3 = _ConvBlock(256, 128, edge_opt, is_bn)
        self.de_block4 = _ConvBlock(128, 64, edge_opt, is_bn)

        self.final_layer = Conv2d(64, n_classed, 1)

    def forward(self, x):
        # x(1, 572, 572)
        x0 = self.en_block1(x)
        # x0(64, 568, 568)
        x1 = self.pool(x0)
        # x1(64, 284, 284)
        x1 = self.en_block2(x1)
        # x1(128, 280, 280)
        x2 = self.pool(x1)
        # x2(128, 140, 140)
        x2 = self.en_block3(x2)
        # x2(256, 136, 136)
        x3 = self.pool(x2)
        # x3(256, 68, 68)
        x3 = self.en_block4(x3)
        # x3(512, 64, 64)
        x = self.pool(x3)
        # x(512, 32, 32)
        x = self.en_block5(x)
        # x(1024, 28, 28)
        x = self.up_sample1(x, x3)
        # x(1024, 56, 56)
        x = self.de_block1(x)
        # x(512, 52, 52)
        x = self.up_sample2(x, x2)
        # x(512, 104, 104)
        x = self.de_block2(x)
        # x(256, 100, 100)
        x = self.up_sample3(x, x1)
        # x(256, 200, 200)
        x = self.de_block3(x)
        # x(128, 196, 196)
        x = self.up_sample4(x, x0)
        # x(128, 392, 392)
        x = self.de_block4(x)
        # x(64, 388, 388)
        x = self.final_layer(x)
        # x(2, 388, 388)

        return x


class _ConvBlock(Module):
    def __init__(self, in_channels, out_channels, edge_opt, is_bn=True):
        super(_ConvBlock, self).__init__()
        padding = 0 if edge_opt else 1
        self.conv1 = Conv2d(in_channels, out_channels, 3, 1, padding)
        self.conv2 = Conv2d(out_channels, out_channels, 3, 1, padding)
        self.bn1 = BatchNorm2d(out_channels)
        self.bn2 = BatchNorm2d(out_channels)

        self.is_bn = is_bn

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x, True)
        if self.is_bn:
            x = self.bn1(x)
        x = self.conv2(x)
        x = F.relu(x, True)
        if self.is_bn:
            x = self.bn2(x)

        return x


class _UpSampleBlock(Module):
    def __init__(self, in_channels, edge_opt, is_bn=True):
        super(_UpSampleBlock, self).__init__()
        self.up_sample = Upsample(None, 2)
        self.conv = Conv2d(in_channels, in_channels // 2, 1)
        self.bn = BatchNorm2d(in_channels // 2,)
        self.is_bn = is_bn

        self.edge_opt = edge_opt

    def forward(self, x_de, x_en):
        h, w = x_en.shape[-2:]
        # 缩小边缘缩小到原来的两倍减2
        h_edge = x_de.shape[-2] * 2
        w_edge = x_de.shape[-1] * 2

        x_de = self.up_sample(x_de)
        x_de = self.conv(x_de)
        x_de = F.relu(x_de, True)
        if self.is_bn:
            x_de = self.bn(x_de)

        x_de = F.interpolate(x_de, (h, w), None, "bilinear", True)

        # Crop
        if self.edge_opt:
            x_en = x_en[:, :, 2: h_edge + 2, 2: w_edge + 2]
            x_de = x_de[:, :, 2: h_edge + 2, 2: w_edge + 2]

        x = torch.cat((x_en, x_de), 1)

        return x
