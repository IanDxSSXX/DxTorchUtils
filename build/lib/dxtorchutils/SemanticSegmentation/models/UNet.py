"""
UNet
    Input: (3, 224, 224)

    Total params: 28,956,546
    Trainable params: 28,956,546
    Non-trainable params: 0

    Input size (MB): 0.57
    Forward/backward pass size (MB): 987.66
    Params size (MB): 110.46
    Estimated Total Size (MB): 1098.69

    MACs/FLOPs: 36,972,988,416
"""
import torch.nn.functional as F
from dxtorchutils.utils.layers import *


class UNet(Module):
    def __init__(self, num_classes=2, in_features=3, edge_opt=False):
        super(UNet, self).__init__()

        self.en_block0 = _ConvBlock(in_features, 64, edge_opt)
        self.en_block1 = _ConvBlock(64, 128, edge_opt)
        self.en_block2 = _ConvBlock(128, 256, edge_opt)
        self.en_block3 = _ConvBlock(256, 512, edge_opt)
        self.en_block4 = _ConvBlock(512, 1024, edge_opt)

        self.pool0 = MaxPool2d(2)
        self.pool1 = MaxPool2d(2)
        self.pool2 = MaxPool2d(2)
        self.pool3 = MaxPool2d(2)


        self.up_sample0 = _UpSampleBlock(1024, edge_opt)
        self.up_sample1 = _UpSampleBlock(512, edge_opt)
        self.up_sample2 = _UpSampleBlock(256, edge_opt)
        self.up_sample3 = _UpSampleBlock(128, edge_opt)

        self.de_block0 = _ConvBlock(1024, 512, edge_opt)
        self.de_block1 = _ConvBlock(512, 256, edge_opt)
        self.de_block2 = _ConvBlock(256, 128, edge_opt)
        self.de_block3 = _ConvBlock(128, 64, edge_opt)

        self.final_layer = Conv2d(64, num_classes, 1)

    def forward(self, input):
        x0 = self.en_block0(input)

        x1 = self.pool0(x0)

        x1 = self.en_block1(x1)

        x2 = self.pool1(x1)

        x2 = self.en_block2(x2)

        x3 = self.pool2(x2)

        x3 = self.en_block3(x3)

        x = self.pool3(x3)

        x = self.en_block4(x)

        x = self.up_sample0(x, x3)

        x = self.de_block0(x)

        x = self.up_sample1(x, x2)

        x = self.de_block1(x)

        x = self.up_sample2(x, x1)

        x = self.de_block2(x)

        x = self.up_sample3(x, x0)

        x = self.de_block3(x)

        output = self.final_layer(x)

        return output


class _ConvBlock(Module):
    def __init__(self, in_channels, out_channels, edge_opt):
        super(_ConvBlock, self).__init__()
        padding = 0 if edge_opt else 1
        self.conv1 = conv_relu_bn(in_channels, out_channels, 3, 1, padding)
        self.conv2 = conv_relu_bn(out_channels, out_channels, 3, 1, padding)


    def forward(self, input):
        x = self.conv1(input)
        output = self.conv2(x)

        return output


class _UpSampleBlock(Module):
    def __init__(self, in_channels, edge_opt):
        super(_UpSampleBlock, self).__init__()
        self.up_sample = Upsample(None, 2)
        self.conv = conv_relu_bn(in_channels, in_channels // 2, 1)

        self.edge_opt = edge_opt

    def forward(self, x_de, x_en):
        h, w = x_en.shape[-2:]
        # 缩小边缘缩小到原来的两倍减2
        h_edge = x_de.shape[-2] * 2
        w_edge = x_de.shape[-1] * 2

        x_de = self.up_sample(x_de)
        x_de = self.conv(x_de)

        x_de = F.interpolate(x_de, (h, w), None, "bilinear", True)

        # Crop
        if self.edge_opt:
            x_en = x_en[:, :, 2: h_edge + 2, 2: w_edge + 2]
            x_de = x_de[:, :, 2: h_edge + 2, 2: w_edge + 2]

        output = torch.cat((x_en, x_de), 1)

        return output


if __name__ == '__main__':
    # calculate parameters
    from thop import profile
    from torchsummary import summary

    model = UNet()
    input = torch.randn((1, 3, 224, 224))
    macs, params = profile(model, inputs=(input, ))
    summary(model, (3, 224, 224))

    print("MACs: {}".format(macs))
