"""
UNet++
    Input: (3, 224, 224)

    Total params: 33,163,848
    Trainable params: 33,163,848
    Non-trainable params: 0

    Input size (MB): 0.57
    Forward/backward pass size (MB): 2459.95
    Params size (MB): 126.51
    Estimated Total Size (MB): 2587.04

    MACs/FLOPs: 83,942,541,312
"""
import torch.nn.functional as F
from dxtorchutils.utils.layers import *


class UNet2p(Module):
    def __init__(self, num_classes=2, in_features=3, edge_opt=False):
        super(UNet2p, self).__init__()
        self.en_block0 = _ConvBlock(in_features, 64, edge_opt)
        self.en_block1 = _ConvBlock(64, 128, edge_opt)
        self.en_block2 = _ConvBlock(128, 256, edge_opt)
        self.en_block3 = _ConvBlock(256, 512, edge_opt)
        self.en_block4 = _ConvBlock(512, 1024, edge_opt)

        self.pool0 = MaxPool2d(2)
        self.pool1 = MaxPool2d(2)
        self.pool2 = MaxPool2d(2)
        self.pool3 = MaxPool2d(2)

        self.up_sample01 = _UpSampleBlockBi(128, edge_opt)
        self.up_sample02 = _UpSampleBlockTri(128, edge_opt)
        self.up_sample03 = _UpSampleBlockTri(128, edge_opt)
        self.up_sample04 = _UpSampleBlockTri(128, edge_opt)
        self.de_block01 = _ConvBlock(128, 64, edge_opt)
        self.de_block02 = _ConvBlock(192, 64, edge_opt)
        self.de_block03 = _ConvBlock(192, 64, edge_opt)
        self.de_block04 = _ConvBlock(192, 64, edge_opt)

        self.up_sample11 = _UpSampleBlockBi(256, edge_opt)
        self.up_sample12 = _UpSampleBlockTri(256, edge_opt)
        self.up_sample13 = _UpSampleBlockTri(256, edge_opt)
        self.de_block11 = _ConvBlock(256, 128, edge_opt)
        self.de_block12 = _ConvBlock(384, 128, edge_opt)
        self.de_block13 = _ConvBlock(384, 128, edge_opt)


        self.up_sample21 = _UpSampleBlockBi(512, edge_opt)
        self.up_sample22 = _UpSampleBlockTri(512, edge_opt)
        self.de_block21 = _ConvBlock(512, 256, edge_opt)
        self.de_block22 = _ConvBlock(768, 256, edge_opt)


        self.up_sample31 = _UpSampleBlockBi(1024, edge_opt)
        self.de_block31 = _ConvBlock(1024, 512, edge_opt)

        self.final_layer1 = Conv2d(64, num_classes, 1)
        self.final_layer2 = Conv2d(64, num_classes, 1)
        self.final_layer3 = Conv2d(64, num_classes, 1)
        self.final_layer4 = Conv2d(64, num_classes, 1)

        self.remaining = 0

    def cut_off(self, remaining):
        self.remaining = remaining

    def forward(self, input):
        if self.training or self.remaining == 0:
            x00 = self.en_block0(input)

            x10 = self.en_block1(self.pool0(x00))

            x20 = self.en_block2(self.pool1(x10))

            x30 = self.en_block3(self.pool2(x20))

            x40 = self.en_block4(self.pool3(x30))

            x01 = self.de_block01(self.up_sample01(x10, x00))

            x11 = self.de_block11(self.up_sample11(x20, x10))

            x21 = self.de_block21(self.up_sample21(x30, x20))

            x31 = self.de_block31(self.up_sample31(x40, x30))

            x02 = self.de_block02(self.up_sample02(x11, x00, x01))

            x12 = self.de_block12(self.up_sample12(x21, x10, x11))

            x22 = self.de_block22(self.up_sample22(x31, x20, x21))

            x03 = self.de_block03(self.up_sample03(x12, x00, x02))

            x13 = self.de_block13(self.up_sample13(x22, x10, x12))

            x04 = self.de_block04(self.up_sample04(x13, x00, x03))


            if self.training:
                output1 = self.final_layer1(x01)
                output2 = self.final_layer2(x02)
                output3 = self.final_layer3(x03)
                output4 = self.final_layer4(x04)

                return [output1, output2, output3, output4]

            output = self.final_layer4(x04)

            return output

        if self.remaining == 1:
            x00 = self.en_block0(input)

            x10 = self.en_block1(self.pool0(x00))

            x01 = self.de_block01(self.up_sample01(x10, x00))

            output = self.final_layer1(x01)

            return output

        if self.remaining == 2:
            x00 = self.en_block0(input)

            x10 = self.en_block1(self.pool0(x00))

            x20 = self.en_block2(self.pool1(x10))

            x01 = self.de_block01(self.up_sample01(x10, x00))

            x11 = self.de_block11(self.up_sample11(x20, x10))

            x02 = self.de_block02(self.up_sample02(x11, x00, x01))

            output = self.final_layer2(x02)

            return output

        if self.remaining == 3:
            x00 = self.en_block0(input)

            x10 = self.en_block1(self.pool0(x00))

            x20 = self.en_block2(self.pool1(x10))

            x30 = self.en_block3(self.pool2(x20))

            x01 = self.de_block01(self.up_sample01(x10, x00))

            x11 = self.de_block11(self.up_sample11(x20, x10))

            x21 = self.de_block21(self.up_sample21(x30, x20))

            x02 = self.de_block02(self.up_sample02(x11, x00, x01))

            x12 = self.de_block12(self.up_sample12(x21, x10, x11))

            x03 = self.de_block03(self.up_sample03(x12, x00, x02))

            output = self.final_layer3(x03)

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


class _UpSampleBlockBi(Module):
    def __init__(self, in_channels, edge_opt=True):
        super(_UpSampleBlockBi, self).__init__()
        self.up_sample = Upsample(None, 2)
        self.conv = Conv2d(in_channels, in_channels // 2, 1)

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



class _UpSampleBlockTri(Module):
    def __init__(self, in_channels, edge_opt=True):
        super(_UpSampleBlockTri, self).__init__()
        self.up_sample = Upsample(None, 2)
        self.conv = conv_relu_bn(in_channels, in_channels // 2, 1)

        self.edge_opt = edge_opt

    def forward(self, x_de, x_en, x_pre):
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
            x_pre = x_pre[:, :, 2: h_edge + 2, 2: w_edge + 2]


        output = torch.cat((x_en, x_de, x_pre), 1)

        return output


if __name__ == '__main__':
    # calculate parameters
    from thop import profile
    from torchsummary import summary

    model = UNet2p()
    input = torch.randn((1, 3, 224, 224))
    macs, params = profile(model, inputs=(input, ))
    summary(model, (3, 224, 224))

    print("MACs: {}".format(macs))
