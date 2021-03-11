"""
DeeplabV3
    Input: (3, 224, 224)

    Total params: 39,057,557
    Trainable params: 39,057,557
    Non-trainable params: 0

    Input size (MB): 0.57
    Forward/backward pass size (MB): 502.58
    Params size (MB): 148.99
    Estimated Total Size (MB): 652.15

    MACs/FLOPs: 22,029,738,004
"""

import torch.nn.functional as F
from dxtorchutils.utils.layers import *


class DeeplabV3p(Module):
    def __init__(self, num_classes=21, resnet_based_type=101):
        super(DeeplabV3p, self).__init__()
        if resnet_based_type == 18:
            self.conv1 = _building_block_layer(2, 2, 2, 2)
        elif resnet_based_type == 34:
            self.conv1 = _building_block_layer(3, 4, 6, 3)
        elif resnet_based_type == 50:
            self.conv1 = _bottle_neck_layer(3, 4, 6, 3)
        elif resnet_based_type == 101:
            self.conv1 = _bottle_neck_layer(3, 4, 23, 3)
        elif resnet_based_type == 152:
            self.conv1 = _bottle_neck_layer(3, 8, 36, 3)
        else:
            exit("Wrong ResNet Type")

        self.conv0 = Conv2d(3, 64, 7, 2, 4)
        self.pool0 = MaxPool2d(3, 2)
        self.pool1 = MaxPool2d(2)

        self.num_classes = num_classes
        if resnet_based_type == (18 or 34):
            self.aspp0 = _aspp(512, 128, 1)
            self.aspp1 = _aspp(512, 128, 6)
            self.aspp2 = _aspp(512, 128, 12)
            self.aspp3 = _aspp(512, 128, 18)

            self.global_pool = Sequential(
                OrderedDict([
                    ("pool", AvgPool2d(2)),
                    ("conv", Conv2d(512, 128, 1)),
                    ("bn", BatchNorm2d(128))
                ])
            )
            self.conv2 = Sequential(
                OrderedDict([
                    ("conv", Conv2d(640, 128, 1)),
                    ("bn", BatchNorm2d(128))
                ])
            )
            self.conv3 = Conv2d(128, num_classes, 1)

        else:
            self.aspp0 = _aspp(2048, 256, 1)
            self.aspp1 = _aspp(2048, 256, 6)
            self.aspp2 = _aspp(2048, 256, 12)
            self.aspp3 = _aspp(2048, 256, 18)

            self.global_pool = Sequential(
                OrderedDict([
                    ("pool", AvgPool2d(2)),
                    ("conv", Conv2d(2048, 256, 1)),
                    ("bn", BatchNorm2d(256))
                ])
            )
            self.conv2 = Sequential(
                OrderedDict([
                    ("conv", Conv2d(1280, 256, 1)),
                    ("bn", BatchNorm2d(256))
                ])
            )
            self.conv3 = Conv2d(256, num_classes, 1)

    def forward(self, input):
        x = self.conv0(input)
        x = torch.relu(x)
        x = self.pool0(x)
        x = self.conv1(x)
        x = self.pool1(x)
        x0 = self.aspp0(x)
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.global_pool(x)
        x4 = F.relu(x4, True)
        h0, w0 = x0.shape[-2:]
        x4 = F.interpolate(x4, (h0, w0), None, "bilinear", True)

        x = torch.cat((x0, x1, x2, x3, x4), 1)
        x = self.conv2(x)
        x = torch.relu(x)
        x = self.conv3(x)
        x = torch.relu(x)

        h, w = input.shape[-2:]
        output = F.interpolate(x, (h, w), None, "bilinear", True)

        return output


def _building_block_layer(num1, num2, num3, num4):
    return Sequential(
        OrderedDict([
            ("Res1", _building_block(64, 64, num1)),
            ("Res2", _building_block(64, 128, num2)),
            ("Res3", _building_block_atrous(128, 256, num3, 2)),
            ("Res4", _building_block_atrous(256, 512, num4, 4))
        ])
    )


def _bottle_neck_layer(num1, num2, num3, num4):
    return Sequential(
        OrderedDict([
            ("Res1", _bottle_neck(64, 256, num1)),
            ("Res2", _bottle_neck(256, 512, num2)),
            ("Res3", _bottle_neck_atrous(512, 1024, num3, 2)),
            ("Res4", _bottle_neck_atrous(1024, 2048, num4, 4))
        ])
    )


def _building_block(in_channel, out_channel, layer_num):
    seq = Sequential()
    # 除了res1，第一次要做个步长为2的卷积相当于pool一下
    if in_channel != out_channel:
        seq.add_module("BuildingBlock1", _BuildingBlock(in_channel, out_channel, 2, True))
    else:
        seq.add_module("BuildingBlock1", _BuildingBlock(in_channel, out_channel, 1, False))

    for i in range(layer_num - 1):
        seq.add_module("BuildingBlock{}".format(i + 2), _BuildingBlock(out_channel, out_channel))

    return seq


def _building_block_atrous(in_channel, out_channel, dilation, layer_num):
    seq = Sequential()
    seq.add_module("BuildingBlock1", _BuildingBlockAtrous(in_channel, out_channel, dilation, True))

    for i in range(layer_num - 1):
        seq.add_module("BuildingBlock{}".format(i + 2), _BuildingBlockAtrous(out_channel, out_channel, dilation))

    return seq


def _bottle_neck(in_channel, out_channel, layer_num):
    seq = Sequential()
    # 除了res1，第一次要做个步长为2的卷积相当于pool一下
    if in_channel == out_channel // 2:
        seq.add_module("BottleNeck1", _BottleNeck(in_channel, out_channel, 2, True))
    else:
        seq.add_module("BottleNeck1", _BottleNeck(in_channel, out_channel, 1, True))

    for i in range(layer_num - 1):
        seq.add_module("BottleNeck{}".format(i + 2), _BottleNeck(out_channel, out_channel))

    return seq


def _bottle_neck_atrous(in_channel, out_channel, dilation, layer_num):
    seq = Sequential()
    seq.add_module("BottleNeck1", _BottleNeckAtrous(in_channel, out_channel, dilation, True))

    for i in range(layer_num - 1):
        seq.add_module("BottleNeck{}".format(i + 2), _BottleNeckAtrous(out_channel, out_channel, dilation))

    return seq


class _BuildingBlock(Module):
    def __init__(self, in_channel, out_channel, stride=1, first_in=False):
        super(_BuildingBlock, self).__init__()
        self.conv0 = Conv2d(in_channel, out_channel, 3, stride, 1)
        self.conv1 = Conv2d(out_channel, out_channel, 3, 1, 1)
        self.bn0 = BatchNorm2d(out_channel)
        self.bn1 = BatchNorm2d(out_channel)

        # 相加时第一次通道数不同，需要增加通道
        self.conv_extend = Conv2d(in_channel, out_channel, 1, stride)
        self.first_in = first_in

    def forward(self, input):
        x = self.conv0(input)
        x = self.bn0(x)
        x = torch.relu(x)
        x = self.conv1(x)
        x = self.bn1(x)

        if self.first_in:
            input = self.conv_extend(input)

        output = torch.relu(x + input)

        return output


class _BuildingBlockAtrous(Module):
    def __init__(self, in_channel, out_channel, dilation, first_in=False):
        super(_BuildingBlockAtrous, self).__init__()
        self.conv0 = Conv2d(in_channel, out_channel, 3, 1, dilation, dilation)
        self.conv1 = Conv2d(out_channel, out_channel, 3, 1, dilation, dilation)
        self.bn0 = BatchNorm2d(out_channel)
        self.bn1 = BatchNorm2d(out_channel)

        # 相加时第一次通道数不同，需要增加通道
        self.conv_extend = Conv2d(in_channel, out_channel, 1, 1)
        self.first_in = first_in

    def forward(self, input):
        x = self.conv0(input)
        x = self.bn0(x)
        x = torch.relu(x)
        x = self.conv1(x)
        x = self.bn1(x)

        if self.first_in:
            input = self.conv_extend(input)

        output = torch.relu(x + input)

        return output


class _BottleNeck(Module):
    def __init__(self, in_channel, out_channel, stride=1, first_in=False):
        super(_BottleNeck, self).__init__()
        # 所有中间层都是输出层通道数的 1/4
        mid_channel = out_channel // 4

        self.conv0 = Conv2d(in_channel, mid_channel, 1, stride)
        self.conv1 = Conv2d(mid_channel, mid_channel, 3, 1, 1)
        self.conv2 = Conv2d(mid_channel, out_channel, 1)
        self.bn0 = BatchNorm2d(mid_channel)
        self.bn1 = BatchNorm2d(out_channel)

        # 相加时第一次通道数不同，需要增加通道
        self.conv_extend = Conv2d(in_channel, out_channel, 1, stride)
        self.first_in = first_in

    def forward(self, input):
        x = self.conv0(input)
        x = self.bn0(x)
        x = torch.relu(x)
        x = self.conv1(x)
        x = self.bn0(x)
        x = torch.relu(x)
        x = self.conv2(x)
        x = self.bn1(x)

        if self.first_in:
            input = self.conv_extend(input)

        output = torch.relu(x + input)

        return output


class _BottleNeckAtrous(Module):
    def __init__(self, in_channel, out_channel, dilation, first_in=False):
        super(_BottleNeckAtrous, self).__init__()
        # 所有中间层都是输出层通道数的 1/4
        mid_channel = out_channel // 4

        self.conv0 = conv_relu_bn(in_channel, mid_channel, 1, 1)
        self.conv1 = conv_relu_bn(mid_channel, mid_channel, 3, 1, dilation, dilation)
        self.conv2 = conv_relu_bn(mid_channel, out_channel, 1)

        # 相加时第一次通道数不同，需要增加通道
        self.conv_extend = Conv2d(in_channel, out_channel, 1, 1)
        self.first_in = first_in

    def forward(self, input):
        x = self.conv0(input)
        x = self.conv1(x)
        x = self.conv2(x)

        if self.first_in:
            input = self.conv_extend(input)

        output = torch.relu(x + input)

        return output


def _aspp(in_channel, out_channel, dilation):
    if dilation == 1:
        padding = 0
        kernel_size = 1
    else:
        padding = dilation
        kernel_size = 3

    return conv_relu(in_channel, out_channel, kernel_size, 1, padding, dilation)


if __name__ == '__main__':
    # calculate parameters
    from thop import profile
    from torchsummary import summary

    model = DeeplabV3()
    input = torch.randn((1, 3, 224, 224))
    macs, params = profile(model, inputs=(input,))
    summary(model, (3, 224, 224))

    print("MACs: {}".format(macs))
