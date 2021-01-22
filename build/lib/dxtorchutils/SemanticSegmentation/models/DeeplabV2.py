""" DeeplabV2 """
from collections import OrderedDict
from torch.nn import *
import torch.nn.functional as F


class DeeplabV2(Module):
    def __init__(self, num_classes=21, resnet_based_type=101):
        super(DeeplabV2, self).__init__()
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
            self.aspp0 = _ASPP(512, 1024, 6, num_classes)
            self.aspp1 = _ASPP(512, 1024, 12, num_classes)
            self.aspp2 = _ASPP(512, 1024, 18, num_classes)
            self.aspp3 = _ASPP(512, 1024, 24, num_classes)
        else:
            self.aspp0 = _ASPP(2048, 4096, 6, num_classes)
            self.aspp1 = _ASPP(2048, 4096, 12, num_classes)
            self.aspp2 = _ASPP(2048, 4096, 18, num_classes)
            self.aspp3 = _ASPP(2048, 4096, 24, num_classes)


    def forward(self, x):
        h, w = x.shape[-2:]
        x = self.conv0(x)
        x = F.relu(x, True)
        x = self.pool0(x)
        x = self.conv1(x)
        x = self.pool1(x)
        x0 = self.aspp0(x)
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)

        x = x0 + x1 + x2 + x3
        x = F.relu(x, True)
        x = F.interpolate(x, (h, w), None, "bilinear", True)

        return x


def _building_block_layer(num1, num2, num3, num4):
    return Sequential(
        OrderedDict([
            ("Res1", _building_block(64, 64, num1)),
            ("Res2", _building_block(64, 128, num2)),
            ("Res3", _building_block(128, 256, num3)),
            ("Res4", _building_block(256, 512, num4))
        ])
    )


def _bottle_neck_layer(num1, num2, num3, num4):
    return Sequential(
        OrderedDict([
            ("Res1", _bottle_neck(64, 256, num1)),
            ("Res2", _bottle_neck(256, 512, num2)),
            ("Res3", _bottle_neck(512, 1024, num3)),
            ("Res4", _bottle_neck(1024, 2048, num4))
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


def _bottle_neck(in_channel, out_channel, layer_num):
    seq = Sequential()
    # 除了res1，第一次要做个步长为2的卷积相当于pool一下
    if in_channel == out_channel // 2:
        seq.add_module("BottleNeck1", _BottleNeck(in_channel, out_channel, 2, True))
    else:
        seq.add_module("BottleNeck1", _BottleNeck(in_channel, out_channel, 1, True))

    for i in range(layer_num-1):
        seq.add_module("BottleNeck{}".format(i+2), _BottleNeck(out_channel, out_channel))

    return seq


class _BuildingBlock(Module):
    def __init__(self, in_channel, out_channel, stride=1, first_in=False):
        super(_BuildingBlock, self).__init__()
        self.conv0 = Conv2d(in_channel, out_channel, 3, stride, 1)
        self.conv1 = Conv2d(out_channel, out_channel, 3, 1, 1)
        self.bn = BatchNorm2d(out_channel)

        # 相加时第一次通道数不同，需要增加通道
        self.conv_extend = Conv2d(in_channel, out_channel, 1, stride)
        self.first_in = first_in

    def forward(self, x):
        out = self.conv0(x)
        out = self.bn(out)
        out = F.relu(out, True)
        out = self.conv1(out)
        out = self.bn(out)

        if self.first_in:
            x = self.conv_extend(x)

        out = F.relu(x + out, True)

        return out


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

    def forward(self, x):
        x0 = self.conv0(x)
        x0 = self.bn0(x0)
        x0 = F.relu(x0, True)
        x0 = self.conv1(x0)
        x0 = self.bn0(x0)
        x0 = F.relu(x0, True)
        x0 = self.conv2(x0)
        x0 = self.bn1(x0)

        if self.first_in:
            x = self.conv_extend(x)

        x0 = F.relu(x + x0, True)

        return x0


class _ASPP(Module):
    def __init__(self, in_channel, out_channel, dilation, num_classes):
        super(_ASPP, self).__init__()
        self.conv = Conv2d(in_channel, out_channel, 3, 1, dilation, dilation)
        self.fconv0 = Conv2d(out_channel, out_channel, 1)
        self.fconv1 = Conv2d(out_channel, num_classes, 1)

    def forward(self, x):
        x = self.conv(x)
        x = F.relu(x, True)
        x = self.fconv0(x)
        x = F.relu(x, True)
        x = self.fconv1(x)

        return x

