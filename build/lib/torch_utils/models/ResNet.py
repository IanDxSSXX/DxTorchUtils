"""ResNet"""
from collections import OrderedDict

import torch
from torch.nn import *
import torch.nn.functional as F


class ResNet18(Module):
    def __init__(self, num_classes=21):
        super(ResNet18, self).__init__()
        self.conv0 = Conv2d(3, 64, 7, 2, 4)
        self.conv1 = _building_block_layer(2, 2, 2, 2)

        self.pool0 = MaxPool2d(3, 2)
        self.pool1 = AvgPool2d(2)

        self.fc = Linear(4608, 1000)

        self.num_classes = num_classes
        self.out = Linear(1000, num_classes)

    def forward(self, x):
        # (n, 3, 244, 244)
        x = self.conv0(x)
        x = F.relu(x, True)
        x = self.pool0(x)

        # (n, 64, 56, 56)
        x = self.conv1(x)

        # (n, 512, 7, 7) / (n, 2048, 7, 7)
        x = self.pool1(x)

        # (n, 512, 3, 3) / (n, 512, 3, 3)
        x = torch.flatten(x, 1)

        # (n, 4608) / (n, 18432)
        x = self.fc(x)

        # (n, 1000)
        x = self.out(x)
        # (n, num_classes)

        return x


class ResNet34(Module):
    def __init__(self, num_classes=21):
        super(ResNet34, self).__init__()
        self.conv0 = Conv2d(3, 64, 7, 2, 4)
        self.conv1 = _building_block_layer(3, 4, 6, 3)
        self.pool0 = MaxPool2d(3, 2)
        self.pool1 = AvgPool2d(2)

        self.fc = Linear(4608, 1000)

        self.num_classes = num_classes
        self.out = Linear(1000, num_classes)

    def forward(self, x):
        # (n, 3, 244, 244)
        x = self.conv0(x)
        x = F.relu(x, True)
        x = self.pool0(x)

        # (n, 64, 56, 56)
        x = self.conv1(x)

        # (n, 512, 7, 7) / (n, 2048, 7, 7)
        x = self.pool1(x)

        # (n, 512, 3, 3) / (n, 512, 3, 3)
        x = torch.flatten(x, 1)

        # (n, 4608) / (n, 18432)
        x = self.fc(x)

        # (n, 1000)
        x = self.out(x)
        # (n, num_classes)

        return x


class ResNet50(Module):
    def __init__(self, num_classes=21):
        super(ResNet50, self).__init__()
        self.conv0 = Conv2d(3, 64, 7, 2, 4)
        self.conv1 = _bottle_neck_layer(3, 4, 6, 3)

        self.pool0 = MaxPool2d(3, 2)
        self.pool1 = AvgPool2d(2)

        self.fc = Linear(18432, 1000)

        self.num_classes = num_classes
        self.out = Linear(1000, num_classes)

    def forward(self, x):
        # (n, 3, 244, 244)
        x = self.conv0(x)
        x = F.relu(x, True)
        x = self.pool0(x)

        # (n, 64, 56, 56)
        x = self.conv1(x)

        # (n, 512, 7, 7) / (n, 2048, 7, 7)
        x = self.pool1(x)

        # (n, 512, 3, 3) / (n, 512, 3, 3)
        x = torch.flatten(x, 1)

        # (n, 4608) / (n, 18432)
        x = self.fc(x)

        # (n, 1000)
        x = self.out(x)
        # (n, num_classes)

        return x


class ResNet101(Module):
    def __init__(self, num_classes=21):
        super(ResNet101, self).__init__()
        self.conv0 = Conv2d(3, 64, 7, 2, 4)
        self.conv1 = _bottle_neck_layer(3, 4, 23, 3)

        self.pool0 = MaxPool2d(3, 2)
        self.pool1 = AvgPool2d(2)

        self.fc = Linear(18432, 1000)

        self.num_classes = num_classes
        self.out = Linear(1000, num_classes)

    def forward(self, x):
        # (n, 3, 244, 244)
        x = self.conv0(x)
        x = F.relu(x, True)
        x = self.pool0(x)

        # (n, 64, 56, 56)
        x = self.conv1(x)

        # (n, 512, 7, 7) / (n, 2048, 7, 7)
        x = self.pool1(x)

        # (n, 512, 3, 3) / (n, 512, 3, 3)
        x = torch.flatten(x, 1)

        # (n, 4608) / (n, 18432)
        x = self.fc(x)

        # (n, 1000)
        x = self.out(x)
        # (n, num_classes)

        return x


class ResNet152(Module):
    def __init__(self, num_classes=21):
        super(ResNet152, self).__init__()
        self.conv0 = Conv2d(3, 64, 7, 2, 4)
        self.conv1 = _bottle_neck_layer(3, 8, 36, 3)

        self.pool0 = MaxPool2d(3, 2)
        self.pool1 = AvgPool2d(2)

        self.fc = Linear(18432, 1000)

        self.num_classes = num_classes
        self.out = Linear(1000, num_classes)

    def forward(self, x):
        # (n, 3, 244, 244)
        x = self.conv0(x)
        x = F.relu(x, True)
        x = self.pool0(x)

        # (n, 64, 56, 56)
        x = self.conv1(x)

        # (n, 512, 7, 7) / (n, 2048, 7, 7)
        x = self.pool1(x)

        # (n, 512, 3, 3) / (n, 512, 3, 3)
        x = torch.flatten(x, 1)

        # (n, 4608) / (n, 18432)
        x = self.fc(x)

        # (n, 1000)
        x = self.out(x)
        # (n, num_classes)

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
