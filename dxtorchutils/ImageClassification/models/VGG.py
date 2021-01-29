"""VGG"""
from collections import OrderedDict
import torch
from torch.nn import *
import torch.nn.functional as torch


class VGG11(Module):
    def __init__(self, bn=True):
        super(VGG11, self).__init__()
        self.conv = _conv_layer(1, 1, 2, 2, 2, bn)
        self.fc = Sequential(
            OrderedDict([
                ("fc1", _FC(25088, 4096)),
                ("fc2", _FC(4096, 4096))
            ])
        )
        self.out = Linear(4096, 1000)


    def forward(self, input):
        # (n, 3, 224, 224)
        x = self.conv(input)
        # (n, 512, 7, 7)
        x = x.view(x.shape[0], -1)
        # (n, 25088)
        x = self.fc(x)
        # (n, 4096)
        output = self.out(x)
        # (n, 1000)

        return output


class VGG13(Module):
    def __init__(self, bn=True):
        super(VGG13, self).__init__()
        self.conv = _conv_layer(2, 2, 2, 2, 2, bn)
        self.fc = Sequential(
            OrderedDict([
                ("fc1", _FC(25088, 4096)),
                ("fc2", _FC(4096, 4096))
            ])
        )
        self.out = Linear(4096, 1000)


    def forward(self, input):
        # (n, 3, 224, 224)
        x = self.conv(input)
        # (n, 512, 7, 7)
        x = x.view(x.shape[0], -1)
        # (n, 25088)
        x = self.fc(x)
        # (n, 4096)
        output = self.out(x)
        # (n, 1000)

        return output


class VGG16(Module):
    def __init__(self, bn=True):
        super(VGG16, self).__init__()
        self.conv = _conv_layer(2, 2, 3, 3, 3, bn)
        self.fc = Sequential(
            OrderedDict([
                ("fc1", _FC(25088, 4096)),
                ("fc2", _FC(4096, 4096))
            ])
        )
        self.out = Linear(4096, 1000)


    def forward(self, input):
        # (n, 3, 224, 224)
        x = self.conv(input)
        # (n, 512, 7, 7)
        x = x.view(x.shape[0], -1)
        # (n, 25088)
        x = self.fc(x)
        # (n, 4096)
        output = self.out(x)
        # (n, 1000)

        return output


class VGG19(Module):
    def __init__(self, bn=True):
        super(VGG19, self).__init__()
        self.conv = _conv_layer(2, 2, 4, 4, 4, bn)
        self.fc = Sequential(
            OrderedDict([
                ("fc1", _FC(25088, 4096)),
                ("fc2", _FC(4096, 4096))
            ])
        )
        self.out = Linear(4096, 1000)


    def forward(self, input):
        # (n, 3, 224, 224)
        x = self.conv(input)
        # (n, 512, 7, 7)
        x = x.view(x.shape[0], -1)
        # (n, 25088)
        x = self.fc(x)
        # (n, 4096)
        output = self.out(x)
        # (n, 1000)

        return output


def _conv_layer(num1, num2, num3, num4, num5, bn):
    return Sequential(
        OrderedDict([
            ("conv0", _Conv(3, 64, num1, bn)),
            ("conv1", _Conv(64, 128, num2, bn)),
            ("conv2", _Conv(128, 256, num3, bn)),
            ("conv3", _Conv(256, 512, num4, bn)),
            ("conv4", _Conv(512, 512, num5, bn))
        ])
    )


class _Conv(Module):
    def __init__(self, in_channels, out_channels, layer_num, is_bn):
        super(_Conv, self).__init__()
        self.seq = Sequential()
        self.seq.add_module("conv0", Conv2d(in_channels, out_channels, 3, 1, 1))
        self.bns = Sequential()
        self.bns.add_module("bn0", BatchNorm2d(out_channels))
        for i in range(layer_num - 1):
            self.seq.add_module("conv{}".format(i + 1), Conv2d(out_channels, out_channels, 3, 1, 1))
            self.bns.add_module("bn{}".format(i + 1), BatchNorm2d(out_channels))

        self.pool = MaxPool2d(2)
        self.is_bn = is_bn

    def forward(self, input):
        x = input
        for bn_name, conv_name in zip(self.bns._modules, self.seq._modules):
            x = self.seq._modules[conv_name](x)
            if self.is_bn:
                x = self.bns._modules[bn_name](x)
            x = torch.relu(x)

        output = self.pool(x)

        return output


class _FC(Module):
    def __init__(self, in_features, out_features):
        super(_FC, self).__init__()
        self.fc = Linear(in_features, out_features)

    def forward(self, input):
        x = self.fc(input)
        output = torch.relu(x)

        return output
