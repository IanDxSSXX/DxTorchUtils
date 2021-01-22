"""VGG"""
from collections import OrderedDict
import torch
from torch.nn import *
import torch.nn.functional as F

class VGG11(Module):
    def __init__(self, bn=True, num_classes=0):
        super(VGG11, self).__init__()
        self.conv = _conv_layer(1, 1, 2, 2, 2, bn)
        self.fc1 = Sequential(
            OrderedDict([
                ("fc1", _FConv(25088, 4096)),
                ("fc2", _FConv(4096, 4096))
            ])
        )
        self.fc2 = Linear(4096, 1000)

        self.num_classes = num_classes
        self.out = Linear(1000, num_classes)

    def forward(self, x):
        # (n, 3, 224, 224)
        x = self.conv(x)
        # (n, 512, 7, 7)
        x = torch.flatten(x, 1)
        # (n, 25088)
        x = self.fc1(x)
        # (n, 4096)
        x = self.fc2(x)
        # (n, 1000)
        if self.num_classes != 0:
            x = self.out(x)
            # (n, num_classes)

        return x


class VGG13(Module):
    def __init__(self, bn=True, num_classes=0):
        super(VGG13, self).__init__()
        self.conv = _conv_layer(2, 2, 2, 2, 2, bn)
        self.fc1 = Sequential(
            OrderedDict([
                ("fc1", _FConv(25088, 4096)),
                ("fc2", _FConv(4096, 4096))
            ])
        )
        self.fc2 = Linear(4096, 1000)

        self.num_classes = num_classes
        self.out = Linear(1000, num_classes)

    def forward(self, x):
        # (n, 3, 224, 224)
        x = self.conv(x)
        # (n, 512, 7, 7)
        x = torch.flatten(x, 1)
        # (n, 25088)
        x = self.fc1(x)
        # (n, 4096)
        x = self.fc2(x)
        # (n, 1000)
        if self.num_classes != 0:
            x = self.out(x)
            # (n, num_classes)

        return x


class VGG16(Module):
    def __init__(self, bn=True, num_classes=0):
        super(VGG16, self).__init__()
        self.conv = _conv_layer(2, 2, 3, 3, 3, bn)
        self.fc1 = Sequential(
            OrderedDict([
                ("fc1", _FConv(25088, 4096)),
                ("fc2", _FConv(4096, 4096))
            ])
        )
        self.fc2 = Linear(4096, 1000)

        self.num_classes = num_classes
        self.out = Linear(1000, num_classes)

    def forward(self, x):
        # (n, 3, 224, 224)
        x = self.conv(x)
        # (n, 512, 7, 7)
        x = torch.flatten(x, 1)
        # (n, 25088)
        x = self.fc1(x)
        # (n, 4096)
        x = self.fc2(x)
        # (n, 1000)
        if self.num_classes != 0:
            x = self.out(x)
            # (n, num_classes)

        return x


class VGG19(Module):
    def __init__(self, bn=True, num_classes=0):
        super(VGG19, self).__init__()
        self.conv = _conv_layer(2, 2, 4, 4, 4, bn)
        self.fc1 = Sequential(
            OrderedDict([
                ("fc1", _FConv(25088, 4096)),
                ("fc2", _FConv(4096, 4096))
            ])
        )
        self.fc2 = Linear(4096, 1000)

        self.num_classes = num_classes
        self.out = Linear(1000, num_classes)

    def forward(self, x):
        # (n, 3, 224, 224)
        x = self.conv(x)
        # (n, 512, 7, 7)
        x = torch.flatten(x, 1)
        # (n, 25088)
        x = self.fc1(x)
        # (n, 4096)
        x = self.fc2(x)
        # (n, 1000)
        if self.num_classes != 0:
            x = self.out(x)
            # (n, num_classes)

        return x


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
    def __init__(self, in_channel, out_channel, layer_num, is_bn):
        super(_Conv, self).__init__()
        self.seq = Sequential()
        self.seq.add_module("conv0", Conv2d(in_channel, out_channel, 3, 1, 1))
        for i in range(layer_num - 1):
            self.seq.add_module("conv{}".format(i + 1), Conv2d(out_channel, out_channel, 3, 1, 1))

        self.pool = MaxPool2d(2)
        self.bn = BatchNorm2d(out_channel)

        self.is_bn = is_bn

    def forward(self, x):
        for layer_name in self.seq._modules:
            x = self.seq._modules[layer_name](x)
            if self.is_bn:
                x = self.bn(x)
            x = F.relu(x, True)
        x = self.pool(x)

        return x


class _FConv(Module):
    def __init__(self, in_features, out_features):
        super(_FConv, self).__init__()
        self.fconv = Conv2d(in_features, out_features, 1)

    def forward(self, x):
        x = self.fconv(x)
        x = F.relu(x, True)

        return x