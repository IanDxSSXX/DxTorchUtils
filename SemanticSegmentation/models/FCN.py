""" VGG based FCN """
from collections import OrderedDict
from torch.nn import *
import torch.nn.functional as F


class FCN32s(Module):
    def __init__(self, n_classes=21, vgg_based_type=16, bn=True):
        super(FCN32s, self).__init__()
        if vgg_based_type == 11:
            self.conv = _conv_layer(1, 1, 2, 2, 2, bn)
        elif vgg_based_type == 12:
            self.conv = _conv_layer(2, 2, 2, 2, 2, bn)
        elif vgg_based_type == 16:
            self.conv = _conv_layer(2, 2, 3, 3, 3, bn)
        elif vgg_based_type == 19:
            self.conv = _conv_layer(2, 2, 4, 4, 4, bn)
        else:
            exit("Wrong Backbone Type")

        self.fconv = Sequential(
            OrderedDict([
                ("fc1", _FConv(512, 4096)),
                ("fc2", _FConv(4096, 4096)),
                ("fc3", Conv2d(4096, n_classes, 1))
            ])
        )

        self.up_sample = Upsample(None, 32)

    def forward(self, x):
        h, w = x.shape[-2:]
        x = self.conv(x)
        x = self.fconv(x)
        x = F.interpolate(x, (h, w), None, "bilinear", True)


        return x


class FCN16s(Module):
    def __init__(self, n_classes=21, vgg_based_type=16, bn=True):
        super(FCN16s, self).__init__()
        if vgg_based_type == 11:
            conv = _conv_layer(1, 1, 2, 2, 2, bn)
        elif vgg_based_type == 12:
            conv = _conv_layer(2, 2, 2, 2, 2, bn)
        elif vgg_based_type == 16:
            conv = _conv_layer(2, 2, 3, 3, 3, bn)
        elif vgg_based_type == 19:
            conv = _conv_layer(2, 2, 4, 4, 4, bn)
        else:
            exit("Wrong Backbone Type")

        self.conv0 = conv._modules["conv0"]
        self.conv1 = conv._modules["conv1"]
        self.conv2 = conv._modules["conv2"]
        self.conv3 = conv._modules["conv3"]
        self.conv4 = conv._modules["conv4"]

        self.conv5 = Conv2d(512, n_classes, 1)
        self.fconv = Sequential(
            OrderedDict([
                ("fconv0", _FConv(512, 4096)),
                ("fconv1", _FConv(4096, 4096)),
                ("fconv2", Conv2d(4096, n_classes, 1))
            ])
        )

        self.up_sample1 = Upsample(None, 2)
        self.up_sample2 = Upsample(None, 16)

    def forward(self, x):
        h, w = x.shape[-2:]
        x = self.conv0(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x1 = self.conv3(x)
        x2 = self.conv4(x1)
        x1 = self.conv5(x1)
        x2 = self.fconv(x2)

        h0, w0 = x1.shape[-2:]
        x2 = F.interpolate(x2, (h0, w0), None, "bilinear", True)

        x = x1 + x2
        x = F.interpolate(x, (h, w), None, "bilinear", True)

        return x


class FCN8s(Module):
    def __init__(self, n_classes=21, vgg_based_type=16, bn=True):
        super(FCN8s, self).__init__()
        if vgg_based_type == 11:
            conv = _conv_layer(1, 1, 2, 2, 2, bn)
        elif vgg_based_type == 12:
            conv = _conv_layer(2, 2, 2, 2, 2, bn)
        elif vgg_based_type == 16:
            conv = _conv_layer(2, 2, 3, 3, 3, bn)
        elif vgg_based_type == 19:
            conv = _conv_layer(2, 2, 4, 4, 4, bn)
        else:
            exit("Wrong Backbone Type")

        self.conv0 = conv._modules["conv0"]
        self.conv1 = conv._modules["conv1"]
        self.conv2 = conv._modules["conv2"]
        self.conv3 = conv._modules["conv3"]
        self.conv4 = conv._modules["conv4"]

        self.conv5 = Conv2d(512, n_classes, 1)
        self.conv6 = Conv2d(256, n_classes, 1)

        self.fconv = Sequential(
            OrderedDict([
                ("fc1", _FConv(512, 4096)),
                ("fc2", _FConv(4096, 4096)),
                ("fc3", Conv2d(4096, n_classes, 1))
            ])
        )


    def forward(self, x):
        h, w = x.shape[-2:]
        x = self.conv0(x)
        x = self.conv1(x)
        x1 = self.conv2(x)
        x2 = self.conv3(x1)
        x3 = self.conv4(x2)
        x3 = self.fconv(x3)

        h0, w0 = x2.shape[-2:]
        x3 = F.interpolate(x3, (h0, w0), None, "bilinear", True)

        x2 = self.conv5(x2)
        x = x2 + x3

        h1, w1 = x1.shape[-2:]
        x = F.interpolate(x, (h1, w1), None, "bilinear", True)

        x1 = self.conv6(x1)
        x = x1 + x

        x = F.interpolate(x, (h, w), None, "bilinear", True)
        print(x.shape)
        return x


def _conv_layer(num1, num2, num3, num4, num5, is_bn):
    return Sequential(
        OrderedDict([
            ("conv0", _Conv(3, 64, num1, is_bn)),
            ("conv1", _Conv(64, 128, num2, is_bn)),
            ("conv2", _Conv(128, 256, num3, is_bn)),
            ("conv3", _Conv(256, 512, num4, is_bn)),
            ("conv4", _Conv(512, 512, num5, is_bn))
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
