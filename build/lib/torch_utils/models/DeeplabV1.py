""" Deeplabv1 backbone: VGG16 """
from collections import OrderedDict
from torch.nn import *
import torch.nn.functional as F


class DeeplabV1(Module):
    def __init__(self, num_classes=21, vgg_based_type=16, bn=True):
        super(DeeplabV1, self).__init__()
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

        self.num_classes = num_classes if num_classes != 0 else 21

        self.fconv = Sequential(
            OrderedDict([
                ("fconv0", _ConvReluDrop(512, 1024, 3, 1, 12, 12)),
                ("fconv1", _ConvReluDrop(1024, 1024, 1, 1)),
                ("fconv2", Conv2d(1024, self.num_classes, 1))
            ])
        )

    def forward(self, x):
        h, w = x.shape[-2:]
        x = self.conv(x)
        x = self.fconv(x)
        x = F.interpolate(x, (h, w), None, "bilinear", True)

        return x


def _conv_layer(num1, num2, num3, num4, num5, bn):
    return Sequential(
        OrderedDict([
            ("conv0", _Conv(3, 64, num1, bn)),
            ("conv1", _Conv(64, 128, num2, bn)),
            ("conv2", _Conv(128, 256, num3, bn)),
            ("conv3", _Conv(256, 512, num4, bn, 1, 2)),
            ("conv4", _Conv(512, 512, num5, bn, 1, 2))
        ])
    )


class _Conv(Module):
    def __init__(self, in_channel, out_channel, layer_num, is_bn, stride=2, dilation=1):
        super(_Conv, self).__init__()
        self.seq = Sequential()
        self.seq.add_module("conv0", Conv2d(in_channel, out_channel, 3, 1, dilation, dilation))
        for i in range(layer_num - 1):
            self.seq.add_module("conv{}".format(i + 1), Conv2d(out_channel, out_channel, 3, 1, dilation, dilation))
        self.pool = MaxPool2d(3, stride, 1)
        self.bn = BatchNorm2d(out_channel)
        self.is_bn = is_bn


    def forward(self, x):
        for idx, layer_name in enumerate(self.seq._modules):
            x = self.seq._modules[layer_name](x)
            if self.is_bn and idx > 0:
                x = self.bn(x)
            x = F.relu(x, True)

        x = self.pool(x)

        return x


class _ConvReluDrop(Module):
    def __init__(self, in_features, out_features, kernel_size, stride, padding=0, dilation=1):
        super(_ConvReluDrop, self).__init__()
        self.conv = Conv2d(in_features, out_features, kernel_size, stride, padding, dilation)
        self.drop = Dropout(0.5)

    def forward(self, x):
        x = self.conv(x)
        x = F.relu(x, True)
        x = self.drop(x)

        return x
