""" Dense ASPP """

from collections import OrderedDict
import torch
from torch.nn import *


class DenseASPP(Module):
    def __init__(self, img_size, num_classes=19, dense_type="201", ASPP_array=None, drop_out=0):
        super(DenseASPP, self).__init__()

        if ASPP_array is None:
            ASPP_array = [3, 6, 12, 18, 24]
        self.num_classes = num_classes

        layer_nums = []
        if dense_type == "121":
            layer_nums = [6, 12, 24, 16, len(ASPP_array)]
        elif dense_type == "169":
            layer_nums = [6, 12, 32, 32, len(ASPP_array)]
        elif dense_type == "201":
            layer_nums = [6, 12, 48, 32, len(ASPP_array)]
        elif dense_type == "263":
            layer_nums = [6, 12, 64, 48, len(ASPP_array)]
        else:
            exit("Wrong DenseASPP Type")

        self.convolution = Conv2d(3, 64, 7, 2, 3)

        self.pooling = Sequential(
            OrderedDict([
                ("bn_relu", _bn_relu(64)),
                ("pool", MaxPool2d(3, 2))
            ])
        )

        self.denseBlock1 = _DenseBlock(64, layer_nums[0])
        dense_out_channels = 64 + layer_nums[0] * 32
        self.transition1 = _transition_layer(dense_out_channels)

        self.denseBlock2 = _DenseBlock(dense_out_channels // 2, layer_nums[1])
        dense_out_channels = dense_out_channels // 2 + layer_nums[1] * 32
        self.transition2 = _transition_layer(dense_out_channels)

        self.denseBlock3 = _DenseBlock(dense_out_channels // 2, layer_nums[2])
        dense_out_channels = dense_out_channels // 2 + layer_nums[2] * 32
        self.transition3 = _transition_layer(dense_out_channels)

        self.denseBlock4 = _DenseBlock(dense_out_channels // 2, layer_nums[3])
        dense_out_channels = dense_out_channels // 2 + layer_nums[3] * 32
        self.transition4 = _transition_layer(dense_out_channels)

        self.denseBlockASPP = _DenseBlock(dense_out_channels // 2, layer_nums[4], ASPP_array)
        dense_out_channels = dense_out_channels // 2 + layer_nums[4] * 32

        self.classification = Sequential(
            OrderedDict([
                ("drop_out", Dropout2d(drop_out)),
                ("conv", Conv2d(dense_out_channels, num_classes, kernel_size=1, padding=0))
            ])
        )

        self.up_sample = Upsample(img_size, mode="bilinear", align_corners=True)

    def forward(self, x_in):
        x = self.convolution(x_in)
        x = self.pooling(x)
        x = self.denseBlock1(x)
        x = self.transition1(x)
        x = self.denseBlock2(x)
        x = self.transition2(x)
        x = self.denseBlock3(x)
        x = self.transition3(x)
        x = self.denseBlock4(x)
        x = self.transition4(x)
        x = self.denseBlockASPP(x)
        x = self.classification(x)
        x = self.up_sample(x)

        return x


class _DenseBlock(Module):
    def __init__(self, in_channels, layer_num, dilation=None):
        super(_DenseBlock, self).__init__()
        if dilation is None:
            dilation = [1] * layer_num
        if isinstance(dilation, int):
            dilation = [dilation] * layer_num
        self.denseLayers = Sequential()
        for i in range(layer_num):
            self.denseLayers.add_module(
                "dense_layer{}".format(i + 1),
                _dense_layer(in_channels + 32 * i, dilation[i])
            )

    def forward(self, x_in):
        for denseLayer in self.denseLayers:
            x_out = denseLayer(x_in)
            x_in = torch.cat((x_in, x_out), 1)

        return x_in


def _bn_relu(in_channels):
    return Sequential(
        OrderedDict([
            ("bn", BatchNorm2d(in_channels)),
            ("relu", ReLU(inplace=True))
        ])
    )


def _dense_layer(in_channels, dilation=1):
    return Sequential(
        OrderedDict([
            ("bn_relu1", _bn_relu(in_channels)),
            ("conv1", Conv2d(in_channels, 128, 1)),
            ("bn_relu2", _bn_relu(128)),
            ("conv2", Conv2d(128, 32, 3, 1, 1 * dilation, dilation=dilation))
        ])
    )


def _transition_layer(in_channels):
    return Sequential(
        OrderedDict([
            ("bn_relu", _bn_relu(in_channels)),
            ("conv", Conv2d(in_channels, in_channels // 2, 1))
        ])
    )
