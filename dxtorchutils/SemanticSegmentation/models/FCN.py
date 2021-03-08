"""
FCN8s VGG16
    Input: (3, 224, 224)

    Total params: 33,707,903
    Trainable params: 33,707,903
    Non-trainable params: 0

    Input size (MB): 0.57
    Forward/backward pass size (MB): 339.72
    Params size (MB): 128.59
    Estimated Total Size (MB): 468.88

    MACs/FLOPs: 16,323,077,225

FCN16s VGG16
    Input: (3, 224, 224)

    Total params: 33,702,506
    Trainable params: 33,702,506
    Non-trainable params: 0

    Input size (MB): 0.57
    Forward/backward pass size (MB): 339.59
    Params size (MB): 128.56
    Estimated Total Size (MB): 468.73

    MACs/FLOPs: 16,318,845,977

FCN32s VGG16
    Input: (3, 224, 224)

    Total params: 33,691,733
    Trainable params: 33,691,733
    Non-trainable params: 0

    Input size (MB): 0.57
    Forward/backward pass size (MB): 339.56
    Params size (MB): 128.52
    Estimated Total Size (MB): 468.66

    MACs/FLOPs: 16,316,734,469
"""
import torch.nn.functional as F
from dxtorchutils.utils.layers import *


class FCN32s(Module):
    def __init__(self, n_classes=21, vgg_based_type=16):
        super(FCN32s, self).__init__()
        if vgg_based_type == 11:
            self.conv = _conv_layer(1, 1, 2, 2, 2)
        elif vgg_based_type == 12:
            self.conv = _conv_layer(2, 2, 2, 2, 2)
        elif vgg_based_type == 16:
            self.conv = _conv_layer(2, 2, 3, 3, 3)
        elif vgg_based_type == 19:
            self.conv = _conv_layer(2, 2, 4, 4, 4)
        else:
            exit("Wrong Backbone Type")

        self.fconv = Sequential(
            OrderedDict([
                ("fc1", _fconv(512, 4096)),
                ("fc2", _fconv(4096, 4096)),
                ("fc3", Conv2d(4096, n_classes, 1))
            ])
        )

        self.up_sample = Upsample(None, 32)


    def forward(self, input):
        h, w = input.shape[-2:]
        x = self.conv(input)
        x = self.fconv(x)
        output = F.interpolate(x, (h, w), None, "bilinear", True)


        return output


class FCN16s(Module):
    def __init__(self, n_classes=21, vgg_based_type=16):
        super(FCN16s, self).__init__()
        if vgg_based_type == 11:
            conv = _conv_layer(1, 1, 2, 2, 2)
        elif vgg_based_type == 12:
            conv = _conv_layer(2, 2, 2, 2, 2)
        elif vgg_based_type == 16:
            conv = _conv_layer(2, 2, 3, 3, 3)
        elif vgg_based_type == 19:
            conv = _conv_layer(2, 2, 4, 4, 4)
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
                ("fconv0", _fconv(512, 4096)),
                ("fconv1", _fconv(4096, 4096)),
                ("fconv2", Conv2d(4096, n_classes, 1))
            ])
        )

        self.up_sample1 = Upsample(None, 2)
        self.up_sample2 = Upsample(None, 16)


    def forward(self, input):
        h, w = input.shape[-2:]
        x = self.conv0(input)
        x = self.conv1(x)
        x = self.conv2(x)
        x1 = self.conv3(x)
        x2 = self.conv4(x1)
        x1 = self.conv5(x1)
        x2 = self.fconv(x2)

        h0, w0 = x1.shape[-2:]
        x2 = F.interpolate(x2, (h0, w0), None, "bilinear", True)

        x = x1 + x2
        output = F.interpolate(x, (h, w), None, "bilinear", True)

        return output


class FCN8s(Module):
    def __init__(self, n_classes=21, vgg_based_type=16):
        super(FCN8s, self).__init__()
        if vgg_based_type == 11:
            conv = _conv_layer(1, 1, 2, 2, 2)
        elif vgg_based_type == 12:
            conv = _conv_layer(2, 2, 2, 2, 2)
        elif vgg_based_type == 16:
            conv = _conv_layer(2, 2, 3, 3, 3)
        elif vgg_based_type == 19:
            conv = _conv_layer(2, 2, 4, 4, 4)
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
                ("fc1", _fconv(512, 4096)),
                ("fc2", _fconv(4096, 4096)),
                ("fc3", Conv2d(4096, n_classes, 1))
            ])
        )


    def forward(self, input):
        h, w = input.shape[-2:]
        x = self.conv0(input)
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

        output = F.interpolate(x, (h, w), None, "bilinear", True)

        return output


def _conv_layer(num1, num2, num3, num4, num5):
    return Sequential(
        OrderedDict([
            ("conv0", _Conv(3, 64, num1)),
            ("conv1", _Conv(64, 128, num2)),
            ("conv2", _Conv(128, 256, num3)),
            ("conv3", _Conv(256, 512, num4)),
            ("conv4", _Conv(512, 512, num5))
        ])
    )


def _fconv(in_channels, out_channels):
    return Sequential(
        OrderedDict([
            ("conv", Conv2d(in_channels, out_channels, 1)),
            ("activation", ReLU(True))
        ])
    )


class _Conv(Module):
    def __init__(self, in_channels, out_channels, layer_num):
        super(_Conv, self).__init__()
        self.convs = Sequential()
        self.convs.add_module("conv0", conv_relu_bn(in_channels, out_channels, 3, 1, 1))

        for i in range(layer_num - 1):
            self.convs.add_module("conv{}".format(i + 1), conv_relu_bn(out_channels, out_channels, 3, 1, 1))

        self.pool = MaxPool2d(2)

    def forward(self, input):
        x = input

        x = self.convs(x)

        output = self.pool(x)

        return output


if __name__ == '__main__':
    # calculate parameters
    from thop import profile
    from torchsummary import summary

    model0 = FCN8s()
    model1 = FCN16s()
    model2 = FCN32s()
    input = torch.randn((1, 3, 224, 224))
    macs0, params0 = profile(model0, inputs=(input,))
    macs1, params1 = profile(model1, inputs=(input,))
    macs2, params2 = profile(model2, inputs=(input,))
    summary(model0, (3, 224, 224))
    summary(model1, (3, 224, 224))
    summary(model2, (3, 224, 224))

    print("MACs: {}".format(macs0))
    print("MACs: {}".format(macs1))
    print("MACs: {}".format(macs2))
