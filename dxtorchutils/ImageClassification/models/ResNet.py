"""
ResNet18
    Input: (3, 224, 224)

    Total params: 11,692,520
    Trainable params: 11,692,520
    Non-trainable params: 0

    Input size (MB): 0.57
    Forward/backward pass size (MB): 60.64
    Params size (MB): 44.60
    Estimated Total Size (MB): 105.82

    MACs/FLOPs: 1,804,745,984

ResNet32
    Input: (3, 224, 224)
    Total params: 21,804,392
    Trainable params: 21,804,392
    Non-trainable params: 0

    Input size (MB): 0.57
    Forward/backward pass size (MB): 93.76
    Params size (MB): 83.18
    Estimated Total Size (MB): 177.51

    MACs/FLOPs: 3,649,970,816

ResNet50
    Input: (3, 224, 224)

    Total params: 25,575,912
    Trainable params: 25,575,912
    Non-trainable params: 0

    Input size (MB): 0.57
    Forward/backward pass size (MB): 259.17
    Params size (MB): 97.56
    Estimated Total Size (MB): 357.31

    MACs/FLOPs: 3,862,772,352

ResNet101
    Input: (3, 224, 224)

    Total params: 44,594,152
    Trainable params: 44,594,152
    Non-trainable params: 0

    Input size (MB): 0.57
    Forward/backward pass size (MB): 402.34
    Params size (MB): 170.11
    Estimated Total Size (MB): 573.03

    MACs/FLOPs: 7,590,347,392

ResNet152
    Input: (3, 224, 224)

    Total params: 60,260,840
    Trainable params: 60,260,840
    Non-trainable params: 0

    Input size (MB): 0.57
    Forward/backward pass size (MB): 579.20
    Params size (MB): 229.88
    Estimated Total Size (MB): 809.65

    MACs/FLOPs: 11,321,535,104
"""
from dxtorchutils.utils.layers import *


class ResNet18(Module):
    def __init__(self):
        super(ResNet18, self).__init__()
        self.conv0 = conv_relu_bn(3, 64, 7, 2, 3)
        self.conv1 = _building_block_layer(2, 2, 2, 2)

        self.pool0 = MaxPool2d(3, 2)
        self.pool1 = AdaptiveAvgPool2d((1, 1))

        self.out = Linear(512, 1000)


    def forward(self, input):
        # (n, 3, 224, 224)
        x = self.conv0(input)

        x = self.pool0(x)

        # (n, 64, 56, 56)
        x = self.conv1(x)

        # (n, 512, 7, 7)
        x = self.pool1(x)

        # (n, 512, 1, 1)
        x = x.view(x.shape[0], -1)

        # (n, 512)
        output = self.out(x)

        # (n, 1000)

        return output


class ResNet34(Module):
    def __init__(self):
        super(ResNet34, self).__init__()
        self.conv0 = conv_relu_bn(3, 64, 7, 2, 3)
        self.conv1 = _building_block_layer(3, 4, 6, 3)

        self.pool0 = MaxPool2d(3, 2)
        self.pool1 = AdaptiveAvgPool2d((1, 1))

        self.out = Linear(512, 1000)


    def forward(self, input):
        # (n, 3, 224, 224)
        x = self.conv0(input)


        x = self.pool0(x)

        # (n, 64, 56, 56)
        x = self.conv1(x)

        # (n, 512, 7, 7)
        x = self.pool1(x)

        # (n, 512, 1, 1)
        x = x.view(x.shape[0], -1)

        # (n, 512)
        output = self.out(x)

        # (n, 1000)

        return output


class ResNet50(Module):
    def __init__(self):
        super(ResNet50, self).__init__()
        self.conv0 = conv_relu_bn(3, 64, 7, 2, 3)
        self.conv1 = _bottle_neck_layer(3, 4, 6, 3)

        self.pool0 = MaxPool2d(3, 2)
        self.pool1 = AdaptiveAvgPool2d((1, 1))

        self.out = Linear(2048, 1000)

    def forward(self, input):
        # (n, 3, 224, 224)
        x = self.conv0(input)

        x = self.pool0(x)

        # (n, 64, 56, 56)
        x = self.conv1(x)

        # (n, 2048, 7, 7)
        x = self.pool1(x)

        # (n, 2048, 1, 1)
        x = x.view(x.shape[0], -1)

        # (n, 2048)
        output = self.out(x)

        # (n, 1000)

        return output


class ResNet101(Module):
    def __init__(self):
        super(ResNet101, self).__init__()
        self.conv0 = conv_relu_bn(3, 64, 7, 2, 3)
        self.conv1 = _bottle_neck_layer(3, 4, 23, 3)

        self.pool0 = MaxPool2d(3, 2)
        self.pool1 = AdaptiveAvgPool2d((1, 1))

        self.out = Linear(2048, 1000)

    def forward(self, input):
        # (n, 3, 224, 224)
        x = self.conv0(input)

        x = self.pool0(x)

        # (n, 64, 56, 56)
        x = self.conv1(x)

        # (n, 2048, 7, 7)
        x = self.pool1(x)

        # (n, 2048, 1, 1)
        x = x.view(x.shape[0], -1)

        # (n, 2048)
        output = self.out(x)

        # (n, 1000)

        return output


class ResNet152(Module):
    def __init__(self):
        super(ResNet152, self).__init__()
        self.conv0 = conv_relu_bn(3, 64, 7, 2, 3)
        self.conv1 = _bottle_neck_layer(3, 8, 36, 3)

        self.pool0 = MaxPool2d(3, 2)
        self.pool1 = AdaptiveAvgPool2d((1, 1))

        self.out = Linear(2048, 1000)

    def forward(self, input):
        # (n, 3, 244, 244)
        x = self.conv0(input)

        x = self.pool0(x)

        # (n, 64, 56, 56)
        x = self.conv1(x)

        # (n, 2048, 7, 7)
        x = self.pool1(x)

        # (n, 2048, 1, 1)
        x = x.view(x.shape[0], -1)

        # (n, 2048)
        output = self.out(x)

        # (n, 1000)

        return output


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
        self.conv0 = conv_relu_bn(in_channel, out_channel, 3, stride, 1)
        self.conv1 = Conv2d(out_channel, out_channel, 3, 1, 1)
        self.normalization = BatchNorm2d(out_channel)
        self.activation = ReLU(True)

        # 相加时第一次通道数不同，需要增加通道
        self.conv_extend = Conv2d(in_channel, out_channel, 1, stride)
        self.first_in = first_in

    def forward(self, input):
        x = self.conv0(input)
        x = self.conv1(x)
        x = self.normalization(x)

        if self.first_in:
            input = self.conv_extend(input)

        output = self.activation(x + input)

        return output


class _BottleNeck(Module):
    def __init__(self, in_channel, out_channel, stride=1, first_in=False):
        super(_BottleNeck, self).__init__()
        # 所有中间层都是输出层通道数的 1/4
        mid_channel = out_channel // 4

        self.conv0 = conv_relu_bn(in_channel, mid_channel, 1, stride)
        self.conv1 = conv_relu_bn(mid_channel, mid_channel, 3, 1, 1)
        self.conv2 = Conv2d(mid_channel, out_channel, 1)
        self.normalization = BatchNorm2d(out_channel)
        self.activation = ReLU(True)

        # 相加时第一次通道数不同，需要增加通道
        self.conv_extend = Conv2d(in_channel, out_channel, 1, stride)
        self.first_in = first_in

    def forward(self, input):
        x = self.conv0(input)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.normalization(x)

        if self.first_in:
            input = self.conv_extend(input)

        output = self.activation(input + x)

        return output


if __name__ == '__main__':
    # calculate parameters
    from thop import profile
    from torchsummary import summary

    model0 = ResNet18()
    model1 = ResNet34()
    model2 = ResNet50()
    model3 = ResNet101()
    model4 = ResNet152()
    input = torch.randn((1, 3, 224, 224))
    macs0, params0 = profile(model0, inputs=(input, ))
    macs1, params1 = profile(model1, inputs=(input, ))
    macs2, params2 = profile(model2, inputs=(input, ))
    macs3, params3 = profile(model3, inputs=(input, ))
    macs4, params4 = profile(model4, inputs=(input, ))
    summary(model0, (3, 224, 224))
    summary(model1, (3, 224, 224))
    summary(model2, (3, 224, 224))
    summary(model3, (3, 224, 224))
    summary(model4, (3, 224, 224))

    print("MACs: {}".format(macs0))
    print("MACs: {}".format(macs1))
    print("MACs: {}".format(macs2))
    print("MACs: {}".format(macs3))
    print("MACs: {}".format(macs4))
