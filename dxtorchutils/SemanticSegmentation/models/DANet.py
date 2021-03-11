from torch.nn import *
from collections import OrderedDict


class DANet(Module):
    def __init__(self, num_classes=21, resnet_based_type=101):
        super(DANet, self).__init__()
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

if __name__ == '__main__':
    # calculate parameters
    from thop import profile
    from torchsummary import summary

    model = DeeplabV3()
    input = torch.randn((1, 3, 224, 224))
    macs, params = profile(model, inputs=(input,))
    summary(model, (3, 224, 224))

    print("MACs: {}".format(macs))
