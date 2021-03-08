"""
Unet +++
    Input size (MB): 0.57
    Forward/backward pass size (MB): 704.38
    Params size (MB): 99.91
    Estimated Total Size (MB): 804.86
"""
from dxtorchutils.utils.layers import *
import torch.nn.functional as F


class UNet3p(Module):
    def __init__(self, n_classed=2, in_features=3):
        super(UNet3p, self).__init__()
        self.en_block0 = _ConvBlock(in_features, 64)
        self.en_block1 = _ConvBlock(64, 128)
        self.en_block2 = _ConvBlock(128, 256)
        self.en_block3 = _ConvBlock(256, 512)
        self.en_block4 = _ConvBlock(512, 1024)

        self.pool0 = MaxPool2d(2)
        self.pool1 = MaxPool2d(2)
        self.pool2 = MaxPool2d(2)
        self.pool3 = MaxPool2d(2)

        self.de_block0 = _FSSC(3)
        self.de_block1 = _FSSC(2)
        self.de_block2 = _FSSC(1)
        self.de_block3 = _FSSC(0)

        self.final_layer = Conv2d(320, n_classed, 1)

    def forward(self, input):
        x0 = self.en_block0(input)

        x1 = self.en_block1(self.pool0(x0))

        x2 = self.en_block2(self.pool1(x1))

        x3 = self.en_block3(self.pool2(x2))

        x4 = self.en_block4(self.pool3(x3))

        x3_de = self.de_block0(x0, x1, x2, x3, x4)

        x2_de = self.de_block1(x0, x1, x2, x3_de, x4)

        x1_de = self.de_block2(x0, x1, x2_de, x3_de, x4)

        x0_de = self.de_block3(x0, x1_de, x2_de, x3_de, x4)

        out = self.final_layer(x0_de)

        return out


class _ConvBlock(Module):
    def __init__(self, in_channels, out_channels):
        super(_ConvBlock, self).__init__()
        self.conv1 = conv_relu_bn(in_channels, out_channels, 3, 1, 1)
        self.conv2 = conv_relu_bn(out_channels, out_channels, 3, 1, 1)


    def forward(self, input):
        x = self.conv1(input)
        output = self.conv2(x)

        return output


class _FSSC(Module):
    def __init__(self, pos):
        super(_FSSC, self).__init__()
        self.convs = Sequential()
        for i in range(pos + 1):
            self.convs.add_module("conv{}".format(i), Conv2d(64 * 2 ** i, 64, 3, 1, 1))

        for i in range(pos + 1, 4):
            self.convs.add_module("conv{}".format(i), Conv2d(320, 64, 3, 1, 1))

        self.convs.add_module("conv4", Conv2d(1024, 64, 3, 1, 1))
        self.conv = conv_relu_bn(320, 320, 3, 1, 1)
        self.pos = pos

        self.pools = Sequential()
        for i in range(pos):
            self.pools.add_module("pool{}".format(i), MaxPool2d(2 ** (pos - i)))


    def forward(self, x0, x1, x2, x3, x4):
        xs = [x0, x1, x2, x3, x4]
        first_in = True
        h0, w0 = xs[self.pos].shape[-2:]
        for idx, x in enumerate(xs):
            if idx != self.pos:
                if idx < self.pos:
                    x = self.pools._modules["pool{}".format(idx)](x)
                else:
                    x = F.interpolate(x, (h0, w0), None, "bilinear", True)

            x = self.convs._modules["conv{}".format(idx)](x)

            if first_in:
                x0 = x
                first_in = False
            else:
                x0 = torch.cat((x0, x), 1)

        output = self.conv(x0)

        return output


if __name__ == '__main__':
    # calculate parameters
    from thop import profile
    from torchsummary import summary

    model = UNet3p()
    input = torch.randn((1, 3, 224, 224))
    macs, params = profile(model, inputs=(input, ))
    summary(model, (3, 224, 224))

    print("MACs: {}".format(macs))
