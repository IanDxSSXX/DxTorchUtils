"""
VGG11
    Input: (3, 224, 224)

    Total params: 132,868,840
    Trainable params: 132,868,840
    Non-trainable params: 0

    Input size (MB): 0.57
    Forward/backward pass size (MB): 193.45
    Params size (MB): 506.85
    Estimated Total Size (MB): 700.88

    MACs/FLOPs: 7,631,368,192

VGG13
    Input: (3, 224, 224)

    Total params: 138,365,992
    Trainable params: 138,365,992
    Non-trainable params: 0

    Input size (MB): 0.57
    Forward/backward pass size (MB): 333.56
    Params size (MB): 527.82
    Estimated Total Size (MB): 861.96

    MACs/FLOPs: 15,510,906,880

VGG16
    Input: (3, 224, 224)

    Total params: 133,053,736
    Trainable params: 133,053,736
    Non-trainable params: 0

    Input size (MB): 0.57
    Forward/backward pass size (MB): 303.70
    Params size (MB): 507.56
    Estimated Total Size (MB): 811.84

    MACs/FLOPs: 11,345,195,008

VGG19
    Input: (3, 224, 224)

    Total params: 143,678,248
    Trainable params: 143,678,248
    Non-trainable params: 0

    Input size (MB): 0.57
    Forward/backward pass size (MB): 363.42
    Params size (MB): 548.09
    Estimated Total Size (MB): 912.08

    MACs/FLOPs: 19,676,618,752
"""
from dxtorchutils.utils.layers import *


class VGG11(Module):
    def __init__(self):
        super(VGG11, self).__init__()
        self.conv = conv_layer(1, 1, 2, 2, 2)
        self.fc = Sequential(
            OrderedDict([
                ("fc1", fc_relu(25088, 4096)),
                ("fc2", fc_relu(4096, 4096))
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
    def __init__(self=True):
        super(VGG13, self).__init__()
        self.conv = conv_layer(2, 2, 2, 2, 2)
        self.fc = Sequential(
            OrderedDict([
                ("fc1", fc_relu(25088, 4096)),
                ("fc2", fc_relu(4096, 4096))
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
    def __init__(self=True):
        super(VGG16, self).__init__()
        self.conv = conv_layer(2, 2, 3, 3, 3)
        self.fc = Sequential(
            OrderedDict([
                ("fc1", fc_relu(25088, 4096)),
                ("fc2", fc_relu(4096, 4096))
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
    def __init__(self=True):
        super(VGG19, self).__init__()
        self.conv = conv_layer(2, 2, 4, 4, 4)
        self.fc = Sequential(
            OrderedDict([
                ("fc1", fc_relu(25088, 4096)),
                ("fc2", fc_relu(4096, 4096))
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


def conv_layer(num1, num2, num3, num4, num5):
    return Sequential(
        OrderedDict([
            ("conv0", _Conv(3, 64, num1)),
            ("conv1", _Conv(64, 128, num2)),
            ("conv2", _Conv(128, 256, num3)),
            ("conv3", _Conv(256, 512, num4)),
            ("conv4", _Conv(512, 512, num5))
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

    model0 = VGG11()
    model1 = VGG16()
    model2 = VGG13()
    model3 = VGG19()
    input = torch.randn((1, 3, 224, 224))
    macs0, params0 = profile(model0, inputs=(input, ))
    macs1, params1 = profile(model1, inputs=(input, ))
    macs2, params2 = profile(model2, inputs=(input, ))
    macs3, params3 = profile(model3, inputs=(input, ))
    summary(model0, (3, 224, 224))
    summary(model1, (3, 224, 224))
    summary(model2, (3, 224, 224))
    summary(model3, (3, 224, 224))

    print("MACs: {}".format(macs0))
    print("MACs: {}".format(macs1))
    print("MACs: {}".format(macs2))
    print("MACs: {}".format(macs3))
