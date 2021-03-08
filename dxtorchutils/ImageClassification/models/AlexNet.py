"""
AlexNet
    Input: (1, 3, 224, 224)
    Total params: 62,378,344
    Trainable params: 62,378,344
    Non-trainable params: 0

    Input size (MB): 0.57
    Forward/backward pass size (MB): 14.69
    Params size (MB): 237.95
    Estimated Total Size (MB): 253.22

    MACs/FLOPs: 1,135,906,176
"""
from dxtorchutils.utils.layers import *


class AlexNet(Module):
    def __init__(self):
        super(AlexNet, self).__init__()
        self.conv0 = conv_tanh_lrn(3, 96, 11, 4, 2)
        self.conv1 = conv_tanh_lrn(96, 256, 5, 1, 2)
        self.conv2 = conv_tanh(256, 384, 3, 1, 1)
        self.conv3 = conv_tanh(384, 384, 3, 1, 1)
        self.conv4 = conv_tanh(384, 256, 3, 1, 1)

        self.pool0 = MaxPool2d(3, 2)
        self.pool1 = MaxPool2d(3, 2)

        self.fc0 = fc_tanh_do(9216, 4096)
        self.fc1 = fc_tanh_do(4096, 4096)

        self.out = Linear(4096, 1000)


    def forward(self, input):
        # (n, 3, 224, 224)
        x = self.conv0(input)
        # (n, 96, 55, 55)
        x = self.pool0(x)
        # (n, 96, 27, 27)
        x = self.conv1(x)
        # (n, 256, 27, 27)
        x = self.pool1(x)
        # (n, 256, 13, 13)
        x = self.conv2(x)
        # (n, 384, 13, 13)
        x = self.conv3(x)
        # (n, 384, 13, 13)
        x = self.conv4(x)
        # (n, 256, 13, 13)
        x = self.pool1(x)
        # (n, 256, 6, 6)
        x = x.view(x.shape[0], -1)
        # (n, 9216)
        x = self.fc0(x)
        # (n, 4096)
        x = self.fc1(x)
        # (n, 4096)
        output = self.out(x)
        # (n, 1000)

        return output


if __name__ == '__main__':
    # calculate parameters
    from thop import profile
    from torchsummary import summary

    model = AlexNet()
    input = torch.randn((1, 3, 224, 224))
    macs, params = profile(model, inputs=(input, ))
    summary(model, (3, 224, 224))

    print("MACs: {}".format(macs))
