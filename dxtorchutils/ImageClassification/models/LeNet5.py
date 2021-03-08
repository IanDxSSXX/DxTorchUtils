"""
LeNet 5
    Input: (1, 32, 32)
    Total params: 61,706
    Trainable params: 61,706
    Non-trainable params: 0

    Input size (MB): 0.00
    Forward/backward pass size (MB): 0.11
    Params size (MB): 0.24
    Estimated Total Size (MB): 0.35

    MACs/FLOPs: 424,520
"""
from dxtorchutils.utils.layers import *


class LeNet5(Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.conv0 = conv_tanh(1, 6, 5)
        self.conv1 = conv_tanh(6, 16, 5)
        self.conv2 = conv_tanh(16, 120, 5)
        self.pool0 = AvgPool2d(2)
        self.pool1 = AvgPool2d(2)
        self.fc = fc_tanh(120, 84)
        self.out = Linear(84, 10)

    def forward(self, x):
        # (n, 1, 32, 32)
        x = self.conv0(x)
        # (n, 6, 28, 28)
        x = self.pool0(x)
        # (n, 6, 14, 14)
        x = self.conv1(x)
        # (n, 16, 10, 10)
        x = self.pool1(x)
        # (n, 16, 5, 5)
        x = self.conv2(x)
        # (n, 120, 1, 1)
        x = x.view(x.shape[0], -1)
        # (n, 120)
        x = self.fc(x)
        # (n, 84)
        output = self.out(x)
        # (n, 10)

        return output


if __name__ == '__main__':
    # calculate parameters
    from thop import profile
    from torchsummary import summary

    model = LeNet5()
    input = torch.randn((1, 1, 32, 32))
    macs, params = profile(model, inputs=(input, ))
    summary(model, (1, 32, 32))

    print("MACs: {}".format(macs))
