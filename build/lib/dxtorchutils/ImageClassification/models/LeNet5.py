"""LeNet 5"""
import torch
from torch.nn import *


class LeNet5(Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.conv0 = _Conv(1, 6)
        self.conv1 = _Conv(6, 16)
        self.conv2 = _Conv(16, 120)
        self.pool0 = AvgPool2d(2)
        self.pool1 = AvgPool2d(2)
        self.fc = _FC(120, 84)
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



class _Conv(Module):
    def __init__(self, in_channels, out_channels):
        super(_Conv, self).__init__()
        self.conv = Conv2d(in_channels, out_channels, 5)

    def forward(self, input):
        x = self.conv(input)
        output = torch.tanh(x)

        return output


class _FC(Module):
    def __init__(self, in_features, out_features):
        super(_FC, self).__init__()
        self.fc = Linear(in_features, out_features)

    def forward(self, input):
        x = self.fc(input)
        output = torch.tanh(x)

        return output
