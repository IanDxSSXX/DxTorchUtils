""" AlexNet """
from collections import OrderedDict
import torch
from torch.nn import *


class AlexNet(Module):
    def __init__(self):
        super(AlexNet, self).__init__()
        self.conv0 = _Conv(3, 96, 11, 4, 2, True)
        self.conv1 = _Conv(96, 256, 5, 1, 2, True)
        self.conv2 = _Conv(256, 384, 3, 1, 1)
        self.conv3 = _Conv(384, 384, 3, 1, 1)
        self.conv4 = _Conv(384, 256, 3, 1, 1)

        self.pool0 = MaxPool2d(3, 2)
        self.pool1 = MaxPool2d(3, 2)

        self.fc0 = _FC(9216, 4096)
        self.fc1 = _FC(4096, 4096)
        self.out = Linear(4096, 1000)



    def forward(self, x):
        # (n, 3, 224, 224)
        x = self.conv0(x)
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


class _Conv(Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, is_lrn=False):
        super(_Conv, self).__init__()
        self.conv = Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.lrn = LocalResponseNorm(5)
        self.is_lrn = is_lrn

    def forward(self, input):
        x = self.conv(input)
        x = torch.tanh(x)
        if self.is_lrn:
            output = self.lrn(x)
        else:
            output = x

        return output


class _FC(Module):
    def __init__(self, in_features, out_features):
        super(_FC, self).__init__()
        self.fc = Linear(in_features, out_features)
        self.lrn = LocalResponseNorm(5)

    def forward(self, input):
        x = self.fc(input)
        x = torch.tanh(x)
        output = torch.dropout(x, 0.5, self.training)

        return output
