""" AlexNet """

import torch
import torch.nn as nn


class AlexNet(nn.Module):
    def __init__(self, num_classes=0):
        super(AlexNet, self).__init__()
        self.conv1 = _conv(3, 96, 11, 4, 2, True)
        self.pool = nn.MaxPool2d(3, 2)
        self.conv2 = _conv(96, 256, 5, 1, 2, True)
        self.conv3 = _conv(256, 384, 3, 1, 1)
        self.conv4 = _conv(384, 384, 3, 1, 1)
        self.conv5 = _conv(384, 256, 3, 1, 1)
        self.fc1 = _fc(9216, 4096)
        self.fc2 = _fc(4096, 4096)
        self.fc3 = nn.Linear(4096, 1000)

        self.num_classes = num_classes
        self.out = nn.Linear(1000, num_classes)

    def forward(self, x):
        # (n, 3, 224, 224)
        x = self.conv1(x)
        # (n, 96, 55, 55)
        x = self.pool(x)
        # (n, 96, 27, 27)
        x = self.conv2(x)
        # (n, 256, 27, 27)
        x = self.pool(x)
        # (n, 256, 13, 13)
        x = self.conv3(x)
        # (n, 384, 13, 13)
        x = self.conv4(x)
        # (n, 384, 13, 13)
        x = self.conv5(x)
        # (n, 256, 13, 13)
        x = self.pool(x)
        # (n, 256, 6, 6)
        x = torch.flatten(x, 1)
        # (n, 9216)
        x = self.fc1(x)
        # (n, 4096)
        x = self.fc2(x)
        # (n, 4096)
        x = self.fc3(x)
        # (n, 1000)
        if self.num_classes != 0:
            x = self.out(x)
            # (n, num_classes)

        return x


def _conv(in_channel, out_channel, kernel_size, stride, padding, LRN=False):
    if LRN:
        return nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size, stride, padding),
            nn.Tanh(),
            nn.LocalResponseNorm(5)
        )
    else:
        return nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size, stride, padding),
            nn.Tanh()
        )


def _fc(in_features, out_features):
    return nn.Sequential(
        nn.Linear(in_features, out_features),
        nn.Tanh(),
        nn.Dropout()
    )