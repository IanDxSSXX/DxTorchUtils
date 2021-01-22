"""LeNet 5"""

import torch
import torch.nn as nn


class LeNet5(nn.Module):
    def __init__(self, num_classes=0):
        super(LeNet5, self).__init__()
        self.conv1 = _conv(1, 6)
        self.conv2 = _conv(6, 16)
        self.conv3 = _conv(16, 120)
        self.pool = nn.AvgPool2d(2)
        self.fc1 = _fc(120, 84)
        self.fc2 = nn.Linear(84, 10)

        self.num_classes = num_classes
        self.out = nn.Linear(10, num_classes)

    def forward(self, x):
        # (n, 1, 32, 32)
        x = self.conv1(x)
        # (n, 6, 28, 28)
        x = self.pool(x)
        # (n, 6, 14, 14)
        x = self.conv2(x)
        # (n, 16, 10, 10)
        x = self.pool(x)
        # (n, 16, 5, 5)
        x = self.conv3(x)
        # (n, 120, 1, 1)
        x = torch.flatten(x, 1)
        # (n, 120)
        x = self.fc1(x)
        # (n, 84)
        x = self.fc2(x)
        # (n, 10)
        if self.num_classes != 0:
            x = self.out(x)
            # (n, num_classes)
        return x


def _conv(in_channel, out_channel):
    return nn.Sequential(
        nn.Conv2d(in_channel, out_channel, 5),
        nn.Tanh()
    )


def _fc(in_features, out_features):
    return nn.Sequential(
        nn.Linear(in_features, out_features),
        nn.Tanh()
    )
