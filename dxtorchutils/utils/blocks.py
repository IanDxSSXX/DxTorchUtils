from dxtorchutils.utils.layers import *
from torch.nn import functional as F


class SeBlock(Module):
    def __init__(self, channels, reduction=16):
        super(SeBlock, self).__init__()

        self.channels = channels
        mid_channel = channels // reduction if channels // reduction > 0 else 1

        self.pool = AdaptiveAvgPool2d(1)
        self.fc0 = fc_relu(channels, mid_channel, False)
        self.fc1 = fc_sigmoid(mid_channel, channels, False)

    def forward(self, input):
        batch_size, channels, _, _ = input.shape
        assert channels == self.channels, "Channel mismatch"

        x = self.pool(input)
        x = x.view(batch_size, channels)
        x = self.fc0(x)
        x = self.fc1(x)
        attention = x.view(batch_size, channels, 1, 1)

        output = input * attention

        return output


class NonLocalBlock(Module):
    def __init__(self, in_channels, inter_channels=None, sub_sample=True):
        super(NonLocalBlock, self).__init__()

        self.sub_sample = sub_sample

        if inter_channels is None:
            inter_channels = in_channels // 2 if in_channels // 2 != 0 else 1

        self.inter_channels = inter_channels

        self.g = Conv2d(in_channels, inter_channels, 1)
        self.phi = Conv2d(in_channels, inter_channels, 1)
        self.theta = Conv2d(in_channels, inter_channels, 1)

        self.w = Sequential(
            OrderedDict([
                ("W", Conv2d(inter_channels, in_channels, 1, 1)),
                ("normalization", BatchNorm2d(in_channels))
            ])
        )

        self.sub_sample = sub_sample

        if sub_sample:
            self.g = Sequential(
                OrderedDict([
                    ("G", self.g),
                    ("pool", MaxPool2d(2))
                ])
            )
            self.phi = Sequential(
                OrderedDict([
                    ("Phi", self.phi),
                    ("pool", MaxPool2d(2))
                ])
            )
            self.pool = MaxPool2d(2)

    def forward(self, input):
        batch_size, _, h, w = input.shape

        x_g = self.g(input).view(batch_size, self.inter_channels, -1)
        x_g = x_g.permute(0, 2, 1)

        x_theta = self.theta(input).view(batch_size, self.inter_channels, -1)
        x_theta = x_theta.permute(0, 2, 1)

        x_phi = self.phi(input).view(batch_size, self.inter_channels, -1)

        x_f = torch.matmul(x_theta, x_phi)
        x_f = F.softmax(x_f, dim=-1)

        x = torch.matmul(x_f, x_g)
        x = x.permute(0, 2, 1)
        x = x.view(batch_size, -1, h, w)
        x_w = self.w(x)

        output = x_w + input

        return output
