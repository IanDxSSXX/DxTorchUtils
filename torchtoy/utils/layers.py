from torch.nn import *
from collections import OrderedDict
import torch


def conv_tanh(in_channels, out_channels, kernel_size, stride=1, padding=0):
    return Sequential(
        OrderedDict([
            ("conv", Conv2d(in_channels, out_channels, kernel_size, stride, padding)),
            ("activation", Tanh()),
        ])
    )


def conv_tanh_lrn(in_channels, out_channels, kernel_size, stride=1, padding=0):
    return Sequential(
        OrderedDict([
            ("conv", Conv2d(in_channels, out_channels, kernel_size, stride, padding)),
            ("activation", Tanh()),
            ("normalization", LocalResponseNorm(5)),
        ])
    )


def fc_tanh_do(in_features, out_features, bias=False):
    return Sequential(
        OrderedDict([
            ("fc", Linear(in_features, out_features, bias)),
            ("activation", Tanh()),
            ("dropout", Dropout())
        ])
    )


def conv_relu(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1):
    return Sequential(
        OrderedDict([
            ("conv", Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation)),
            ("activation", ReLU(True))
        ])
    )


def fc_relu_fc(in_features, mid_features, out_features):
    return Sequential(
        OrderedDict([
            ("fc0", Linear(in_features, mid_features)),
            ("activation", ReLU(True)),
            ("fc1", Linear(mid_features, out_features))
        ])
    )


def fc_tanh(in_features, out_features, bias=True):
    return Sequential(
        OrderedDict([
            ("fc", Linear(in_features, out_features, bias)),
            ("activation", Tanh()),
        ])
    )


def fc_sigmoid(in_features, out_features, bias=True):
    return Sequential(
        OrderedDict([
            ("fc", Linear(in_features, out_features, bias)),
            ("activation", Sigmoid()),
        ])
    )


def conv_relu_bn(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1):
    return Sequential(
        OrderedDict([
            ("conv", Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation)),
            ("activation", ReLU(True)),
            ("normalization", BatchNorm2d(out_channels))
        ])
    )


def fc_relu(in_features, out_features, bias=False):
    return Sequential(
        OrderedDict([
            ("fc", Linear(in_features, out_features, bias)),
            ("activation", ReLU(True)),
        ])
    )


def mobilev1_bn_relu(in_channels, out_channels, kernel_size, stride=1, padding=0):
    return Sequential(
        OrderedDict([
            ("dwconv", Conv2d(in_channels, in_channels, kernel_size, stride, padding, 1, in_channels)),
            ("normalization0", BatchNorm2d(in_channels)),
            ("activation0", ReLU(True)),
            ("pwconv", Conv2d(in_channels, out_channels, 1)),
            ("normalization1", BatchNorm2d(out_channels)),
            ("activation1", ReLU(True))
        ])
    )


