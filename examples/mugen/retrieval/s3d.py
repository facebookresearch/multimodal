# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn.functional as F
from torch import nn


class S3D(nn.Module):
    """S3D is a video classification model that improves over I3D in speed by replacing
        3D convolutions with a spatial 2D convolution followed by a temporal 1D convolution.
        Paper: https://arxiv.org/abs/1712.04851
        Code: https://github.com/kylemin/S3D

    Args:
        num_class (int): number of classes for the classification task

    Inputs:
        x (Tensor): batch of videos with dimensions (batch, channel, time, height, width)
    """

    def __init__(self, num_class):
        super(S3D, self).__init__()
        self.base = nn.Sequential(
            SepConv3d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),
            BasicConv3d(64, 64, kernel_size=1, stride=1),
            SepConv3d(64, 192, kernel_size=3, stride=1, padding=1),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),
            MixedConvsBlock(192, 64, 96, 128, 16, 32, 32),
            MixedConvsBlock(256, 128, 128, 192, 32, 96, 64),
            nn.MaxPool3d(kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1)),
            MixedConvsBlock(480, 192, 96, 208, 16, 48, 64),
            MixedConvsBlock(512, 160, 112, 224, 24, 64, 64),
            MixedConvsBlock(512, 128, 128, 256, 24, 64, 64),
            MixedConvsBlock(512, 112, 144, 288, 32, 64, 64),
            MixedConvsBlock(528, 256, 160, 320, 32, 128, 128),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=(0, 0, 0)),
            MixedConvsBlock(832, 256, 160, 320, 32, 128, 128),
            MixedConvsBlock(832, 384, 192, 384, 48, 128, 128),
        )
        self.fc = nn.Conv3d(1024, num_class, kernel_size=1, stride=1, bias=True)

    def forward(self, x):
        y = self.base(x)
        y = F.avg_pool3d(y, (2, y.size(3), y.size(4)), stride=1)
        y = self.fc(y)
        y = y.view(y.size(0), y.size(1), y.size(2))
        logits = torch.mean(y, 2)

        return logits


class BasicConv3d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride, padding=0):
        super(BasicConv3d, self).__init__()
        self.conv = nn.Conv3d(
            in_planes,
            out_planes,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
        )
        self.bn = nn.BatchNorm3d(out_planes, eps=1e-3, momentum=0.001, affine=True)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class SepConv3d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride, padding=0):
        super(SepConv3d, self).__init__()
        self.conv_s = nn.Conv3d(
            in_planes,
            out_planes,
            kernel_size=(1, kernel_size, kernel_size),
            stride=(1, stride, stride),
            padding=(0, padding, padding),
            bias=False,
        )
        self.bn_s = nn.BatchNorm3d(out_planes, eps=1e-3, momentum=0.001, affine=True)
        self.relu_s = nn.ReLU()

        self.conv_t = nn.Conv3d(
            out_planes,
            out_planes,
            kernel_size=(kernel_size, 1, 1),
            stride=(stride, 1, 1),
            padding=(padding, 0, 0),
            bias=False,
        )
        self.bn_t = nn.BatchNorm3d(out_planes, eps=1e-3, momentum=0.001, affine=True)
        self.relu_t = nn.ReLU()

    def forward(self, x):
        x = self.conv_s(x)
        x = self.bn_s(x)
        x = self.relu_s(x)

        x = self.conv_t(x)
        x = self.bn_t(x)
        x = self.relu_t(x)
        return x


class Branch0(nn.Sequential):
    def __init__(self, in_planes, out_planes):
        super(Branch0, self).__init__(
            BasicConv3d(in_planes, out_planes, kernel_size=1, stride=1)
        )


class Branch1or2(nn.Sequential):
    def __init__(self, in_planes, mid_planes, out_planes):
        super(Branch1or2, self).__init__(
            BasicConv3d(in_planes, mid_planes, kernel_size=1, stride=1),
            SepConv3d(mid_planes, out_planes, kernel_size=3, stride=1, padding=1),
        )


class Branch3(nn.Sequential):
    def __init__(self, in_planes, out_planes):
        super(Branch3, self).__init__(
            nn.MaxPool3d(kernel_size=(3, 3, 3), stride=1, padding=1),
            BasicConv3d(in_planes, out_planes, kernel_size=1, stride=1),
        )


class MixedConvsBlock(nn.Module):
    def __init__(self, in_planes, b0_out, b1_mid, b1_out, b2_mid, b2_out, b3_out):
        super().__init__()
        self.branch0 = Branch0(in_planes, b0_out)
        self.branch1 = Branch1or2(in_planes, b1_mid, b1_out)
        self.branch2 = Branch1or2(in_planes, b2_mid, b2_out)
        self.branch3 = Branch3(in_planes, b3_out)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = torch.cat((x0, x1, x2, x3), 1)

        return out
