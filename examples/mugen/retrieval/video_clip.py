# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from examples.mugen.retrieval.s3d import S3D
from torch import nn


class VideoEncoder(nn.Module):
    """Encode videos to a fixed size vector. Adapted from VideoCLIP
        (https://github.com/facebookresearch/fairseq/blob/main/examples/MMPT/mmpt/processors/models/s3dg.py)

    Args:
        trainable (bool): true if model weights should be trained
        preprocess_mean (list): sequence of means to normalize each channel to
        preprocess_std (list): sequence of standard deviations to normalize each channel to

    Inputs:
        x (Tensor): batch of videos with dimensions (batch, channel, time, height, width)
    """

    def __init__(self):
        super().__init__()
        self.model = S3D(400)
        self.embedding_dim = list(self.model.fc.children())[0].in_channels
        self.model.fc = nn.Identity()

    def forward(self, x):
        return self.model(x)
