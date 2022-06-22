# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn.functional as F
from torchmultimodal.utils.common import shift_dim
from torchvision.transforms.functional import normalize, resize


MUGEN_DEFAULT_TIME_SAMPLES = 32
DEFAULT_NORMALIZE_MEAN = [0.43216, 0.394666, 0.37645]
DEFAULT_NORMALIZE_STD = [0.22803, 0.22145, 0.216989]
DEFAULT_RESIZE_SHAPE = (224, 224)


class VideoTransform:
    """Transform videos for encoding

    Args:
        time_samples (int): number of frames to sample in the time dimension
        normalize_mean (list): sequence of means to normalize each channel to
        normalize_std (list): sequence of standard deviations to normalize each channel to
        resize_shape (tuple): shape to resize each frame to

    Inputs:
        video (Tensor): batch of videos with dimensions (batch, time, height, width, channel)

    """

    def __init__(
        self,
        time_samples=MUGEN_DEFAULT_TIME_SAMPLES,
        normalize_mean=DEFAULT_NORMALIZE_MEAN,
        normalize_std=DEFAULT_NORMALIZE_STD,
        resize_shape=DEFAULT_RESIZE_SHAPE,
    ):
        self.time_samples = time_samples
        self.normalize_mean = normalize_mean
        self.normalize_std = normalize_std
        self.resize_shape = resize_shape

    def __call__(self, video: torch.Tensor) -> torch.Tensor:
        b, t, h, w, c = video.shape
        if t != self.time_samples:
            video = F.interpolate(
                shift_dim(video, -1, 1), size=[self.time_samples, h, w]
            )  # "b t h w c -> b c t h w"
            video = shift_dim(video, 1, -1)  # "b c t h w -> b t h w c"
        assert c == 3
        video = video.flatten(start_dim=0, end_dim=1)  # "b t h w c -> (b t) h w c"
        video = shift_dim(video, -1, 1)  # "(b t) h w c -> (b t) c h w"
        video = (
            resize(video, self.resize_shape) if (h, w) != self.resize_shape else video
        )
        # normalize rgb video
        video = video.float() / 255.0
        video = normalize(video, mean=self.normalize_mean, std=self.normalize_std)
        # convert to BCTHW
        video = video.unflatten(
            dim=0, sizes=(b, self.time_samples)
        )  # "(b t) c h w -> b t c h w"
        video = shift_dim(video, 2, 1)  # "b t c h w -> b c t h w"
        return video
