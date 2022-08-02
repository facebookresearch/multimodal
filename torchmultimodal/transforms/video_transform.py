# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Tuple

import torch
import torch.nn.functional as F
from torchmultimodal.utils.common import shift_dim
from torchvision.transforms.functional import normalize, resize


MUGEN_DEFAULT_TIME_SAMPLES = 32
DEFAULT_MEAN = (0.43216, 0.394666, 0.37645)
DEFAULT_STD = (0.22803, 0.22145, 0.216989)
DEFAULT_RESIZE_SHAPE = (224, 224)


class VideoTransform:
    """Transform videos for encoding

    Args:
        time_samples (int): number of frames to sample in the time dimension
        mean (Tuple[float, float, float]): sequence of means of each channel
        std (Tuple[float, float, float]): sequence of standard deviations of each channel
        resize_shape (Tuple[int, int]): shape to resize each frame to

    Inputs:
        video (Tensor): batch of videos with dimensions (batch, time, height, width, channel)

    Returns:
        Tensor: processed batch of videos with dimensions
            (batch, channel, time_samples, resize_shape[0], resize_shape[1])

    """

    def __init__(
        self,
        time_samples: int = MUGEN_DEFAULT_TIME_SAMPLES,
        mean: Tuple[float, float, float] = DEFAULT_MEAN,
        std: Tuple[float, float, float] = DEFAULT_STD,
        resize_shape: Tuple[int, int] = DEFAULT_RESIZE_SHAPE,
    ):
        self.time_samples = time_samples
        self.mean = mean
        self.std = std
        self.resize_shape = resize_shape

    def __call__(self, video: torch.Tensor) -> torch.Tensor:
        if video.shape[-1] != 3:
            raise ValueError("Video must have 3 channels")

        video = self.sample_frames(video)
        video = self.resize_hw(video)
        video = self.normalize(video)
        video = shift_dim(video, -1, 1)
        return video

    def sample_frames(self, video: torch.Tensor) -> torch.Tensor:
        """Samples frames from video of dims (b, t, h, w, c)"""
        _, t, h, w, _ = video.shape
        if t != self.time_samples:
            video = F.interpolate(
                shift_dim(video, -1, 1), size=[self.time_samples, h, w]
            )  # "b t h w c -> b c t h w"
            video = shift_dim(video, 1, -1)  # "b c t h w -> b t h w c"
        return video

    def resize_hw(self, video: torch.Tensor) -> torch.Tensor:
        """Resizes height and width of video of dims (b, t, h, w, c)"""
        b, t, h, w, _ = video.shape
        video = video.flatten(start_dim=0, end_dim=1)  # "b t h w c -> (b t) h w c"
        video = shift_dim(video, -1, 1)  # "(b t) h w c -> (b t) c h w"

        video = (
            resize(video, self.resize_shape) if (h, w) != self.resize_shape else video
        )

        video = video.unflatten(dim=0, sizes=(b, t))  # "(b t) c h w -> b t c h w"
        video = shift_dim(video, 2, -1)  # "b t c h w -> b t h w c"
        return video

    def normalize(self, video: torch.Tensor) -> torch.Tensor:
        """Normalizes video of dims (b, t, h, w, c) to mean 0, std 1"""
        b, t, _, _, _ = video.shape
        video = video.flatten(start_dim=0, end_dim=1)  # "b t h w c -> (b t) h w c"
        video = shift_dim(video, -1, 1)  # "(b t) h w c -> (b t) c h w"

        video = video.float() / 255.0
        video = normalize(video, mean=self.mean, std=self.std)

        video = video.unflatten(dim=0, sizes=(b, t))  # "(b t) c h w -> b t c h w"
        video = shift_dim(video, 2, -1)  # "b t c h w -> b t h w c"
        return video
