# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from functools import partial
from typing import List, Tuple, Union

import torch
import torchvision.transforms as tv
from PIL.Image import Image

from torch import nn, Tensor
from torchmultimodal.modules.diffusion.schedules import DiffusionSchedule
from torchmultimodal.utils.diffusion_utils import cascaded_resize


def normalize(x: Tensor, image_min: int, image_max: int) -> Tensor:
    # Normalize image values between min and max
    return (image_max - image_min) * x + image_min


class Dalle2ImageTransform(nn.Module):
    """Dalle image transform normalizes the data between a min and max value. Defaults
    are -1 and 1 like the Dalle2 Paper.

    Args:
        image_size (int): desired output image size.
        image_min (float): min of images, used for normalization.
        image_max (float): max of images, used for normalization.

    Inputs:
        image (Union[List[Image], Image]): Image or batch of images upon which
            to apply the transform.
    """

    def __init__(
        self,
        image_size: int = 64,
        image_min: float = -1.0,
        image_max: float = 1.0,
    ) -> None:
        super().__init__()
        self.image_transform = tv.Compose(
            [
                partial(cascaded_resize, resolution=image_size),
                tv.CenterCrop(image_size),
                tv.ToTensor(),
                partial(normalize, image_min=image_min, image_max=image_max),
            ]
        )

    def forward(self, image: Union[List[Image], Image]) -> Tensor:
        if isinstance(image, Image):
            image = [image]
        # pyre-ignore
        image_result = torch.stack([self.image_transform(x) for x in image])
        return image_result


class RandomDiffusionSteps(nn.Module):
    """Data Transform to randomly sample noised data from the diffusion schedule.
    During diffusion training, random diffusion steps are sampled per model update.
    This transform samples steps and returns the steps (t), seed noise, and transformed
    data at time t (xt).

    Attributes:
        schedule (DiffusionSchedule): defines diffusion of noise through time
        batched (bool): if True, transform expects a batched input

    Args:
        x (Tensor): data representing x0, artifact being learned. The 0 represents zero diffusion steps.
    """

    def __init__(self, schedule: DiffusionSchedule, batched: bool = True):
        super().__init__()
        self.schedule = schedule
        self.batched = batched

    def __call__(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        if not self.batched:
            t = self.schedule.sample_steps(x.unsqueeze(0))
            t = t.squeeze(0)
        else:
            t = self.schedule.sample_steps(x)
        noise = self.schedule.sample_noise(x)
        xt = self.schedule.q_sample(x, noise, t)
        return x, xt, noise, t
