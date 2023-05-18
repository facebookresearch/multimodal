# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from typing import Optional

import PIL
import torch
from PIL.Image import Image
from torch import Tensor


@dataclass
class DiffusionOutput:
    prediction: Tensor
    variance_value: Optional[Tensor] = None
    mean: Optional[Tensor] = None
    log_variance: Optional[Tensor] = None


def cascaded_resize(pil_image: Image, resolution: int) -> Image:
    """Cascaded resizing
    Based on D2Go: https://fburl.com/nxlme9rj, resize by powers of 2 to nearest power of 2
    for improved BICUBIC performance.

    Args:
        image (Image): PIL image
        resolution (int): image minimum size target
    """
    while min(*pil_image.size) >= 2 * resolution:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=PIL.Image.BOX
        )
    scale = resolution / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size),
        resample=PIL.Image.BICUBIC,
    )

    return pil_image


def denormalize_to_0_1(images: Tensor) -> Tensor:
    """Denormalize tensors from range [-1, 1] to [0, 1]"""
    denormed_images = torch.clamp((images + 1) / 2, 0, 1)

    return denormed_images
