# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from functools import partial
from typing import List, Union

import torch
import torchvision.transforms as tv
from PIL.Image import Image

from torch import nn, Tensor
from torchmultimodal.diffusion_labs.utils.common import cascaded_resize, normalize


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
