# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from functools import partial
from typing import Any, Dict

import torch
import torchvision.transforms as tv
from PIL.Image import Image

from torch import nn
from torchmultimodal.diffusion_labs.utils.common import cascaded_resize, normalize


class Dalle2ImageTransform(nn.Module):
    """Dalle image transform normalizes the data between a min and max value. Defaults
    are -1 and 1 like the Dalle2 Paper.

    Args:
        image_size (int): desired output image size.
        image_min (float): min of images, used for normalization.
        image_max (float): max of images, used for normalization.
        image_field (str): key name for the image

    Inputs:
        image (Union[List[Image], Image]): Image or batch of images upon which
            to apply the transform.
    """

    def __init__(
        self,
        image_size: int = 64,
        image_min: float = -1.0,
        image_max: float = 1.0,
        image_field: str = "x",
    ) -> None:
        super().__init__()
        self.image = image_field
        self.image_transform = tv.Compose(
            [
                partial(cascaded_resize, resolution=image_size),
                tv.CenterCrop(image_size),
                tv.ToTensor(),
                partial(normalize, image_min=image_min, image_max=image_max),
            ]
        )

    def forward(self, x: Dict[str, Any]) -> Dict[str, Any]:
        assert self.image in x, f"{type(self).__name__} expects key {self.image}"
        image = x[self.image]
        if isinstance(image, Image):
            im = self.image_transform(image)
        else:
            # pyre-ignore
            im = torch.stack([self.image_transform(x) for x in image])
        x[self.image] = im
        return x
