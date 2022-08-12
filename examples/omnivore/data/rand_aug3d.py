# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import math
from typing import List, Optional

import torch
from torch import Tensor
from torchvision.transforms import autoaugment, functional as F, InterpolationMode

__all__ = ["RandAugment3d"]


def _apply_op(
    img: Tensor,
    op_name: str,
    magnitude: float,
    interpolation: InterpolationMode,
    fill: Optional[List[float]],
):
    # Modified from torchvision.transforms.autoaugment._apply_op
    # we assume the input img has type float and in range 0 to 1
    if op_name == "ShearX":
        # magnitude should be arctan(magnitude)
        # official autoaug: (1, level, 0, 0, 1, 0)
        # https://github.com/tensorflow/models/blob/dd02069717128186b88afa8d857ce57d17957f03/research/autoaugment/augmentation_transforms.py#L290
        # compared to
        # torchvision:      (1, tan(level), 0, 0, 1, 0)
        # https://github.com/pytorch/vision/blob/0c2373d0bba3499e95776e7936e207d8a1676e65/torchvision/transforms/functional.py#L976
        img = F.affine(
            img,
            angle=0.0,
            translate=[0, 0],
            scale=1.0,
            shear=[math.degrees(math.atan(magnitude)), 0.0],
            interpolation=interpolation,
            fill=fill,
            center=[0, 0],
        )
    elif op_name == "ShearY":
        # magnitude should be arctan(magnitude)
        # See above
        img = F.affine(
            img,
            angle=0.0,
            translate=[0, 0],
            scale=1.0,
            shear=[0.0, math.degrees(math.atan(magnitude))],
            interpolation=interpolation,
            fill=fill,
            center=[0, 0],
        )
    elif op_name == "TranslateX":
        img = F.affine(
            img,
            angle=0.0,
            translate=[int(magnitude), 0],
            scale=1.0,
            interpolation=interpolation,
            shear=[0.0, 0.0],
            fill=fill,
        )
    elif op_name == "TranslateY":
        img = F.affine(
            img,
            angle=0.0,
            translate=[0, int(magnitude)],
            scale=1.0,
            interpolation=interpolation,
            shear=[0.0, 0.0],
            fill=fill,
        )
    elif op_name == "Rotate":
        img = F.rotate(img, magnitude, interpolation=interpolation, fill=fill)
    elif op_name == "Brightness":
        img = F.adjust_brightness(img, 1.0 + magnitude)
    elif op_name == "Color":
        img = F.adjust_saturation(img, 1.0 + magnitude)
    elif op_name == "Contrast":
        img = F.adjust_contrast(img, 1.0 + magnitude)
    elif op_name == "Sharpness":
        img = F.adjust_sharpness(img, 1.0 + magnitude)
    elif op_name == "Posterize":
        # The tensor dtype must be torch.uint8
        # and values are expected to be in [0, 255]
        img = (img * 255.9999).to(dtype=torch.uint8)
        img = F.posterize(img, int(magnitude))
        img = (img / 255.9999).to(dtype=torch.float32)
    elif op_name == "Solarize":
        # The tensor dtype must be torch.uint8
        # and values are expected to be in [0, 255]
        img = (img * 255.9999).to(dtype=torch.uint8)
        img = F.solarize(img, int(magnitude))
        img = (img / 255.9999).to(dtype=torch.float32)
    elif op_name == "AutoContrast":
        img = F.autocontrast(img)
    elif op_name == "Equalize":
        # The tensor dtype must be torch.uint8
        # and values are expected to be in [0, 255]
        img = (img * 255.9999).to(dtype=torch.uint8)
        img = F.equalize(img)
        img = (img / 255.9999).to(dtype=torch.float32)
    elif op_name == "Invert":
        img = F.invert(img)
    elif op_name == "Identity":
        pass
    else:
        raise ValueError(f"The provided operator {op_name} is not recognized.")
    return img


class RandAugment3d(autoaugment.RandAugment):
    """Modified RandAugment in order to handle single-view depth image.
    In here, color / non-geometric operation will only be applied on RGB channel.

    Args:
        num_ops (int): Number of augmentation transformations to apply sequentially.
        magnitude (int): Magnitude for all the transformations.
        num_magnitude_bins (int): The number of different magnitude values.
        interpolation (InterpolationMode): Desired interpolation enum defined by
            :class:`torchvision.transforms.InterpolationMode`. Default is ``InterpolationMode.NEAREST``.
            If input is Tensor, only ``InterpolationMode.NEAREST``, ``InterpolationMode.BILINEAR`` are supported.
        fill (sequence or number, optional): Pixel fill value for the area outside the transformed
            image. If given a number, the value is used for all bands respectively.
    """

    def __init__(
        self,
        num_ops: int = 2,
        magnitude: int = 9,
        num_magnitude_bins: int = 31,
        interpolation: InterpolationMode = InterpolationMode.NEAREST,
        fill: Optional[List[float]] = None,
    ) -> None:
        super().__init__(
            num_ops=num_ops,
            magnitude=magnitude,
            num_magnitude_bins=num_magnitude_bins,
            interpolation=interpolation,
            fill=fill,
        )
        self.geom_ops = {
            "Identity",
            "ShearX",
            "ShearY",
            "TranslateX",
            "TranslateY",
            "Rotate",
        }

    def forward(self, img: Tensor) -> Tensor:
        """
            img (PIL Image or Tensor): Image to be transformed.

        Returns:
            PIL Image or Tensor: Transformed image.
        """
        fill = self.fill
        channels, height, width = F.get_dimensions(img)
        if isinstance(img, Tensor):
            if isinstance(fill, (int, float)):
                fill = [float(fill)] * channels
            elif fill is not None:
                fill = [float(f) for f in fill]

        op_meta = self._augmentation_space(self.num_magnitude_bins, (height, width))
        for _ in range(self.num_ops):
            op_index = int(torch.randint(len(op_meta), (1,)).item())
            op_name = list(op_meta.keys())[op_index]
            magnitudes, signed = op_meta[op_name]
            magnitude = (
                float(magnitudes[self.magnitude].item()) if magnitudes.ndim > 0 else 0.0
            )
            if signed and torch.randint(2, (1,)):
                magnitude *= -1.0
            if op_name in self.geom_ops:
                # apply geometric operation on RGBD image
                img = _apply_op(
                    img, op_name, magnitude, interpolation=self.interpolation, fill=fill
                )
            else:
                # Apply non_geom operation on the RGB channels only
                img[:3, :, :] = _apply_op(
                    img[:3, :, :],
                    op_name,
                    magnitude,
                    interpolation=self.interpolation,
                    fill=fill,
                )
        return img
