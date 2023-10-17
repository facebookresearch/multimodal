# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import math
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

from PIL import Image, ImageDraw
from torch import nn, Tensor

MIN_IMAGE_SIZE = 64
MIN_BRUSH_STROKE = 20
MAX_BRUSH_STROKE = 35
MAX_BBOX = 48
MIN_NUM_VERTEX = 1
MAX_NUM_VERTEX = 3
MASK_VALUE = 1.0


@dataclass
class BBox:
    top: int
    left: int
    height: int
    width: int


def random_inpaint_mask_image(
    im: Tensor, vertical_margin: int = 0, horizontal_margin: int = 0
) -> Tensor:
    """
    Generate a random inpainting mask for an image.

    Args:
        im (Tensor): The input image tensor of shape (C, H, W), where C is the number of channels,
                     H is the image height, and W is the image width.
        vertical_margin (int): Vertical margin to exclude from the mask. Defaults to 0.
        horizontal_margin (int): Horizontal margin to exclude from the mask. Defaults to 0.

    Returns:
        Tensor: The generated inpainting mask tensor of shape (1, H, W), where H is the image height
                and W is the image width. The inpainting mask has a value of 1.0 within the masked region
                and 0.0 outside the masked region.

    """
    img_height, img_width = im.shape[-2], im.shape[-1]
    height = int(MAX_BBOX * img_height / MIN_IMAGE_SIZE)
    width = int(MAX_BBOX * img_width / MIN_IMAGE_SIZE)
    max_top = img_height - vertical_margin - height
    max_left = img_width - horizontal_margin - width
    top = int(torch.randint(low=vertical_margin, high=max_top, size=(1,)).item())
    left = int(torch.randint(low=horizontal_margin, high=max_left, size=(1,)).item())

    bbox = BBox(top=top, left=left, height=height, width=width)

    mask = torch.zeros(size=(1, img_height, img_width), dtype=im.dtype)
    h = int(torch.randint(low=0, high=height // 3 + 1, size=(1,)).item())
    w = int(torch.randint(low=0, high=width // 3 + 1, size=(1,)).item())
    mask[
        :,
        bbox.top + h : bbox.top + bbox.height - h,
        bbox.left + w : bbox.left + bbox.width - w,
    ] = MASK_VALUE
    return mask


def random_outpaint_mask_image(im: Tensor, min_delta: int = 0) -> Tensor:
    """
    Generates a random outpaint mask for an input image.

    Args:
        im (Tensor): The input image tensor of shape (C, H, W), where C is the number of channels,
                     H is the image height, and W is the image width.
        min_delta (int): The minimum size of the outpaint mask. Default is 0.

    Returns:
        Tensor: The generated outpaint mask tensor of shape (1, H, W), where H is the image height
                and W is the image width. The outpaint mask has a value of MASK_VALUE within the masked region
                and 0.0 outside the masked region.
    """
    img_height, img_width = im.shape[-2], im.shape[-1]

    bbox = BBox(top=0, left=0, height=img_height, width=img_width)

    side = int(torch.randint(low=0, high=4, size=(1,)).item())
    max_delta = img_height // 2 if side in [0, 1] else img_width // 2
    size = int(torch.randint(low=min_delta, high=max_delta, size=(1,)).item())
    if side == 0:
        bbox.top = img_height - size
    elif side == 1:
        bbox.height = size
    elif side == 2:
        bbox.left = img_width - size
    elif side == 3:
        bbox.width = size

    mask = torch.zeros(size=(1, img_height, img_width), dtype=im.dtype)
    mask[:, bbox.top : bbox.height, bbox.left : bbox.width] = MASK_VALUE
    return mask


def generate_vertexes(
    mask: Image.Image, num_vertexes: int, img_width: int, img_height: int
) -> List[Tuple[int, int]]:
    """
    Generates a list of vertexes based on the mask, number of vertexes, image width, and image height.

    Args:
        mask (Image.Image): The mask image.
        num_vertexes (int): Number of vertexes.
        img_width (int): Image width.
        img_height (int): Image height.

    Returns:
        List[Tuple[int, int]]: List of vertexes.
    """
    vertex = []
    vertex.append(
        (
            int(np.random.randint(img_width // 2, img_width - img_width // 4)),
            int(np.random.randint(img_height // 2, img_height - img_height // 4)),
        )
    )
    average_radius = math.sqrt(img_height * img_width + img_width * img_height) / 8
    angles = []
    mean_angle = 2 * math.pi / 2
    angle_range = 2 * math.pi / 8
    angle_min = mean_angle - np.random.uniform(0, angle_range)
    angle_max = mean_angle + np.random.uniform(0, angle_range)
    for i in range(num_vertexes):
        if i % 2 == 0:
            angles.append(2 * math.pi - np.random.uniform(angle_min, angle_max))
        else:
            angles.append(np.random.uniform(angle_min, angle_max))
    for i in range(num_vertexes):
        r = np.clip(
            np.random.normal(loc=average_radius, scale=average_radius // 2),
            0,
            2 * average_radius,
        )
        new_x = np.clip(vertex[-1][0] + r * math.cos(angles[i]), 0, img_width)
        new_y = np.clip(vertex[-1][1] + r * math.sin(angles[i]), 0, img_height)
        vertex.append((int(new_x), int(new_y)))
    return vertex


def draw_strokes(
    mask: Image.Image, vertexes: List[Tuple[int, int]], width: int
) -> None:
    """
    Draws the brush strokes on the mask using the provided vertexes and width.

    Args:
        mask (Image.Image): The mask image.
        vertexes (List[Tuple[int, int]]): List of vertexes.
        width (int): Width of the brush strokes.
    """
    # breakpoint()
    draw = ImageDraw.Draw(mask)
    draw.line(vertexes, fill=1, width=width)
    for v in vertexes:
        draw.ellipse(
            (
                v[0] - width // 2,
                v[1] - width // 2,
                v[0] + width // 2,
                v[1] + width // 2,
            ),
            fill=1,
        )


def brush_stroke_mask_image(im: Tensor) -> Tensor:
    """
    Generates a brush stroke mask for an input image.

    Args:
        im (Tensor): The input image tensor of shape (C, H, W), where C is the number of channels,
                     H is the image height, and W is the image width.

    Returns:
        Tensor: The generated brush stroke mask tensor of shape (1, H, W), where H is the image height
                and W is the image width. The brush stroke mask has a value of 1.0 within the brush stroke regions
                and 0.0 outside the brush stroke regions.
    """
    img_height, img_width = im.shape[-2], im.shape[-1]

    min_width = int(MIN_BRUSH_STROKE * img_width / MIN_IMAGE_SIZE)
    max_width = int(MAX_BRUSH_STROKE * img_width / MIN_IMAGE_SIZE)
    mask = Image.new("1", (img_width, img_height), 0)

    for _ in range(np.random.randint(1, 3)):
        num_vertexes = np.random.randint(MIN_NUM_VERTEX, MAX_NUM_VERTEX)
        vertex = generate_vertexes(mask, num_vertexes, img_width, img_height)
        width = int(np.random.uniform(min_width, max_width))
        draw_strokes(mask, vertex, width)

    if np.random.normal() > 0:
        mask.transpose(Image.FLIP_LEFT_RIGHT)
    if np.random.normal() > 0:
        mask.transpose(Image.FLIP_TOP_BOTTOM)

    mask = np.reshape(mask, (1, img_height, img_width))
    mask = torch.tensor(mask, dtype=im.dtype).permute(0, 1, 2)

    return mask


def mask_full_image(im: Tensor) -> Tensor:
    """
    Create a mask covering the entire image.

    Args:
        image (Tensor): Input image tensor.

    Returns:
        Tensor: Mask covering the entire image.

    """
    img_height, img_width = im.shape[-2], im.shape[-1]
    mask = torch.ones(size=(1, img_height, img_width), dtype=im.dtype)
    return mask


class RandomInpaintingMask(nn.Module):
    """Data transform to generate mask for training with inpainting. This approach
    is based on "Palette: Image-to-Image Diffusion Models" (https://arxiv.org/abs/2111.05826)
    and "GLIDE: Towards Photorealistic Image Generation and Editing with
    Text-Guided Diffusion Models" (https://arxiv.org/abs/2112.10741). A random mask is generated
    and concatenated with the original image and masked image.

    Attributes:
        prob_masking_threshold (float): Probability of fully masking each image.
        batched (bool): if True, transform expects a batched input
        data_field (str): key name for data
        mask_field (str): key name add mask

    Args:
        x (Dict): data containing tensor "x".

    """

    def __init__(
        self,
        prob_masking_threshold: float = 0.25,
        batched: bool = True,
        data_field: str = "x",
        mask_field: str = "mask",
    ):
        super().__init__()
        self.prob_masking_threshold = prob_masking_threshold
        self.batched = batched
        self.data = data_field
        self.mask = mask_field

    def _random_mask(self, image: Tensor) -> Tensor:
        prob_masking = random.random()
        if prob_masking < self.prob_masking_threshold:
            mask = mask_full_image(image)
        else:
            chosen_mask = random.randint(0, 3)
            if chosen_mask == 0:
                mask = random_inpaint_mask_image(image)
            elif chosen_mask == 1:
                mask = brush_stroke_mask_image(image)
            else:
                mask = random_outpaint_mask_image(image)
        return 1 - mask.to(image.dtype)

    def forward(self, x: Dict[str, Any]) -> Dict[str, Any]:
        assert self.data in x, f"{type(self).__name__} expects key {self.data}"
        data = x[self.data]
        if self.batched:
            x[self.mask] = torch.stack([self._random_mask(i) for i in data])
        else:
            x[self.mask] = self._random_mask(data)
        return x
