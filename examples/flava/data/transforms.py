# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import math
import random
import warnings
from functools import partial
from typing import List, Tuple, Optional, Callable, Any, Union

import numpy as np
import torch
import torchvision.transforms.functional as F
from PIL import Image
from torchvision import transforms
from transformers import BertTokenizer


LOGIT_LAPLACE_EPS: float = 0.1
IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)
IMAGE_DEFAULT_SIZE = (224, 224)
IMAGE_PRETRAINING_MEAN = (0.48145466, 0.4578275, 0.40821073)
IMAGE_PRETRAINING_STD = (0.26862954, 0.26130258, 0.27577711)
VL_MAX_LENGTH_DEFAULT = 77
TEXT_MAX_LENGTH_DEFAULT = 512


def map_pixels(x: torch.Tensor) -> torch.Tensor:
    if x.dtype != torch.float:
        raise ValueError("expected input to have type float")

    return (1 - 2 * LOGIT_LAPLACE_EPS) * x + LOGIT_LAPLACE_EPS


def encode_text_batch(batch, tokenizer, text_column="text", *args, **kwargs):
    return tokenizer(batch[text_column], *args, **kwargs)


def encode_text(text, tokenizer, *args, **kwargs):
    return tokenizer(text, *args, **kwargs)


def default_torchvision_transforms():
    transform = transforms.Compose(
        [
            transforms.Resize(IMAGE_DEFAULT_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=IMAGENET_DEFAULT_MEAN,
                std=IMAGENET_DEFAULT_STD,
            ),
        ]
    )
    return transform, transform


def default_image_pretraining_transforms():
    return MaskedImageModelingTransform(), MaskedImageModelingTransform()


def default_text_transform(
    text_tokenizer: Optional[Callable] = None,
    max_text_length: int = TEXT_MAX_LENGTH_DEFAULT,
    **kwargs: Any,
):
    if text_tokenizer is None:
        text_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    text_transform = partial(
        encode_text,
        tokenizer=text_tokenizer,
        padding="max_length",
        max_length=max_text_length,
        truncation=True,
        return_tensors="pt",
        return_special_tokens_mask=True,
    )

    return text_transform


def pad_batch(batch, batch_size):
    for item in batch.keys():
        if isinstance(batch[item], torch.Tensor):
            diff = batch_size - batch[item].size(0)
            pad = batch[item][-diff:].detach().clone()
            batch[item] = torch.cat([batch[item], pad], dim=0)
    return batch


class TwoWayRandomResizedCrop(transforms.RandomResizedCrop):
    """
    Similar to RandomResizedCrop but returns two versions of the
    random crop with different sizings and interpolations.
    Note that the crop is same but the two returned images
    have different final sizes and interpolations
    """

    def __init__(
        self,
        size,
        second_size=None,
        second_interpolation=transforms.InterpolationMode.LANCZOS,
        **kwargs,
    ):
        super().__init__(size, **kwargs)
        # Backward compatibility with integer value
        if isinstance(second_interpolation, int):
            warnings.warn(
                "Argument second_interpolation should be of type InterpolationMode instead of int. "
                "Please, use InterpolationMode enum."
            )
            second_interpolation = transforms._interpolation_modes_from_int(
                second_interpolation
            )

        if not isinstance(second_size, (list, tuple)):
            second_size = (second_size, second_size)

        self.second_size = second_size
        self.second_interpolation = second_interpolation

    def __call__(self, img):
        i, j, h, w = self.get_params(img, self.scale, self.ratio)
        if isinstance(self.interpolation, (tuple, list)):
            interpolation = random.choice(self.interpolation)
        else:
            interpolation = self.interpolation

        if self.second_size is None:
            return F.resized_crop(img, i, j, h, w, self.size, interpolation)
        else:
            return (
                F.resized_crop(img, i, j, h, w, self.size, interpolation),
                F.resized_crop(
                    img, i, j, h, w, self.second_size, self.second_interpolation
                ),
            )


class MaskingGenerator:
    def __init__(
        self,
        input_size,
        num_masking_patches,
        min_num_patches=4,
        max_num_patches=None,
        min_aspect=0.3,
        max_aspect=None,
    ):
        if not isinstance(input_size, tuple):
            input_size = (input_size,) * 2
        self.height, self.width = input_size

        self.num_patches = self.height * self.width
        self.num_masking_patches = num_masking_patches

        self.min_num_patches = min_num_patches
        self.max_num_patches = (
            num_masking_patches if max_num_patches is None else max_num_patches
        )

        max_aspect = max_aspect or 1 / min_aspect
        self.log_aspect_ratio = (math.log(min_aspect), math.log(max_aspect))

    def __repr__(self):
        repr_str = "Generator(%d, %d -> [%d ~ %d], max = %d, %.3f ~ %.3f)" % (
            self.height,
            self.width,
            self.min_num_patches,
            self.max_num_patches,
            self.num_masking_patches,
            self.log_aspect_ratio[0],
            self.log_aspect_ratio[1],
        )
        return repr_str

    def get_shape(self):
        return self.height, self.width

    def _mask(self, mask, max_mask_patches):
        delta = 0
        for _attempt in range(10):
            target_area = random.uniform(self.min_num_patches, max_mask_patches)
            aspect_ratio = math.exp(random.uniform(*self.log_aspect_ratio))
            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))
            if w < self.width and h < self.height:
                top = random.randint(0, self.height - h)
                left = random.randint(0, self.width - w)

                num_masked = mask[top : top + h, left : left + w].sum()
                # Overlap
                if 0 < h * w - num_masked <= max_mask_patches:
                    for i in range(top, top + h):
                        for j in range(left, left + w):
                            if mask[i, j] == 0:
                                mask[i, j] = 1
                                delta += 1

                if delta > 0:
                    break
        return delta

    def __call__(self):
        mask = np.zeros(shape=self.get_shape(), dtype=np.int)
        mask_count = 0
        while mask_count < self.num_masking_patches:
            max_mask_patches = self.num_masking_patches - mask_count
            max_mask_patches = min(max_mask_patches, self.max_num_patches)

            delta = self._mask(mask, max_mask_patches)
            if delta == 0:
                break
            else:
                mask_count += delta

        return mask


class MaskedImageModelingTransform:
    def __init__(
        self,
        encoder_input_size: int = 224,
        codebook_input_size: int = 112,
        scale: Tuple[float, float] = (0.9, 1.0),
        encoder_interpolation: str = transforms.InterpolationMode.BICUBIC,
        codebook_interpolation: str = transforms.InterpolationMode.LANCZOS,
        image_mean: Tuple[float, float, float] = IMAGE_PRETRAINING_MEAN,
        image_std: Tuple[float, float, float] = IMAGE_PRETRAINING_STD,
        mask_window_size: int = 14,
        mask_num_patches: int = 75,
        mask_max_patches: Optional[int] = None,
        mask_min_patches: int = 16,
    ):
        self.common_transform = transforms.Compose(
            [
                transforms.Lambda(
                    lambda img: img.convert("RGB") if img.mode != "RGB" else img
                ),
                TwoWayRandomResizedCrop(
                    size=encoder_input_size,
                    second_size=codebook_input_size,
                    scale=scale,
                    interpolation=encoder_interpolation,
                    second_interpolation=codebook_interpolation,
                ),
            ]
        )

        self.image_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=torch.tensor(image_mean),
                    std=torch.tensor(image_std),
                ),
            ]
        )

        self.codebook_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                map_pixels,
            ]
        )
        self.masked_position_generator = MaskingGenerator(
            mask_window_size,
            num_masking_patches=mask_num_patches,
            max_num_patches=mask_max_patches,
            min_num_patches=mask_min_patches,
        )

    def transform(self, image):
        image, image_for_codebook = self.common_transform(image)
        return {
            "image": self.image_transform(image),
            "image_for_codebook": self.codebook_transform(image_for_codebook),
            "image_patches_mask": torch.from_numpy(self.masked_position_generator()),
        }

    def __call__(self, images: Union[List[Image.Image], Image.Image]):
        if isinstance(images, list):
            output = {}
            for image in images:
                transformed_output = self.transform(image)
                for key in transformed_output:
                    if key not in output:
                        output[key] = []
                    output[key].append(transformed_output[key])
            return output
        else:
            return self.transform(images)


class VLTransform:
    def __init__(self, image_transform, text_transform):
        self.image_transform = image_transform
        self.text_transform = text_transform

    def __call__(self, info, dataset, itm_probability):
        output = {}
        text = info["text"]
        image = info["image"]
        if itm_probability > 0:
            output["itm_labels"] = torch.ones((1), dtype=torch.long)

        if random.random() < itm_probability:
            while text == info["text"]:
                text = dataset.select([random.randint(0, len(dataset) - 1)])[0]["text"]
            output["itm_labels"] = torch.zeros((1), dtype=torch.long)

        output.update(self.image_transform(image))
        output.update(self.text_transform(text))
        return output
