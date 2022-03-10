# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, Iterable, Tuple, Union

import torch
import torch.nn.functional as F
import torchtext.transforms as text_transforms
from PIL.Image import Image
from pytorch.text.fb.transforms.clip_bpe import DEFAULT_BPE, CLIPBPETransform
from torchvision import transforms

CLIP_DEFAULT_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_DEFAULT_STD = (0.26862954, 0.26130258, 0.27577711)


def convert_to_rgb(img: Image):
    return img.convert("RGB")


class CLIPTransform:
    """Image and text transform for CLIP model.

    Image transform: either random resized crop (train mode) or resize and center
        crop, followed by RGB conversion, tensor conversion, and normalization.
    Text transform: applies CLIP's BPE tokenizer transform, adds start and end
        tokens, then pads/truncates to text_max_length as necessary.


    Args:
        image_size (Union[int, Tuple[int]): desired output image size.
        image_interpolation (torchvision.transforms.InterpolationMode):
            Torchvision interpolation mode used during resizing. Defaults to bicubic.
        image_mean (Tuple[float]): mean of images, used for normalization.
        image_std (Tuple[float]): std of images, used for normalization.
        text_max_length (int): Maximum length of text token sequences.
        text_bpe_path (str): Location of BPE file for text transform.
        text_start_token (str): Special start token passed to BPE tokenizer.
        text_end_token (str): Special end token passed to BPE tokenizer.
        is_train (bool): Whether transform is run in train mode.

    Inputs:
        image (Union[Iterable[Image], Image]): Image or batch of images upon which
            to apply the transform.
        text (Union[Iterable[str],str]): Text or batch of texts upon which to apply
            the transform.
    """

    def __init__(
        self,
        image_size: Union[int, Tuple[int]] = (224, 224),
        image_interpolation=transforms.InterpolationMode.BICUBIC,
        image_mean: Tuple[float] = CLIP_DEFAULT_MEAN,
        image_std: Tuple[float] = CLIP_DEFAULT_STD,
        text_max_length: int = 77,
        text_bpe_path: str = DEFAULT_BPE,
        text_start_token: str = "<start_of_text>",
        text_end_token: str = "<end_of_text>",
        is_train: bool = True,
    ):
        joint_transforms = [
            convert_to_rgb,
            transforms.ToTensor(),
            transforms.Normalize(image_mean, image_std),
        ]
        if isinstance(image_size, int):
            image_size = (image_size, image_size)
        if is_train:
            self.image_transform = transforms.Compose(
                [
                    transforms.RandomResizedCrop(
                        image_size, interpolation=image_interpolation
                    )
                ]
                + joint_transforms,
            )
        else:
            self.image_transform = transforms.Compose(
                [
                    transforms.Resize(image_size, interpolation=image_interpolation),
                    transforms.CenterCrop(image_size),
                ]
                + joint_transforms
            )

        bpe_transform = CLIPBPETransform(
            text_bpe_path, [text_start_token, text_end_token]
        )
        self.text_start_token = bpe_transform([text_start_token])[0][0]
        self.text_end_token = bpe_transform([text_end_token])[0][0]
        self.text_max_length = text_max_length

        self.text_transform = transforms.Compose(
            [
                bpe_transform,
                text_transforms.AddToken(self.text_start_token, begin=True),
                text_transforms.AddToken(self.text_end_token, begin=False),
                text_transforms.Truncate(self.text_max_length),
                text_transforms.ToTensor(padding_value=0),
            ]
        )

    def __call__(
        self, image: Union[Iterable[Image], Image], text: Union[Iterable[str], str]
    ) -> Dict[str, torch.Tensor]:
        # Convert to list
        if isinstance(text, str):
            text = [text]
        if isinstance(image, Image):
            image = [image]

        # Text transform
        text_result = self.text_transform(text)

        # Zero padding
        # TODO: consider collapsing into nn.Sequential
        max_encoded_length = text_result.size(-1)
        if max_encoded_length < self.text_max_length:
            pad_amount = self.text_max_length - max_encoded_length
            text_result = F.pad(text_result, (0, pad_amount))

        # Image transform
        image_result = torch.stack([self.image_transform(x) for x in image])

        return {"image": image_result, "text": text_result}
