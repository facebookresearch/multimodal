# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Iterable, Tuple, Union

import torch
from PIL.Image import Image
from torchmultimodal.transforms.bert_text_transform import BertTextTransform
from torchvision import transforms
from torchvision.transforms import RandAugment


MEAN = (0.48145466, 0.4578275, 0.40821073)
STDEV = (0.26862954, 0.26130258, 0.27577711)


class ALBEFTransform:
    def __init__(
        self,
        image_size: int = 384,
        scale: Tuple[float, float] = (0.5, 1.0),
        image_interpolation=transforms.InterpolationMode.BICUBIC,
        mean: Tuple[float, float, float] = MEAN,
        stdev: Tuple[float, float, float] = STDEV,
        is_train: bool = True,
    ) -> None:
        if is_train:
            self.image_transform = transforms.Compose(
                [
                    transforms.RandomResizedCrop(
                        image_size, scale=scale, interpolation=image_interpolation
                    ),
                    transforms.RandomHorizontalFlip(),
                    RandAugment(),
                    transforms.ToTensor(),
                    transforms.Normalize(mean, stdev),
                ]
            )
            self.text_transform = BertTextTransform(truncate=True, max_seq_len=25)
        else:
            self.image_transform = transforms.Compose(
                [
                    transforms.Resize(
                        (image_size, image_size), interpolation=image_interpolation
                    ),
                    transforms.ToTensor(),
                    transforms.Normalize(mean, stdev),
                ]
            )
            self.text_transform = BertTextTransform()

    def __call__(
        self,
        image: Union[Iterable[Image], Image],
        text: Union[Iterable[str], str],
        is_answer: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Convert to list
        if isinstance(text, str):
            text = [text]
        if isinstance(image, Image):
            image = [image]

        # Text transform
        text_result = self.text_transform(text)
        if is_answer:
            # TODO: add 102 here to the end of each text_result (before padding)
            pass

        # Image transform
        image_result = torch.stack([self.image_transform(x) for x in image])
        return image_result, text_result
