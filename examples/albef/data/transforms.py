# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from examples.albef.data.randaugment import RandomAugment
from PIL import Image
from torchvision import transforms


normalize = transforms.Normalize(
    (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)
)

train_transform = transforms.Compose(
    [
        transforms.RandomResizedCrop(
            384, scale=(0.5, 1.0), interpolation=Image.BICUBIC
        ),
        transforms.RandomHorizontalFlip(),
        RandomAugment(
            2,
            7,
            is_pil=True,
            augs=[
                "Identity",
                "AutoContrast",
                "Equalize",
                "Brightness",
                "Sharpness",
                "ShearX",
                "ShearY",
                "TranslateX",
                "TranslateY",
                "Rotate",
            ],
        ),
        transforms.ToTensor(),
        normalize,
    ]
)

test_transform = transforms.Compose(
    [
        transforms.Resize(
            (384, 384), interpolation=Image.BICUBIC
        ),  # TODO: change 384 to image_size
        transforms.ToTensor(),
        normalize,
    ]
)
