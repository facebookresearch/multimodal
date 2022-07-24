# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import examples.omnivore.data.transforms as CT  # custom transforms
import torch
import torchvision.transforms as T
from examples.omnivore.data.rand_aug3d import RandAugment3d
from torchvision.transforms.functional import InterpolationMode


# Image presets
class ImageNetClassificationPresetTrain:
    def __init__(
        self,
        *,
        crop_size,
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
        interpolation=InterpolationMode.BICUBIC,
        hflip_prob=0.5,
        auto_augment_policy=None,
        random_erase_prob=0.0,
        color_jitter_factor=(0.1, 0.1, 0.1, 0.1),
    ):
        transform_list = [T.RandomResizedCrop(crop_size, interpolation=interpolation)]
        if hflip_prob > 0:
            transform_list.append(T.RandomHorizontalFlip(hflip_prob))
        if auto_augment_policy is not None:
            if auto_augment_policy == "ra":
                transform_list.append(
                    T.autoaugment.RandAugment(interpolation=interpolation)
                )
            elif auto_augment_policy == "ta_wide":
                transform_list.append(
                    T.autoaugment.TrivialAugmentWide(interpolation=interpolation)
                )
            elif auto_augment_policy == "augmix":
                transform_list.append(T.autoaugment.AugMix(interpolation=interpolation))
            else:
                aa_policy = T.autoaugment.AutoAugmentPolicy(auto_augment_policy)
                transform_list.append(
                    T.autoaugment.AutoAugment(
                        policy=aa_policy, interpolation=interpolation
                    )
                )
        transform_list.extend(
            [
                T.ColorJitter(*color_jitter_factor),
                T.PILToTensor(),
                T.ConvertImageDtype(torch.float),
                T.Normalize(mean=mean, std=std),
            ]
        )
        if random_erase_prob > 0:
            transform_list.append(T.RandomErasing(p=random_erase_prob))

        # For omnivore to make the image look like a video with C D H W layout
        transform_list.append(CT.Unsqueeze(pos=1))

        self.transforms = T.Compose(transform_list)

    def __call__(self, img):
        return self.transforms(img)


class ImageNetClassificationPresetEval:
    def __init__(
        self,
        *,
        crop_size,
        resize_size=256,
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
        interpolation=InterpolationMode.BICUBIC,
    ):

        self.transforms = T.Compose(
            [
                T.Resize(resize_size, interpolation=interpolation),
                T.CenterCrop(crop_size),
                T.PILToTensor(),
                T.ConvertImageDtype(torch.float),
                T.Normalize(mean=mean, std=std),
                # For omnivore to make the image look like a video with C D H W layout
                CT.Unsqueeze(pos=1),
            ]
        )

    def __call__(self, img):
        return self.transforms(img)


# Video presets
class VideoClassificationPresetTrain:
    def __init__(
        self,
        *,
        crop_size,
        resize_size,
        mean=(0.43216, 0.394666, 0.37645),
        std=(0.22803, 0.22145, 0.216989),
        hflip_prob=0.5,
    ):
        transform_list = [
            T.ConvertImageDtype(torch.float32),
            T.Resize(resize_size),
        ]
        if hflip_prob > 0:
            transform_list.append(T.RandomHorizontalFlip(hflip_prob))
        transform_list.extend(
            [
                T.Normalize(mean=mean, std=std),
                T.RandomCrop(crop_size),
                CT.ConvertTCHWtoCTHW(),
            ]
        )
        self.transforms = T.Compose(transform_list)

    def __call__(self, x):
        return self.transforms(x)


class VideoClassificationPresetEval:
    def __init__(
        self,
        *,
        crop_size,
        resize_size,
        mean=(0.43216, 0.394666, 0.37645),
        std=(0.22803, 0.22145, 0.216989),
    ):
        self.transforms = T.Compose(
            [
                T.ConvertImageDtype(torch.float32),
                T.Resize(resize_size),
                T.Normalize(mean=mean, std=std),
                T.CenterCrop(crop_size),
                CT.ConvertTCHWtoCTHW(),
            ]
        )

    def __call__(self, x):
        return self.transforms(x)


# Depth Presets
class DepthClassificationPresetTrain:
    def __init__(
        self,
        *,
        crop_size,
        max_depth=75,
        mean=(0.485, 0.456, 0.406, 0.0418),
        std=(0.229, 0.224, 0.225, 0.0295),
        interpolation=InterpolationMode.BILINEAR,
        hflip_prob=0.5,
        random_erase_prob=0.0,
        color_jitter_factor=(0.1, 0.1, 0.1, 0.1),
    ):
        transform_list = [
            CT.DepthNorm(max_depth=max_depth, clamp_max_before_scale=True),
            T.RandomResizedCrop(crop_size, interpolation=interpolation),
        ]

        if hflip_prob > 0:
            transform_list.append(T.RandomHorizontalFlip(hflip_prob))

        transform_list.extend(
            [
                RandAugment3d(interpolation=interpolation, num_ops=1),
                CT.ColorJitter3d(*color_jitter_factor),
            ]
        )
        if random_erase_prob > 0:
            transform_list.append(T.RandomErasing(p=random_erase_prob))

        transform_list.append(T.Normalize(mean=mean, std=std))
        transform_list.append(
            CT.DropChannels(
                channel_probs=[0.5, 0.5, 0.5, 0],
                tie_channels=[0, 1, 2],
                fill_values=[0, 0, 0, 0],
            )
        )
        # For omnivore to make the rgbd look like video with C D H W layout
        transform_list.append(CT.Unsqueeze(pos=1))

        self.transforms = T.Compose(transform_list)

    def __call__(self, img):
        return self.transforms(img)


class DepthClassificationPresetEval:
    def __init__(
        self,
        *,
        crop_size,
        resize_size=256,
        max_depth=75,
        mean=(0.485, 0.456, 0.406, 0.0418),
        std=(0.229, 0.224, 0.225, 0.0295),
        interpolation=InterpolationMode.BILINEAR,
    ):

        self.transforms = T.Compose(
            [
                CT.DepthNorm(max_depth=max_depth, clamp_max_before_scale=True),
                T.Resize(resize_size, interpolation=interpolation),
                T.CenterCrop(crop_size),
                T.Normalize(mean=mean, std=std),
                # For omnivore to make the depth image look like video with C D H W layout
                CT.Unsqueeze(pos=1),
            ]
        )

    def __call__(self, img):
        return self.transforms(img)
