# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# This file, data_builder.py, consists of functions to build the dataset
# for training in torchmultimodal/examples/omnivore/train.py
# Since there are a lot of parameters, we allow to pass args here

import datetime
import logging
import os
import time

import omnivore.utils as utils
import torch
import torchvision
import torchvision.datasets.samplers as video_samplers
from omnivore.data import datasets, presets, transforms
from torch.utils.data.dataloader import default_collate
from torchvision.transforms.functional import InterpolationMode


logger = logging.getLogger(__name__)


def get_video_sampler(dataset, mode, args):
    # Get sampler for video dataset
    if mode == "train":
        sampler_class = video_samplers.RandomClipSampler
        clips_per_video = args.train_clips_per_video
    elif mode == "val":
        sampler_class = video_samplers.UniformClipSampler
        clips_per_video = args.val_clips_per_video

    sampler = sampler_class(dataset.video_clips, clips_per_video)

    if args.distributed:
        sampler = video_samplers.DistributedSampler(sampler)
    return sampler


def get_image_sampler(dataset, mode, args):
    # Get sampler for image with rgb or rgbd channel dataset
    if args.distributed:
        if mode == "train":
            shuffle = True
        elif mode == "val":
            shuffle = False
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset, shuffle=shuffle
        )
    else:
        if mode == "train":
            sampler = torch.utils.data.RandomSampler(dataset)
        elif mode == "val":
            sampler = torch.utils.data.SequentialSampler(dataset)
    return sampler


def construct_data_loader(dataset, sampler, num_workers, mode, args, drop_last=False):
    collate_fn = None
    if mode == "train":
        num_classes = len(dataset.classes)
        mixup_transforms = []
        if args.mixup_alpha > 0.0:
            mixup_transforms.append(
                transforms.RandomMixup(num_classes, p=1.0, alpha=args.mixup_alpha)
            )
        if args.cutmix_alpha > 0.0:
            mixup_transforms.append(
                transforms.RandomCutmix(num_classes, p=1.0, alpha=args.cutmix_alpha)
            )
        if mixup_transforms:
            mixupcutmix = torchvision.transforms.RandomChoice(mixup_transforms)
            # Since not all dataset return tuple of same length, we take the
            # first two elements for mixupcutmix during training
            collate_fn = lambda batch: mixupcutmix(*(default_collate(batch)[:2]))  # noqa: E731

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=args.loader_pin_memory,
        collate_fn=collate_fn,
        drop_last=drop_last,
    )
    return data_loader


def _get_cache_path(filepath):
    import hashlib

    h = hashlib.sha1(filepath.encode()).hexdigest()
    cache_path = os.path.join(
        "~", ".torch", "torchmultimodal", "omnivore_kinetics", h[:10] + ".pt"
    )
    cache_path = os.path.expanduser(cache_path)
    return cache_path


def get_kinetics_dataset(
    kinetics_path,
    split,
    transform,
    step_between_clips,
    args,
    frame_rate=16,
    frames_per_clip=32,
):
    data_dir = os.path.join(kinetics_path, split)
    cache_path = _get_cache_path(data_dir)
    logger.info(f"cache_path: {cache_path}")
    if args.cache_video_dataset and os.path.exists(cache_path):
        logger.info(f"Loading {split} dataset from {cache_path}")
        dataset, _ = torch.load(cache_path)
        dataset.transform = transform
    else:
        if args.distributed:
            logger.info(
                "It is recommended to pre-compute the dataset cache on a single-gpu first, it will be faster!"
            )
        logger.info("Building kinetics dataset")
        dataset = datasets.OmnivoreKinetics(
            kinetics_path,
            num_classes="400",
            extensions=("avi", "mp4"),
            output_format="TCHW",
            frames_per_clip=frames_per_clip,
            frame_rate=frame_rate,
            step_between_clips=step_between_clips,
            split=split,
            transform=transform,
            num_workers=args.kinetics_dataset_workers,
        )
        if args.cache_video_dataset:
            logger.info(f"Saving {split} dataset to {cache_path}")
            utils.mkdir(os.path.dirname(cache_path))
            utils.save_on_master((dataset, data_dir), cache_path)
    return dataset


def get_imagenet_data_loader(mode, num_workers, args):
    logger.info(f"Start getting {mode} imagenet data_loader")
    # Get imagenet data
    imagenet_path = args.imagenet_data_path

    if mode == "train":
        preset = presets.ImageNetClassificationPresetTrain(
            crop_size=args.train_crop_size,
            interpolation=InterpolationMode.BICUBIC,
            random_erase_prob=args.random_erase,
            color_jitter_factor=args.color_jitter_factor,
        )
        drop_last = args.loader_drop_last
    elif mode == "val":
        preset = presets.ImageNetClassificationPresetEval(
            crop_size=args.val_crop_size, interpolation=InterpolationMode.BICUBIC
        )
        drop_last = False

    dataset_dir = os.path.join(imagenet_path, mode)
    dataset = torchvision.datasets.folder.ImageFolder(dataset_dir, preset)
    sampler = get_image_sampler(dataset, mode, args)
    data_loader = construct_data_loader(
        dataset, sampler, num_workers, mode, args, drop_last=drop_last
    )

    logger.info(f"Finish getting {mode} imagenet data_loader")
    return data_loader


def get_kinetics_data_loader(mode, num_workers, args):
    logger.info(f"Start getting {mode} video data_loader")
    # Get kinetics data
    kinetics_path = args.kinetics_data_path
    if mode == "train":
        preset = presets.VideoClassificationPresetTrain(
            crop_size=args.train_crop_size,
            resize_size=args.train_resize_size,
        )
        drop_last = args.loader_drop_last
    elif mode == "val":
        preset = presets.VideoClassificationPresetEval(
            crop_size=args.val_crop_size,
            resize_size=args.val_resize_size,
        )
        drop_last = False

    start_time = time.time()
    logger.info(f"Start getting {mode} video dataset")
    dataset = get_kinetics_dataset(
        kinetics_path,
        split=mode,
        transform=preset,
        step_between_clips=1,
        args=args,
    )
    logger.info(f"Took {time.time() - start_time} seconds to get {mode} video dataset")

    sampler = get_video_sampler(dataset, mode, args)
    data_loader = construct_data_loader(
        dataset, sampler, num_workers, mode, args, drop_last=drop_last
    )
    logger.info(f"Finish getting {mode} video data_loader")
    return data_loader


def get_sunrgbd_data_loader(mode, num_workers, args):
    logger.info(f"Start creating {mode} depth dataset")
    # Get sunrgbd data
    sunrgbd_path = args.sunrgbd_data_path
    if mode == "train":
        preset = presets.DepthClassificationPresetTrain(
            crop_size=args.train_crop_size,
            interpolation=InterpolationMode.BILINEAR,
            random_erase_prob=args.random_erase,
            max_depth=75.0,
            mean=(0.485, 0.456, 0.406, 0.0418),
            std=(0.229, 0.224, 0.225, 0.0295),
            color_jitter_factor=args.color_jitter_factor,
        )
        drop_last = args.loader_drop_last
    elif mode == "val":
        preset = presets.DepthClassificationPresetEval(
            crop_size=args.val_crop_size,
            interpolation=InterpolationMode.BILINEAR,
            max_depth=75.0,
            mean=(0.485, 0.456, 0.406, 0.0418),
            std=(0.229, 0.224, 0.225, 0.0295),
        )
        drop_last = False

    dataset = datasets.OmnivoreSunRgbdDatasets(
        root=sunrgbd_path, split=mode, transform=preset
    )
    sampler = get_image_sampler(dataset, mode, args)
    data_loader = construct_data_loader(
        dataset, sampler, num_workers, mode, args, drop_last=drop_last
    )
    logger.info(f"Finish getting {mode} depth dataset")
    return data_loader


def get_omnivore_data_loader(mode, args):
    modalities = args.modalities
    data_loader_list = []
    data_loader_builder_map = {
        "image": get_imagenet_data_loader,
        "video": get_kinetics_data_loader,
        "rgbd": get_sunrgbd_data_loader,
    }
    if mode == "train":
        data_sampling_factor = args.train_data_sampling_factor
        shuffle = True
    elif mode == "val":
        data_sampling_factor = args.val_data_sampling_factor
        shuffle = False

    for i, modality in enumerate(modalities):
        # Determine the number of workers
        num_workers = args.workers
        if modality == "video":
            # Have extra workers for video data loader
            num_workers += args.extra_video_dataloader_workers
        if mode == "val":
            # Adjust num val workers with args.val_num_worker_ratio
            num_workers = max(int(num_workers * args.val_num_worker_ratio), 1)
        # Sampling factor 0 means the modality produce no data, hence no need for worker
        if data_sampling_factor[i] == 0:
            num_workers = 0
        # Build data_loader
        data_loader = data_loader_builder_map[modality](mode, num_workers, args)
        data_loader_list.append(data_loader)

    omnivore_data_loader = datasets.ConcatDataLoader(
        data_loader_list,
        modalities,
        data_sampling_factor,
        shuffle=shuffle,
    )
    return omnivore_data_loader
