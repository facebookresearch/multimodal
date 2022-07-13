# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# This file, data_builder.py, consists of functions to build the dataset
# for training in torchmultimodal/examples/omnivore/train.py
# Since there are a lot of parameters, we will pass it with args

import os
import datetime
import time
import torch
import torchvision
import torchvision.datasets.samplers as video_samplers
import examples.omnivore.utils as utils
from examples.omnivore.data import datasets, presets, transforms, sampler
from torch.utils.data.dataloader import default_collate
from torchvision.transforms.functional import InterpolationMode


def lprint(*x):
    print(f"[{datetime.datetime.now()}]", *x)


def get_sampler(train_dataset, val_dataset, dataset_name, args):
    if dataset_name == "kinetics":
        train_sampler = video_samplers.RandomClipSampler(
            train_dataset.video_clips, args.train_clips_per_video
        )
        val_sampler = video_samplers.UniformClipSampler(
            val_dataset.video_clips, args.val_clips_per_video
        )
        if args.distributed:
            train_sampler = video_samplers.DistributedSampler(train_sampler)
            val_sampler = video_samplers.DistributedSampler(val_sampler)
    else:
        if args.distributed:
            if hasattr(args, "ra_sampler") and args.ra_sampler:
                train_sampler = sampler.RASampler(
                    train_dataset, shuffle=True, repetitions=args.ra_reps
                )
            else:
                train_sampler = torch.utils.data.distributed.DistributedSampler(
                    train_dataset
                )
            val_sampler = torch.utils.data.distributed.DistributedSampler(
                val_dataset, shuffle=False
            )
        else:
            train_sampler = torch.utils.data.RandomSampler(train_dataset)
            val_sampler = torch.utils.data.SequentialSampler(val_dataset)
    return train_sampler, val_sampler


def get_single_data_loader_from_dataset(train_dataset, val_dataset, dataset_name, args):
    train_sampler, val_sampler = get_sampler(
        train_dataset, val_dataset, dataset_name, args
    )
    collate_fn = None
    num_classes = len(train_dataset.classes)
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
        # first two elements that assumed to be image/video and label
        collate_fn = lambda batch: mixupcutmix(*(default_collate(batch)[:2]))  # noqa: E731

    num_train_workers = args.workers
    num_val_workers = args.workers
    modality_dataset_match = {
        "image": "imagenet",
        "video": "kinetics",
        "rgbd": "sunrgbd",
    }
    for i, modality in enumerate(args.modalities):
        if dataset_name == modality_dataset_match.get(modality):
            if dataset_name == "kinetics":
                # Have extra workers for kinetics
                num_train_workers += args.extra_kinetics_dataloader_workers
                num_val_workers += args.extra_kinetics_dataloader_workers
            # Reduce worker to 1 if sampling factor is 0
            if args.train_data_sampling_factor[i] == 0:
                num_train_workers = 1
            if args.val_data_sampling_factor[i] == 0:
                num_val_workers = 1


    # Reduce the amount of validation workers
    num_val_workers = (num_val_workers // 2) + 1

    print(f"dataset {dataset_name} have {num_train_workers} train_workers and {num_val_workers} val_workers")

    train_data_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=num_train_workers,
        pin_memory=args.loader_pin_memory,
        collate_fn=collate_fn,
        drop_last=args.loader_drop_last,
    )
    val_data_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        sampler=val_sampler,
        num_workers=num_val_workers,
        pin_memory=args.loader_pin_memory,
        drop_last=args.loader_drop_last,
    )
    return train_data_loader, val_data_loader


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
    lprint(f"cache_path: {cache_path}")
    if args.cache_video_dataset and os.path.exists(cache_path):
        lprint(f"Loading {split} dataset from {cache_path}")
        dataset, _ = torch.load(cache_path)
        dataset.transform = transform
    else:
        if args.distributed:
            print(
                "It is recommended to pre-compute the dataset cache on a single-gpu first, it will be faster!"
            )
        lprint("Building kinetics dataset")
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
            print(f"Saving {split} dataset to {cache_path}")
            utils.mkdir(os.path.dirname(cache_path))
            utils.save_on_master((dataset, data_dir), cache_path)
    return dataset


def get_imagenet_data_loader(args):
    # Get imagenet data
    imagenet_path = args.imagenet_data_path
    lprint("Start getting imagenet dataset")

    imagenet_train_preset = presets.ImageNetClassificationPresetTrain(
        crop_size=args.train_crop_size,
        interpolation=InterpolationMode.BICUBIC,
        auto_augment_policy="ra",
        random_erase_prob=args.random_erase,
        color_jitter_factor=args.color_jitter_factor,
    )
    imagenet_val_preset = presets.ImageNetClassificationPresetEval(
        crop_size=args.val_crop_size, interpolation=InterpolationMode.BICUBIC
    )

    imagenet_train_dataset = torchvision.datasets.folder.ImageFolder(
        os.path.join(imagenet_path, "train"), imagenet_train_preset
    )
    imagenet_val_dataset = torchvision.datasets.folder.ImageFolder(
        os.path.join(imagenet_path, "val"), imagenet_val_preset
    )

    (
        imagenet_train_data_loader,
        imagenet_val_data_loader,
    ) = get_single_data_loader_from_dataset(
        imagenet_train_dataset, imagenet_val_dataset, "imagenet", args
    )
    lprint("Finish getting imagenet dataset")
    return imagenet_train_data_loader, imagenet_val_data_loader


def get_kinetics_data_loader(args):
    # Get kinetics data
    kinetics_path = args.kinetics_data_path
    video_train_preset = presets.VideoClassificationPresetTrain(
        crop_size=args.train_crop_size,
        resize_size=args.train_resize_size,
    )
    video_val_preset = presets.VideoClassificationPresetEval(
        crop_size=args.val_crop_size,
        resize_size=args.val_resize_size,
    )

    start_time = time.time()
    lprint("Start getting video dataset")
    video_train_dataset = get_kinetics_dataset(
        kinetics_path,
        split="train",
        transform=video_train_preset,
        step_between_clips=1,
        args=args,
    )
    video_val_dataset = get_kinetics_dataset(
        kinetics_path,
        split="val",
        transform=video_val_preset,
        step_between_clips=1,
        args=args,
    )
    lprint(f"Took {time.time() - start_time} seconds to get video dataset")

    (
        video_train_data_loader,
        video_val_data_loader,
    ) = get_single_data_loader_from_dataset(
        video_train_dataset, video_val_dataset, "kinetics", args
    )
    return video_train_data_loader, video_val_data_loader


def get_sunrgbd_data_loader(args):
    # Get sunrgbd data
    sunrgbd_path = args.sunrgbd_data_path
    lprint("Start creating depth dataset")
    depth_train_preset = presets.DepthClassificationPresetTrain(
        crop_size=args.train_crop_size,
        interpolation=InterpolationMode.BILINEAR,
        random_erase_prob=args.random_erase,
        max_depth=75.0,
        mean=(0.485, 0.456, 0.406, 0.0418),
        std=(0.229, 0.224, 0.225, 0.0295),
        color_jitter_factor=args.color_jitter_factor,
    )
    depth_val_preset = presets.DepthClassificationPresetEval(
        crop_size=args.val_crop_size,
        interpolation=InterpolationMode.BILINEAR,
        max_depth=75.0,
        mean=(0.485, 0.456, 0.406, 0.0418),
        std=(0.229, 0.224, 0.225, 0.0295),
    )

    depth_train_dataset = datasets.OmnivoreSunRgbdDatasets(
        root=sunrgbd_path, split="train", transform=depth_train_preset
    )
    depth_val_dataset = datasets.OmnivoreSunRgbdDatasets(
        root=sunrgbd_path, split="val", transform=depth_val_preset
    )

    (
        depth_train_data_loader,
        depth_val_data_loader,
    ) = get_single_data_loader_from_dataset(
        depth_train_dataset, depth_val_dataset, "sunrgbd", args
    )

    lprint("Finish getting depth dataset")
    return depth_train_data_loader, depth_val_data_loader


def get_omnivore_data_loader(args):
    modalities = args.modalities
    train_data_loader_list = []
    val_data_loader_list = []
    for modality in modalities:
        if modality == "image":
            train_data_loader, val_data_loader = get_imagenet_data_loader(args)
        elif modality == "video":
            train_data_loader, val_data_loader = get_kinetics_data_loader(args)
        elif modality == "rgbd":
            train_data_loader, val_data_loader = get_sunrgbd_data_loader(args)
        train_data_loader_list.append(train_data_loader)
        val_data_loader_list.append(val_data_loader)

    train_data_loader = datasets.ConcatIterable(
        train_data_loader_list,
        modalities,
        args.train_data_sampling_factor,
    )
    val_data_loader = datasets.ConcatIterable(
        val_data_loader_list,
        modalities,
        args.val_data_sampling_factor,
    )
    return train_data_loader, val_data_loader
