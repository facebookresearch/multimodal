# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import List

import torch
from examples.albef.data.transforms import ALBEFTransform
from examples.albef.data.vqa_dataset import VQADataset
from torch.utils.data import DataLoader


def create_dataset(
    image_size: int,
    dataset: str,
    vqa_root: str,
    vg_root: str,
    answer_list: str,
    train_file: List[str],
    test_file: List[str],
):
    if dataset == "vqa":
        train_transform = ALBEFTransform(image_size)
        test_transform = ALBEFTransform(image_size, is_train=False)
        train_dataset = VQADataset(
            train_file,
            vqa_root,
            vg_root,
            train_transform,
            split="train",
        )
        vqa_test_dataset = VQADataset(
            test_file,
            vqa_root,
            vg_root,
            test_transform,
            split="test",
            answer_list=answer_list,
        )
        return train_dataset, vqa_test_dataset


def create_sampler(
    datasets, shuffles, num_tasks, global_rank
):  # args: is_distributed, optional num_tasks and global_rank
    samplers = []
    for dataset, shuffle in zip(datasets, shuffles):
        sampler = torch.utils.data.DistributedSampler(
            dataset, num_replicas=num_tasks, rank=global_rank, shuffle=shuffle
        )
        samplers.append(sampler)
    return samplers


def create_loader(datasets, samplers, batch_size, num_workers, is_trains, collate_fns):
    loaders = []
    for dataset, sampler, bs, n_worker, is_train, collate_fn in zip(
        datasets, samplers, batch_size, num_workers, is_trains, collate_fns
    ):
        if is_train:
            shuffle = sampler is None
            drop_last = True
        else:
            shuffle = False
            drop_last = False
        loader = DataLoader(
            dataset,
            batch_size=bs,
            num_workers=n_worker,
            pin_memory=True,  # TODO: what is this???
            sampler=sampler,
            shuffle=shuffle,
            collate_fn=collate_fn,
            drop_last=drop_last,
        )
        loaders.append(loader)
    return loaders
