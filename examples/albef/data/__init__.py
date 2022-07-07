# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import List

import torch
from examples.albef.data.transforms import test_transform, train_transform
from examples.albef.data.vqa_dataset import VQADataset
from torch.utils.data import DataLoader


def create_dataset(
    dataset: str,
    vqa_root: str,
    vg_root: str,
    answer_list: str,
    train_file: List[str],
    test_file: List[str],
):
    if dataset == "vqa":
        train_dataset = VQADataset(
            train_file,
            train_transform,
            vqa_root,
            vg_root,
            split="train",
        )
        vqa_test_dataset = VQADataset(
            test_file,
            test_transform,
            vqa_root,
            vg_root,
            split="test",
            answer_list=answer_list,
        )
        return train_dataset, vqa_test_dataset


def vqa_collate_fn(batch):
    image_list, question_list, answer_list, weight_list, n = [], [], [], [], []
    for image, question, answer, weights in batch:
        image_list.append(image)
        question_list.append(question)
        weight_list += weights
        answer_list += answer
        n.append(len(answer))
    return (
        torch.stack(image_list, dim=0),
        question_list,
        answer_list,
        torch.Tensor(weight_list),
        n,
    )


def create_sampler(datasets, shuffles, num_tasks, global_rank):
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
            pin_memory=True,
            sampler=sampler,
            shuffle=shuffle,
            collate_fn=collate_fn,
            drop_last=drop_last,
        )
        loaders.append(loader)
    return loaders
