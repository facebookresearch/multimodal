# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from functools import partial
from typing import Callable, Optional

import torch
from data.dataset import build_flickr, build_gqa, collate_fn
from data.transforms import MDETRTransform
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, DistributedSampler
from transformers import RobertaTokenizerFast


class FlickrDataModule(LightningDataModule):
    def __init__(self, dataset_config, tokenizer: Optional[Callable] = None):
        super().__init__()
        self.dataset_config = dataset_config
        self.distributed = dataset_config.distributed
        self.batch_size = dataset_config.batch_size
        self.tokenizer = tokenizer

    def setup(self, stage: Optional[str] = None):
        if self.tokenizer is None:
            self.tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")
        self.transform = MDETRTransform(self.tokenizer, is_train=False)
        self.val = build_flickr(
            "val", self.tokenizer, self.transform, self.dataset_config
        )

    def val_dataloader(self):
        if self.distributed:
            sampler = DistributedSampler(self.val, shuffle=False)
        else:
            sampler = torch.utils.data.SequentialSampler(self.val)

        data_loader_val = DataLoader(
            self.val,
            batch_size=self.batch_size,
            sampler=sampler,
            drop_last=False,
            collate_fn=partial(collate_fn, self.tokenizer),
        )
        return data_loader_val


class GQADataModule(LightningDataModule):
    def __init__(self, dataset_config, tokenizer: Optional[Callable] = None):
        super().__init__()
        self.dataset_config = dataset_config
        self.distributed = dataset_config.distributed
        self.batch_size = dataset_config.batch_size
        self.epoch_chunks = dataset_config.epoch_chunks
        self.tokenizer = tokenizer

    def setup(self, stage: Optional[str] = None):
        if self.tokenizer is None:
            self.tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")
        self.train_transform = MDETRTransform(self.tokenizer, is_train=True)
        self.val_transform = MDETRTransform(self.tokenizer, is_train=False)

        if stage == "train":
            self.train = build_gqa(
                stage, self.tokenizer, self.train_transform, self.dataset_config
            )
        if stage == "val":
            self.val = build_gqa(
                stage, self.tokenizer, self.val_transform, self.dataset_config
            )

    def train_dataloader(self):
        # To handle very big datasets, we chunk it into smaller parts.
        if self.epoch_chunks > 0:
            print(
                f"Splitting the training set into {self.epoch_chunks} chunks of size approximately "
                f" {len(self.train) // self.epoch_chunks}"
            )
            chunks = torch.chunk(torch.arange(len(self.train)), self.epoch_chunks)
            datasets = [
                torch.utils.data.Subset(self.train, chunk.tolist()) for chunk in chunks
            ]
            if self.distributed:
                self.samplers_train = [
                    DistributedSampler(ds, shuffle=True) for ds in datasets
                ]
            else:
                self.samplers_train = [
                    torch.utils.data.RandomSampler(ds) for ds in datasets
                ]

            batch_samplers_train = [
                torch.utils.data.BatchSampler(
                    sampler_train, self.batch_size, drop_last=True
                )
                for sampler_train in self.samplers_train
            ]
            assert len(batch_samplers_train) == len(datasets)
            train_dataloaders = [
                DataLoader(
                    ds,
                    batch_sampler=batch_sampler_train,
                    collate_fn=partial(collate_fn, self.tokenizer),
                )
                for ds, batch_sampler_train in zip(datasets, batch_samplers_train)
            ]
            return train_dataloaders
        else:
            if self.distributed:
                self.sampler_train = DistributedSampler(self.train, shuffle=True)
            else:
                self.sampler_train = torch.utils.data.RandomSampler(self.train)
            batch_sampler_train = torch.utils.data.BatchSampler(
                self.sampler_train, self.batch_size, drop_last=True
            )
            train_dataloader = DataLoader(
                self.train,
                batch_sampler=batch_sampler_train,
                collate_fn=partial(collate_fn, self.tokenizer),
            )
            return train_dataloader

    def val_dataloader(self):
        if self.distributed:
            sampler = DistributedSampler(self.val, shuffle=False)
        else:
            sampler = torch.utils.data.SequentialSampler(self.val)

        data_loader_val = DataLoader(
            self.val,
            batch_size=self.batch_size,
            sampler=sampler,
            drop_last=False,
            collate_fn=partial(collate_fn, self.tokenizer),
        )
        return data_loader_val
