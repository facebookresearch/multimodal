# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Callable, NamedTuple, Optional

import pytorch_lightning as pl
import torch
import torch.distributed as dist
import torch.utils.data as data

from .mugen_data import MUGENDataset, MUGENDatasetArgs


class DataModuleArgs(NamedTuple):
    batch_size: int = 16
    num_workers: int = 4


class MUGENDataModule(pl.LightningDataModule):
    """General lightning data module for MUGEN dataset.

    Args:
        mugen_dataset_args (MUGENDatasetArgs): arguments for MUGENDataset.
        datamodule_args (DataModuleArgs): arguments for this LightningDataModule.
            See DataModuleArgs definition for defaults.
        text_transform (Optional[Callable]): transform for text batches.
            Defaults to ``None``.
        video_transform (Optional[Callable]): transform for video batches.
            Defaults to ``None``.
        audio_transform (Optional[Callable]): transform for audio batches.
            Defaults to ``None``.
        shuffle (bool): whether to reshuffle data after each epoch.
            Defaults to ``True``.
    """

    def __init__(
        self,
        mugen_dataset_args: MUGENDatasetArgs,
        data_module_args: DataModuleArgs = DataModuleArgs(),
        text_transform: Optional[Callable] = None,
        video_transform: Optional[Callable] = None,
        audio_transform: Optional[Callable] = None,
        shuffle=True,
    ):
        super().__init__()
        self.data_module_args = data_module_args
        self.mugen_dataset_args = mugen_dataset_args
        self.text_transform = text_transform
        self.video_transform = video_transform
        self.audio_transform = audio_transform
        self.shuffle = shuffle

    @property
    def n_classes(self):
        dataset = self._dataset(True)
        return dataset.n_classes

    def _custom_collate_fn(self, batch):
        collated_batch = {}
        if self.mugen_dataset_args.get_game_frame:
            video = [elem["video"] for elem in batch]
            video = torch.stack(video)
            video = self.video_transform(video) if self.video_transform else video
            collated_batch["video"] = video
        if self.mugen_dataset_args.get_text_desc:
            text = [elem["text"] for elem in batch]
            # cannot be torch.stack'ed because still in raw text form, not Tensor
            text = self.text_transform(text) if self.text_transform else text
            collated_batch["text"] = text
        if self.mugen_dataset_args.get_audio:
            audio = [elem["audio"] for elem in batch]
            audio = torch.stack(audio)
            audio = self.audio_transform(audio) if self.audio_transform else audio
            collated_batch["audio"] = audio
        return collated_batch

    def _dataset(self, split):
        dataset = MUGENDataset(args=self.mugen_dataset_args, split=split)
        return dataset

    def _dataloader(self, split):
        dataset = self._dataset(split)
        if dist.is_initialized():
            sampler = data.distributed.DistributedSampler(
                dataset, num_replicas=dist.get_world_size(), rank=dist.get_rank()
            )
        else:
            sampler = None
        dataloader = data.DataLoader(
            dataset,
            batch_size=self.data_module_args.batch_size,
            num_workers=self.data_module_args.num_workers,
            pin_memory=True,
            sampler=sampler,
            shuffle=sampler is None and self.shuffle is True,
            collate_fn=self._custom_collate_fn,
        )
        return dataloader

    def train_dataloader(self):
        return self._dataloader("train")

    def val_dataloader(self):
        return self._dataloader("val")

    def test_dataloader(self):
        return self._dataloader("test")
