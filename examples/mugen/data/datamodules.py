# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Callable, NamedTuple

import pytorch_lightning as pl
import torch
import torch.distributed as dist
import torch.utils.data as data
from torchmultimodal.transforms.bert_text_transform import BertTextTransform
from torchmultimodal.transforms.video_transform import VideoTransform

from .mugen_data import MUGENDataset, MUGENDatasetArgs


class DataModuleArgs(NamedTuple):
    batch_size: int = 16
    num_workers: int = 4


class MUGENDataModuleBase(pl.LightningDataModule):
    """General lightning data module for MUGEN dataset.

    Args:
        mugen_dataset_args (MUGENDatasetArgs): arguments for MUGENDataset.
        datamodule_args (DataModuleArgs): arguments for this LightningDataModule.
            See DataModuleArgs definition for defaults.
        shuffle (bool): whether to reshuffle data after each epoch.
            Defaults to True.
    """

    def __init__(
        self,
        mugen_dataset_args: MUGENDatasetArgs,
        data_module_args: DataModuleArgs = DataModuleArgs(),
        shuffle=True,
    ):
        super().__init__()
        self.data_module_args = data_module_args
        self.mugen_dataset_args = mugen_dataset_args
        self.shuffle = shuffle

    @property
    def n_classes(self):
        dataset = self._dataset(True)
        return dataset.n_classes

    def _collate_fn(self):
        return data.default_collate

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
            collate_fn=self._collate_fn(),
        )
        return dataloader

    def train_dataloader(self):
        return self._dataloader("train")

    def val_dataloader(self):
        return self._dataloader("val")

    def test_dataloader(self):
        return self._dataloader("test")


class VideoCLIPDataModule(MUGENDataModuleBase):
    """Lightning data module for MUGEN dataset texts and videos.

    Args:
        datamodule_args (DataModuleArgs): arguments for this LightningDataModule.
            See DataModuleArgs definition for defaults.
        shuffle (bool): whether to reshuffle data after each epoch.
            Defaults to True.
        text_transform (Callable): transform for text batches.
            Defaults to ``BertTextTransform()``
        video_transform (Callable): transform for video batches.
            Defaults to ``VideoTransform()``.
        **mugen_dataset_kwargs (Any): additional keyword arguments for MUGENDatasetArgs.
            Cannot contain args ``get_game_frame``, ``get_text_desc``.
    """

    def __init__(
        self,
        data_module_args: DataModuleArgs = DataModuleArgs(),
        shuffle: bool = True,
        text_transform: Callable = BertTextTransform(),
        video_transform: Callable = VideoTransform(),
        **mugen_dataset_kwargs,
    ):
        # Must get video and text when loading MUGEN dataset
        mugen_dataset_args = MUGENDatasetArgs(
            get_game_frame=True, get_text_desc=True, **mugen_dataset_kwargs
        )

        super().__init__(mugen_dataset_args, data_module_args, shuffle)
        self.text_transform = text_transform
        self.video_transform = video_transform

    def _collate_fn(self):
        def collate_with_transform(batch):
            video = [elem["video"] for elem in batch]
            video = torch.stack(video)
            video = self.video_transform(video)
            text = [elem["text"] for elem in batch]
            text = self.text_transform(text)
            return {"video": video, "text": text}

        return collate_with_transform
