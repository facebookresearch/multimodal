# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from functools import partial
from typing import Callable, Optional

import torch
from examples.mdetr.data.dataset import build_flickr, collate_fn
from examples.mdetr.data.transforms import MDETRTransform
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
