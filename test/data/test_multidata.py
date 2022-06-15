# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

from omegaconf import DictConfig
from pytorch_lightning import LightningDataModule
from torchmultimodal.data import (
    MultiDataLoader,
    MultiDataModule,
    iteration_strategy_factory,
)

from .test_iteration_strategies import (
    EmptyTensorDataset,
    _dataloader_empty_tensor_dataset,
)


class TestDataModule(LightningDataModule):
    def __init__(self, size: int):
        super().__init__()
        self.size = size

    def train_dataloader(self):
        return _dataloader_empty_tensor_dataset(self.size)

    def val_dataloader(self):
        return _dataloader_empty_tensor_dataset(self.size)

    def test_dataloader(self):
        return _dataloader_empty_tensor_dataset(self.size)


class TestMultiData(unittest.TestCase):
    def test_multidataloader(self):
        dataloaders = {
            "image": _dataloader_empty_tensor_dataset(10),
            "text": _dataloader_empty_tensor_dataset(10),
        }
        mdl = MultiDataLoader(dataloaders)
        idxs = []
        it = iter(mdl)
        for _ in range(20):
            batch = next(it)
            idxs.append(batch["datamodule_index"])
        self.assertEqual(idxs, [0, 1] * 10)

    def test_multidatamodule(self):
        mdm = MultiDataModule(
            datamodules={
                "image": TestDataModule(10),
                "text": TestDataModule(10),
            },
            iteration_strategy_factory=iteration_strategy_factory(
                config=DictConfig(
                    content={
                        "type": "round_robin",
                    }
                )
            ),
        )
