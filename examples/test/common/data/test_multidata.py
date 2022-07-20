# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import random

import numpy
import pytest
from common.data import (
    MultiDataLoader,
    MultiDataModule,
    iteration_strategy_factory,
)
from omegaconf import DictConfig
from pytorch_lightning import LightningDataModule

from .test_iteration_strategies import _dataloader_empty_tensor_dataset


class _TestDataModule(LightningDataModule):
    def __init__(self, size: int):
        super().__init__()
        self.size = size

    def train_dataloader(self):
        return _dataloader_empty_tensor_dataset(self.size)

    def val_dataloader(self):
        return _dataloader_empty_tensor_dataset(self.size)

    def test_dataloader(self):
        return _dataloader_empty_tensor_dataset(self.size)


@pytest.fixture(autouse=True)
def rng():
    random.seed(0)
    numpy.random.seed(0)
    print("setup-rng-0")
    yield "rng"
    print("setup-rng-1")


class TestMultiData:
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
        assert idxs == [0, 1] * 10

    def test_multidatamodule(self):
        mdm = MultiDataModule(
            datamodules={
                "image": _TestDataModule(10),
                "text": _TestDataModule(10),
            },
            iteration_strategy_factory=iteration_strategy_factory(
                config=DictConfig(
                    content={
                        "type": "round_robin",
                    }
                )
            ),
        )
