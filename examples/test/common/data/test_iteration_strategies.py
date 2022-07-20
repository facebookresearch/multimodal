# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import random
from collections import Counter
from typing import Dict, List, Optional

import numpy
import pytest
import torch
from common.data import iteration_strategy_factory
from omegaconf import DictConfig


class EmptyTensorDataset(torch.utils.data.Dataset):
    def __init__(self, len):
        self.len = len

    def __len__(self):
        return self.len

    def __getitem__(self, any):
        return torch.empty(0)


def _dataloader_empty_tensor_dataset(size: int):
    return torch.utils.data.DataLoader(EmptyTensorDataset(size), batch_size=1)


@pytest.fixture(autouse=True)
def rng():
    random.seed(0)
    numpy.random.seed(0)
    yield "rng"


class TestIterationStrategies:
    def test(self):
        def fn(
            self,
            config: DictConfig,
            dataloaders: Dict[str, torch.utils.data.DataLoader],
            expected_len: int,
            expected_idxs: Optional[List[int]] = None,
            expected_count: Optional[Dict[int, int]] = None,
        ):
            f = iteration_strategy_factory(config)
            iter_strat = f(dataloaders)
            counter = Counter()
            for i in range(expected_len):
                x = iter_strat()
                counter[x] += 1
                if expected_idxs is not None:
                    assert x == expected_idxs[i]

            if expected_count is not None:
                for idx, cnt in counter.items():
                    assert cnt == expected_count[idx]

        fn(
            self,
            config=DictConfig(
                content={
                    "type": "constant",
                    "idx": 1,
                }
            ),
            dataloaders={
                "image": _dataloader_empty_tensor_dataset(10),
                "text": _dataloader_empty_tensor_dataset(10),
            },
            expected_len=10,
            expected_idxs=[1] * 10,
        )

        fn(
            self,
            config=DictConfig(
                content={
                    "type": "round_robin",
                }
            ),
            dataloaders={
                "image": _dataloader_empty_tensor_dataset(10),
                "text": _dataloader_empty_tensor_dataset(10),
                "audio": _dataloader_empty_tensor_dataset(10),
            },
            expected_len=30,
            expected_idxs=[0, 1, 2] * 10,
        )

        fn(
            self,
            config=DictConfig(
                content={
                    "type": "size_proportional",
                }
            ),
            dataloaders={
                "image": _dataloader_empty_tensor_dataset(240),
                "text": _dataloader_empty_tensor_dataset(30),
                "audio": _dataloader_empty_tensor_dataset(30),
            },
            expected_len=300,
            expected_count={
                0: 241,
                1: 27,
                2: 32,
            },
        )

        fn(
            self,
            config=DictConfig(
                content={
                    "type": "ratios",
                    "params": {
                        "sampling_ratios": {
                            "image": 0.6,
                            "text": 0.2,
                            "audio": 0.2,
                        },
                    },
                }
            ),
            dataloaders={
                "image": _dataloader_empty_tensor_dataset(100),
                "text": _dataloader_empty_tensor_dataset(100),
                "audio": _dataloader_empty_tensor_dataset(100),
            },
            expected_len=300,
            expected_count={
                0: 183,
                1: 56,
                2: 61,
            },
        )
