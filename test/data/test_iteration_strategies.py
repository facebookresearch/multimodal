# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
from omegaconf import DictConfig
from torchmultimodal.data import iteration_strategy_factory


class EmptyTensorDataset(torch.utils.data.Dataset):
    def __init__(self, len):
        self.len = len

    def __len__(self):
        return self.len

    def __getitem__(self, any):
        return torch.empty(0)


class TestIterationStrategies(unittest.TestCase):
    def setUp(self):
        pass

    def test_constant(self):
        config = DictConfig(
            content={
                "type": "constant",
                "idx": 1,
            }
        )
        text_len = 2
        dataloaders = {
            "image": torch.utils.data.DataLoader(EmptyTensorDataset(10), batch_size=1),
            "text": torch.utils.data.DataLoader(EmptyTensorDataset(10), batch_size=1),
        }
        f = iteration_strategy_factory(config)

        iter_strat = f(dataloaders)

        for x in range(text_len):
            self.assertEqual(iter_strat(), 1)
