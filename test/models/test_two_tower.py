# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from typing import List

import torch
from torch import nn, Tensor
from torchmultimodal.models.late_fusion import LateFusion
from torchmultimodal.models.two_tower import TwoTower
from torchmultimodal.modules.fusions.concat_fusion import ConcatFusionModule


class Concat(nn.Module):
    def forward(self, x: List[Tensor]) -> Tensor:
        return torch.cat(x, dim=-1)


class TestTwoTower(unittest.TestCase):
    def setUp(self):
        self.tower_1 = LateFusion(
            {"c1": nn.Identity(), "c2": nn.Identity()},
            ConcatFusionModule(),
            nn.Identity(),
        )
        self.tower_2 = LateFusion(
            {"c3": nn.Identity(), "c4": nn.Identity()},
            ConcatFusionModule(),
            nn.Identity(),
        )
        self.tower_fusion = Concat()
        self.batch_size = 3
        self.data = {
            "c1": torch.rand(self.batch_size, 8),
            "c2": torch.rand(self.batch_size, 16),
            "c3": torch.rand(self.batch_size, 4),
            "c4": torch.rand(self.batch_size, 12),
        }

    def test_two_tower(self):
        two_tower = TwoTower(
            tower_id_to_tower={"tower_1": self.tower_1, "tower_2": self.tower_2},
            tower_fusion=self.tower_fusion,
        )
        out = two_tower(self.data)
        self.assertEqual(out[0].size(), (self.batch_size, 40))

    def test_shared_two_tower(self):
        two_tower = TwoTower(
            tower_id_to_tower={"tower_1": self.tower_1, "tower_2": self.tower_1},
            tower_fusion=self.tower_fusion,
            shared_tower_id_to_channel_mapping={"tower_2": {"c1": "c3", "c2": "c4"}},
        )
        out = two_tower(self.data)
        self.assertEqual(out[0].size(), (self.batch_size, 40))

    def test_two_tower_scripting(self):
        torch.jit.script(
            TwoTower(
                tower_id_to_tower={"tower_1": self.tower_1, "tower_2": self.tower_2},
                tower_fusion=self.tower_fusion,
            )
        )
        torch.jit.script(
            TwoTower(
                tower_id_to_tower={"tower_1": self.tower_1, "tower_2": self.tower_1},
                tower_fusion=self.tower_fusion,
                shared_tower_id_to_channel_mapping={
                    "tower_2": {"c3": "c1", "c4": "c2"}
                },
            )
        )
