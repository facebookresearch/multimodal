# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import List

import pytest

import torch
from test.test_utils import assert_expected
from torch import nn, Tensor
from torchmultimodal.models.late_fusion import LateFusion
from torchmultimodal.models.two_tower import TwoTower
from torchmultimodal.modules.fusions.concat_fusion import ConcatFusionModule


@pytest.fixture
def tower_fusion():
    class Concat(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x: List[Tensor]) -> Tensor:
            return torch.cat(x, dim=-1)

    return Concat()


class TestTwoTower:
    @pytest.fixture
    def tower_1(self):
        return LateFusion(
            {"c1": nn.Identity(), "c2": nn.Identity()},
            ConcatFusionModule(),
            nn.Identity(),
        )

    @pytest.fixture
    def tower_2(self):
        return LateFusion(
            {"c3": nn.Identity(), "c4": nn.Identity()},
            ConcatFusionModule(),
            nn.Identity(),
        )

    @pytest.fixture
    def batch_size(self):
        return 3

    @pytest.fixture
    def data(self, batch_size):
        return {
            "c1": torch.rand(batch_size, 8),
            "c2": torch.rand(batch_size, 16),
            "c3": torch.rand(batch_size, 4),
            "c4": torch.rand(batch_size, 12),
        }

    @pytest.fixture
    def two_tower(self, tower_1, tower_2, tower_fusion):
        return TwoTower(
            tower_id_to_tower={"tower_1": tower_1, "tower_2": tower_2},
            tower_fusion=tower_fusion,
        )

    @pytest.fixture
    def shared_two_tower(self, tower_1, tower_fusion):
        return TwoTower(
            tower_id_to_tower={"tower_1": tower_1, "tower_2": tower_1},
            tower_fusion=tower_fusion,
            shared_tower_id_to_channel_mapping={"tower_2": {"c1": "c3", "c2": "c4"}},
        )

    @pytest.fixture
    def shared_two_tower_scripting(self, tower_1, tower_fusion):
        return TwoTower(
            tower_id_to_tower={"tower_1": tower_1, "tower_2": tower_1},
            tower_fusion=tower_fusion,
            shared_tower_id_to_channel_mapping={"tower_2": {"c3": "c1", "c4": "c2"}},
        )

    def test_two_tower(self, two_tower, data, batch_size):
        out = two_tower(data)
        assert_expected(out[0].size(), (batch_size, 40))

    def test_shared_two_tower(self, shared_two_tower, data, batch_size):
        out = shared_two_tower(data)
        assert_expected(out[0].size(), (batch_size, 40))

    def test_two_tower_scripting(self, two_tower, shared_two_tower_scripting):
        torch.jit.script(two_tower)
        torch.jit.script(shared_two_tower_scripting)
