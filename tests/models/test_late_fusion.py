# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest

import torch
from tests.test_utils import assert_expected
from torchmultimodal.models.late_fusion import LateFusion
from torchmultimodal.modules.fusions.concat_fusion import ConcatFusionModule


class TestLateFusion:
    @pytest.fixture
    def encoders(self):
        return torch.nn.ModuleDict(
            {"c1": torch.nn.Identity(), "c2": torch.nn.Identity()}
        )

    @pytest.fixture
    def fusion_module(self):
        return ConcatFusionModule()

    @pytest.fixture
    def head_module(self):
        return torch.nn.Identity()

    @pytest.fixture
    def late_fusion(self, encoders, fusion_module, head_module):
        return LateFusion(
            encoders,
            fusion_module,
            head_module,
        )

    @pytest.fixture
    def modalities_1(self):
        return {
            "c1": torch.Tensor(
                [
                    [1, 0, 0.25, 0.75],
                    [0, 1, 0.6, 0.4],
                ]
            ),
            "c2": torch.Tensor(
                [
                    [3, 1, 0.8, 0.9],
                    [0.7, 2, 0.6, 0],
                ]
            ),
        }

    @pytest.fixture
    def modalities_2(self):
        return {
            "c1": torch.Tensor(
                [
                    [7, 0, 0.65],
                    [88, 5, 0.3],
                ]
            ),
            "c2": torch.Tensor(
                [
                    [8, 9, 0.8],
                    [0.74, 2, 0],
                ]
            ),
        }

    @pytest.fixture
    def modalities_3(self):
        return {
            "c3": torch.Tensor(
                [
                    [8, 0, 0.5, 0.7],
                    [1, 6, 0.6, 0.4],
                ]
            ),
        }

    def test_forward(self, late_fusion, modalities_1):
        actual = late_fusion(modalities_1)
        expected = torch.Tensor(
            [[1, 0, 0.25, 0.75, 3, 1, 0.8, 0.9], [0, 1, 0.6, 0.4, 0.7, 2, 0.6, 0]]
        )

        assert_expected(actual, expected)

    def test_script(self, late_fusion, modalities_2):
        scripted_late_fusion = torch.jit.script(late_fusion)
        actual = scripted_late_fusion(modalities_2)
        expected = torch.Tensor([[7, 0, 0.65, 8, 9, 0.8], [88, 5, 0.3, 0.74, 2, 0]])
        assert_expected(actual, expected)

    def test_missing_key_in_modalities(self, late_fusion, modalities_3):
        with pytest.raises(AssertionError):
            late_fusion(modalities_3)
