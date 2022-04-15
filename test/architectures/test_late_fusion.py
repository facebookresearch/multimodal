# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
from torchmultimodal.architectures.late_fusion import LateFusionArchitecture
from torchmultimodal.modules.fusions.concat_fusion import ConcatFusionModule


class TestLateFusion(unittest.TestCase):
    def setUp(self):
        self.encoders = torch.nn.ModuleDict(
            {"c1": torch.nn.Identity(), "c2": torch.nn.Identity()}
        )
        self.fusion_module = ConcatFusionModule()
        self.head_module = torch.nn.Identity()
        self.late_fusion = LateFusionArchitecture(
            self.encoders,
            self.fusion_module,
            self.head_module,
        )

    def test_forward(self):
        modalities = {
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
        actual = self.late_fusion(modalities)
        expected = torch.Tensor(
            [[1, 0, 0.25, 0.75, 3, 1, 0.8, 0.9], [0, 1, 0.6, 0.4, 0.7, 2, 0.6, 0]]
        )

        torch.testing.assert_close(
            actual,
            expected,
            msg=f"actual: {actual}, expected: {expected}",
        )

    def test_script(self):
        modalities = {
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
        scripted_late_fusion = torch.jit.script(self.late_fusion)
        actual = scripted_late_fusion(modalities)
        expected = torch.Tensor([[7, 0, 0.65, 8, 9, 0.8], [88, 5, 0.3, 0.74, 2, 0]])
        torch.testing.assert_close(
            actual,
            expected,
            msg=f"actual: {actual}, expected: {expected}",
        )

    def test_missing_key_in_modalities(self):
        modalities = {
            "c3": torch.Tensor(
                [
                    [8, 0, 0.5, 0.7],
                    [1, 6, 0.6, 0.4],
                ]
            ),
        }
        with self.assertRaises(AssertionError):
            self.late_fusion(modalities)
