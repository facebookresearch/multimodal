# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch
from tests.test_utils import assert_expected
from torchmultimodal.modules.losses.reconstruction_loss import ReconstructionLoss


class TestReconstructionLoss:
    @pytest.fixture
    def pred(self):
        return torch.Tensor(
            [
                [
                    [-0.6, 1.7, -0.5],
                    [-0.6, 1.9, -0.4],
                    [-0.5, 1.5, -0.2],
                    [-0.6, 1.8, -0.3],
                ]
            ]
        )

    @pytest.fixture
    def target(self):
        return torch.ones(1, 4, 3)

    @pytest.mark.parametrize(
        "normalize_target, expected_output", [(True, 1.2044), (False, 1.6489)]
    )
    def test_forward(self, normalize_target, expected_output, pred, target):
        loss_fn = ReconstructionLoss(normalize_target=normalize_target)
        mask = torch.Tensor([[1.0, 1.0, 0.0, 1.0], [0.0, 1.0, 1.0, 1.0]])
        loss = loss_fn(pred, target, mask)
        assert_expected(loss.item(), expected_output, atol=0, rtol=1e-4)

    def test_no_masking_error(self, pred, target):
        loss_fn = ReconstructionLoss()
        mask = torch.zeros(1, 4, 3)
        with pytest.raises(ValueError):
            loss_fn(pred, target, mask)
