# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
from test.test_utils import set_rng_seed
from torchmultimodal.modules.losses.vqvae import CommitmentLoss


class TestCommitment(unittest.TestCase):
    """
    Test the Commitment Loss
    """

    def setUp(self):
        torch.set_printoptions(precision=10)
        set_rng_seed(4)
        self.quantized = torch.randn((2, 3))
        self.encoded = torch.randn((2, 3))

    def test_loss_value(self):
        commitment = CommitmentLoss()
        loss = commitment(self.quantized, self.encoded)

        actual = loss.item()
        expected = 1.2070025206

        torch.testing.assert_close(
            actual,
            expected,
            msg=f"actual: {actual}, expected: {expected}",
        )
