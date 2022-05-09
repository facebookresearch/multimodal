# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
from test.test_utils import assert_tensors_equal
from torchmultimodal.modules.losses.vqvae import CommitmentLoss


class TestCommitment(unittest.TestCase):
    """
    Test the Commitment Loss
    """

    def setUp(self):
        self.quantized = torch.Tensor([[-1, 0, 1], [2, 1, 0]])
        self.encoded = torch.Tensor([[-2, -1, 0], [0, 2, -2]])
        self.commitment = CommitmentLoss()

    def test_loss_value(self):
        loss = self.commitment(self.quantized, self.encoded)

        actual = loss.item()
        expected = 2.0

        assert_tensors_equal(actual, expected)
