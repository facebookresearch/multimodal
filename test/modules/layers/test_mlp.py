# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from functools import partial

import torch
from test.test_utils import set_rng_seed
from torchmultimodal.modules.layers.mlp import MLP


class TestMLP(unittest.TestCase):
    """
    Test the MLP class
    """

    def setUp(self):
        torch.set_printoptions(precision=10)
        set_rng_seed(0)
        self.in_dim = 5
        self.out_dim = 3
        self.input = torch.randn((4, 5))
        self.hidden_dims = [2, 6]

    def test_no_hidden_layers(self):
        mlp = MLP(in_dim=self.in_dim, out_dim=self.out_dim)
        actual = mlp(self.input)
        expected = torch.Tensor(
            [
                [0.165539, 0.455205, -0.331436],
                [1.186858, -0.380429, -0.888067],
                [0.813341, -1.444306, 0.507025],
                [1.710142, -0.744562, -0.199996],
            ],
        )
        torch.testing.assert_close(
            actual,
            expected,
            msg=f"actual: {actual}, expected: {expected}",
        )

    def test_pass_hidden_dims(self):
        mlp = MLP(
            in_dim=self.in_dim, out_dim=self.out_dim, hidden_dims=self.hidden_dims
        )
        actual = mlp(self.input)
        expected = torch.Tensor(
            [
                [-0.104062, 0.289350, 0.052587],
                [-0.114036, 0.186682, 0.028555],
                [0.243891, 0.085128, 0.087790],
                [0.395047, 1.070629, -0.927500],
            ],
        )
        torch.testing.assert_close(
            actual,
            expected,
            msg=f"actual: {actual}, expected: {expected}",
        )

    def test_activation_and_normalization(self):
        activation = torch.nn.LeakyReLU
        normalization = partial(torch.nn.BatchNorm1d, eps=0.1)
        mlp = MLP(
            in_dim=self.in_dim,
            out_dim=self.out_dim,
            hidden_dims=self.hidden_dims,
            activation=activation,
            normalization=normalization,
        )
        actual = mlp(self.input)
        expected = torch.Tensor(
            [
                [0.089560, 0.057747, -0.035710],
                [-0.069851, -0.418727, -0.457506],
                [-0.072189, -0.415917, -0.464918],
                [0.348458, 0.898804, -0.778149],
            ]
        )
        torch.testing.assert_close(
            actual,
            expected,
            msg=f"actual: {actual}, expected: {expected}",
        )

    def test_torchscript(self):
        mlp = MLP(in_dim=self.in_dim, out_dim=self.out_dim)
        torch.jit.script(mlp)
