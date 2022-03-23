# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
from torch import nn
from torchmultimodal.modules.encoders.mil_encoder import MILEncoder
from torchmultimodal.modules.layers.mlp import MLP


class DummyEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.transformer = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(8, 2, batch_first=True),
            num_layers=1,
            norm=nn.LayerNorm(8),
        )

    def forward(self, x):
        return self.transformer(x)[:, 0, :]


class TestMILEncoder(unittest.TestCase):
    def setUp(self):
        self.batch_size = 2
        dim = 50
        self.input = torch.rand((self.batch_size, dim))
        self.input_bsz_1 = torch.rand(1, dim)
        self.partition_size = 10
        self.mlp_out_dim = 5
        self.shared_enc_dim = 8
        self.shared_encoder = nn.Linear(self.partition_size, self.shared_enc_dim)
        self.mlp = MLP(in_dim=self.shared_enc_dim, out_dim=self.mlp_out_dim)
        self.shared_test_encoder = DummyEncoder()
        self.transformer = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(
                self.shared_enc_dim, 2, batch_first=True
            ),
            num_layers=1,
            norm=nn.LayerNorm(self.shared_enc_dim),
        )

    def test_forward(self):
        partition_sizes = [self.partition_size] * 5
        mil_encoder = MILEncoder(
            partition_sizes,
            self.shared_encoder,
            self.shared_enc_dim,
            self.mlp,
            torch.sum,
        )
        out = mil_encoder(self.input)
        self.assertEqual(out.size(), (self.batch_size, self.mlp_out_dim))

        out = mil_encoder(self.input_bsz_1)
        self.assertEqual(out.size(), (1, self.mlp_out_dim))

    def test_transformer_pooling(self):
        partition_sizes = [2, 1]
        mil_encoder = MILEncoder(
            partition_sizes,
            self.shared_test_encoder,
            8,
            MLP(in_dim=self.shared_enc_dim, out_dim=self.mlp_out_dim),
            self.transformer,
        )
        input = torch.rand(self.batch_size, 3, 8)
        out = mil_encoder(input)
        self.assertEqual(out.size(), (self.batch_size, self.mlp_out_dim))

    def test_scripting(self):
        partition_sizes = [self.partition_size] * 5
        mil_encoder = MILEncoder(
            partition_sizes,
            self.shared_encoder,
            self.shared_enc_dim,
            self.mlp,
            torch.sum,
        )
        scripted_encoder = torch.jit.script(mil_encoder)
        scripted_encoder(self.input)

    def test_invalid_partitioning(self):
        partition_sizes = [12] * 5
        mil_encoder = MILEncoder(
            partition_sizes,
            self.shared_encoder,
            self.shared_enc_dim,
            self.mlp,
            torch.sum,
        )
        with self.assertRaises(ValueError):
            mil_encoder(self.input)
