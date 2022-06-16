# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
from test.test_utils import assert_expected, set_rng_seed
from torch import Tensor
from torchmultimodal.modules.encoders.albef_text_encoder import (
    ALBEFAttention,
    ALBEFIntermediate,
    ALBEFOutputLayer,
    ALBEFSelfAttention,
    ALBEFTextEmbeddings,
)


class TestALBEFTextEncoder:
    def test_intermediate(self):
        set_rng_seed(0)
        intermediate = ALBEFIntermediate(hidden_size=3)
        input = torch.randn(1, 3, 3)
        output = intermediate(input)
        expected = Tensor(
            [
                [0.582390, -0.167281, 0.187117],
                [-0.052958, 0.088763, -0.136094],
                [0.848244, -0.091195, 0.307349],
            ]
        ).unsqueeze(0)
        assert_expected(output, expected, rtol=0, atol=1e-4)

    def test_attention(self):
        set_rng_seed(0)
        attention = ALBEFAttention(
            hidden_size=3,
            num_attention_heads=1,
        )
        input = torch.randn(1, 3, 3)
        hidden_states = torch.randn(3, 3)
        (output,) = attention(input, hidden_states)
        expected = Tensor(
            [
                [-0.559244, 1.404537, -0.845293],
                [-1.367553, 0.995783, 0.371770],
                [0.154898, -1.294825, 1.139927],
            ]
        ).unsqueeze(0)
        assert_expected(output, expected, rtol=0, atol=1e-4)

    def test_self_attention(self):
        # test the output layer of ALBEFTextEncoder
        set_rng_seed(0)
        self_attention = ALBEFSelfAttention(hidden_size=3, num_attention_heads=1)
        input = torch.randn(1, 2, 3)
        (output,) = self_attention(input)
        expected = Tensor(
            [[0.037105, -1.265525, -0.781167], [-0.033819, -0.167038, -1.050785]]
        ).unsqueeze(0)
        assert_expected(output, expected, rtol=0, atol=1e-4)

    def test_attention_output(self):
        set_rng_seed(0)
        self_output = ALBEFOutputLayer(hidden_size=3)
        input = torch.randn(1, 2, 3)
        hidden_states = torch.randn(2, 3)
        output = self_output(input, hidden_states)
        expected = Tensor(
            [[0.068623, 1.188991, -1.257613], [0.769695, -1.412309, 0.642614]]
        ).unsqueeze(0)
        assert_expected(output, expected, rtol=0, atol=1e-4)

    def test_conv_proj(self):
        # test the embeddings of the ALBEFTextEncoder
        set_rng_seed(0)
        text_embeddings = ALBEFTextEmbeddings(hidden_size=3)
        input = torch.randint(10, (2, 2))
        output = text_embeddings(input)
        expected = Tensor(
            [
                [[0.306898, -1.349008, 1.042110], [-1.241071, 0.033332, 1.207738]],
                [[1.097125, -1.321374, 0.224249], [-0.619877, 1.410763, -0.790886]],
            ]
        )
        assert_expected(output, expected, rtol=0, atol=1e-4)
