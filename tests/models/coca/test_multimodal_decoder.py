# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch
from tests.test_utils import assert_expected, init_weights_with_constant
from torch import nn, Tensor
from torchmultimodal.models.coca.multimodal_decoder import CoCaMultimodalDecoder


class TestCoCaMultimodalDecoder:
    @pytest.fixture
    def batch_size(self):
        return 2

    @pytest.fixture
    def input_seq_len(self):
        return 5

    @pytest.fixture
    def num_image_positions(self):
        return 10

    @pytest.fixture
    def text_embedding_dim(self):
        return 4

    @pytest.fixture
    def multimodal_decoder(self, input_seq_len, batch_size, text_embedding_dim):
        decoder = CoCaMultimodalDecoder(
            input_seq_len=input_seq_len,
            text_embedding_dim=text_embedding_dim,
            n_layer=2,
            n_head=2,
            dim_feedforward=4 * text_embedding_dim,
            output_dim=3,
            final_layer_norm_eps=1e-5,
        )
        init_weights_with_constant(decoder)

        # Custom init final MLP layer weight, final LN, and text projection
        decoder.transformer_decoder.layer[1].feedforward.model[2].weight = nn.Parameter(
            torch.arange(
                decoder.transformer_decoder.layer[1]
                .feedforward.model[2]
                .weight.numel(),
                dtype=torch.float,
            ).reshape(
                decoder.transformer_decoder.layer[1].feedforward.model[2].weight.shape
            )
        )
        decoder.output_projection.weight = nn.Parameter(
            torch.arange(decoder.output_projection.weight.numel(), dtype=torch.float)
            .reshape(decoder.output_projection.weight.T.shape)
            .T
        )
        decoder.transformer_decoder.final_layer_norm.weight = nn.Parameter(
            torch.arange(
                decoder.transformer_decoder.final_layer_norm.weight.numel(),
                dtype=torch.float,
            )
        )
        decoder.eval()
        return decoder

    @pytest.fixture
    def text_inputs(self, batch_size, input_seq_len, text_embedding_dim):
        return torch.arange(0.0, 1.0, 1.0 / 40).reshape(
            batch_size, input_seq_len, text_embedding_dim
        )

    @pytest.fixture
    def image_inputs(self, batch_size, num_image_positions, text_embedding_dim):
        return torch.arange(10.0, 20.0, 1.0 / 8).reshape(
            batch_size, num_image_positions, text_embedding_dim
        )

    @pytest.fixture
    def expected(self):
        return Tensor(
            [
                [
                    [58.2492, 66.7214, 75.1935],
                    [58.2492, 66.7214, 75.1935],
                    [58.2492, 66.7214, 75.1935],
                    [58.2492, 66.7214, 75.1935],
                    [58.2492, 66.7214, 75.1935],
                ],
                [
                    [58.2492, 66.7214, 75.1935],
                    [58.2492, 66.7214, 75.1935],
                    [58.2492, 66.7214, 75.1935],
                    [58.2492, 66.7214, 75.1935],
                    [58.2492, 66.7214, 75.1935],
                ],
            ]
        )

    def test_coca_multimodal_decoder(
        self, text_inputs, image_inputs, multimodal_decoder, expected
    ):
        actual = multimodal_decoder(text_inputs, image_inputs)
        assert_expected(actual, expected, rtol=0, atol=1e-4)

    def test_scripting(self, text_inputs, image_inputs, multimodal_decoder):
        scripted_multimodal_decoder = torch.jit.script(multimodal_decoder)
        assert_expected(
            scripted_multimodal_decoder(text_inputs, image_inputs),
            multimodal_decoder(text_inputs, image_inputs),
            rtol=0,
            atol=1e-4,
        )
