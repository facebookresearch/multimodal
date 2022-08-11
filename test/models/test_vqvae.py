# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest

import torch
from test.test_utils import assert_expected, set_rng_seed
from torch import nn

from torchmultimodal.models.vqvae import VQVAE
from torchmultimodal.modules.layers.codebook import CodebookOutput


@pytest.fixture(autouse=True)
def random():
    set_rng_seed(4)


@pytest.fixture
def num_embeddings():
    return 4


@pytest.fixture
def embedding_dim():
    return 2


@pytest.fixture
def encoder():
    enc = nn.Linear(2, 2, bias=False)
    enc.weight = nn.Parameter(torch.tensor([[1.0, 2.0], [3.0, 4.0]]))
    return enc


@pytest.fixture
def decoder():
    dec = nn.Linear(2, 2, bias=False)
    dec.weight = nn.Parameter(torch.tensor([[1.0, 2.0], [3.0, 4.0]]))
    return dec


@pytest.fixture
def vqvae(encoder, decoder, num_embeddings, embedding_dim):
    return VQVAE(encoder, decoder, num_embeddings, embedding_dim)


class TestVQVAE:
    @pytest.fixture
    def decoder_input(self):
        d = torch.tensor([[[[3.0, 7.0], [6.0, 14.0]], [[9.0, 21.0], [12.0, 28.0]]]])
        return d

    @pytest.fixture
    def test_data(self, decoder_input):
        x = torch.tensor([[[[1.0, 1.0], [2.0, 2.0]], [[3.0, 3.0], [4.0, 4.0]]]])
        expected_decoded = torch.tensor(
            [[[[17.0, 37.0], [34.0, 74.0]], [[51.0, 111.0], [68.0, 148.0]]]]
        )
        expected_out = CodebookOutput(
            encoded_flat=torch.tensor(
                [[3.0, 9.0], [7.0, 21.0], [6.0, 12.0], [14.0, 28.0]]
            ),
            quantized=decoder_input,
            quantized_flat=torch.tensor(
                [[3.0, 9.0], [7.0, 21.0], [6.0, 12.0], [14.0, 28.0]]
            ),
            codebook_indices=torch.tensor([1, 3, 0, 2]),
        )
        return x, expected_decoded, expected_out

    def test_encode(self, test_data, vqvae):
        x, _, expected_out = test_data
        out = vqvae.encode(x)

        actual_quantized = out.quantized
        expected_quantized = expected_out.quantized
        assert_expected(actual_quantized, expected_quantized)

        actual_encoded_flat = out.encoded_flat
        expected_encoded_flat = expected_out.encoded_flat
        assert_expected(actual_encoded_flat, expected_encoded_flat)

        actual_codebook_indices = out.codebook_indices
        expected_codebook_indices = expected_out.codebook_indices
        assert_expected(actual_codebook_indices, expected_codebook_indices)

    def test_decode(self, decoder_input, test_data, decoder, vqvae):
        _, expected_decoded, _ = test_data
        actual_decoded = vqvae.decode(decoder_input)
        assert_expected(actual_decoded, expected_decoded)

        expected_decoded = decoder(decoder_input)
        assert_expected(actual_decoded, expected_decoded)

    def test_tokenize(self, test_data, vqvae):
        x, _, expected_out = test_data
        actual_quantized_flat = vqvae.tokenize(x)
        expected_quantized_flat = expected_out.quantized_flat
        assert_expected(actual_quantized_flat, expected_quantized_flat)

    def test_forward(self, test_data, vqvae):
        x, expected_decoded, expected_out = test_data
        out = vqvae(x)
        actual_decoded = out.decoded
        assert_expected(actual_decoded, expected_decoded)

        actual_quantized = out.codebook_output.quantized
        expected_quantized = expected_out.quantized
        assert_expected(actual_quantized, expected_quantized)

        actual_encoded_flat = out.codebook_output.encoded_flat
        expected_encoded_flat = expected_out.encoded_flat
        assert_expected(actual_encoded_flat, expected_encoded_flat)

        actual_codebook_indices = out.codebook_output.codebook_indices
        expected_codebook_indices = expected_out.codebook_indices
        assert_expected(actual_codebook_indices, expected_codebook_indices)
