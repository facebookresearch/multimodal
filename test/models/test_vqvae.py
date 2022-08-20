# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest

import torch

from test.test_utils import assert_expected, assert_expected_namedtuple, set_rng_seed
from torch import nn

from torchmultimodal.models.vqvae import VQVAE


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
def embedding_weights():
    return torch.tensor([[6.0, 12.0], [3.0, 9.0], [14.0, 28.0], [7.0, 21.0]])


@pytest.fixture
def encoder():
    class Encoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer = nn.Linear(2, 2, bias=False)
            self.layer.weight = nn.Parameter(torch.tensor([[1.0, 2.0], [3.0, 4.0]]))

        def forward(self, x):
            return self.layer(x)

        def get_latent_shape(self, input_shape):
            return input_shape  # dummy method

    return Encoder()


@pytest.fixture
def bad_encoder():
    class Encoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer = nn.Linear(2, 2, bias=False)
            self.layer.weight = nn.Parameter(torch.tensor([[1.0, 2.0], [3.0, 4.0]]))

        def forward(self, x):
            return self.layer(x)

    return Encoder()


@pytest.fixture
def decoder():
    dec = nn.Linear(2, 2, bias=False)
    dec.weight = nn.Parameter(torch.tensor([[1.0, 2.0], [3.0, 4.0]]))
    return dec


@pytest.fixture
def vqvae_builder(decoder, num_embeddings, embedding_dim, embedding_weights):
    def _vqvae(encoder):
        vqvae = VQVAE(encoder, decoder, num_embeddings, embedding_dim)
        vqvae.codebook.embedding = embedding_weights
        return vqvae.eval()  # switch off embedding weights initialization

    return _vqvae


@pytest.fixture
def indices():
    return torch.tensor([[[1, 3], [0, 2]]])  # (b, d1, d2)


class TestVQVAE:
    @pytest.fixture
    def vqvae(self, vqvae_builder, encoder):
        return vqvae_builder(encoder)

    @pytest.fixture
    def vqvae_bad_encoder(self, vqvae_builder, bad_encoder):
        return vqvae_builder(bad_encoder)

    @pytest.fixture
    def vqvae_bad_codebook(self, vqvae_builder, encoder, mocker):
        class BadCodebook(nn.Module):
            def __init__(self, num_embeddings, embedding_dim):
                super().__init__()

        mock_codebook = mocker.patch(
            "torchmultimodal.models.vqvae.Codebook", wraps=BadCodebook
        )

        return vqvae_builder(encoder), mock_codebook

    @pytest.fixture
    def x(self):
        return torch.tensor(
            [[[[1.0, 1.0], [2.0, 2.0]], [[3.0, 3.0], [4.0, 4.0]]]]
        )  # (b, c, d1, d2)

    @pytest.fixture
    def expected_decoded(self):
        return torch.tensor(
            [[[[17.0, 37.0], [34.0, 74.0]], [[51.0, 111.0], [68.0, 148.0]]]]
        )

    @pytest.fixture
    def expected_encoded_flat(self):
        return torch.tensor(
            [[3.0, 9.0], [7.0, 21.0], [6.0, 12.0], [14.0, 28.0]]  # (b x d1 x d2, c)
        )

    @pytest.fixture
    def expected_quantized_flat(self):
        return torch.tensor(
            [
                [3.0, 9.0],
                [7.0, 21.0],
                [6.0, 12.0],
                [14.0, 28.0],
            ]  # (b x d1 x d2, emb_dim)
        )

    @pytest.fixture
    def expected_quantized(self):
        return torch.tensor(
            [
                [[[3.0, 7.0], [6.0, 14.0]], [[9.0, 21.0], [12.0, 28.0]]]
            ]  # (b, emb_dim, d1, d2)
        )

    @pytest.fixture
    def expected_codebook_indices(self, indices):
        return indices

    @pytest.fixture
    def expected_codebook_output(
        self,
        expected_encoded_flat,
        expected_quantized_flat,
        expected_codebook_indices,
        expected_quantized,
    ):
        return {
            "encoded_flat": expected_encoded_flat,
            "quantized_flat": expected_quantized_flat,
            "codebook_indices": expected_codebook_indices,
            "quantized": expected_quantized,
        }

    def test_encode(self, vqvae, x, expected_codebook_indices):
        actual_codebook_indices = vqvae.encode(x)
        assert_expected(actual_codebook_indices, expected_codebook_indices)

    def test_encode_return_embeddings(
        self, vqvae, x, expected_quantized, expected_codebook_indices
    ):
        actual_codebook_indices, actual_quantized = vqvae.encode(
            x, return_embeddings=True
        )

        assert_expected(actual_quantized, expected_quantized)
        assert_expected(actual_codebook_indices, expected_codebook_indices)

    def test_decode(self, vqvae, indices, expected_decoded):
        actual_decoded = vqvae.decode(indices)
        assert_expected(actual_decoded, expected_decoded)

    def test_forward(self, vqvae, x, expected_decoded, expected_codebook_output):
        actual = vqvae(x)
        expected = {
            "decoded": expected_decoded,
            "codebook_output": expected_codebook_output,
        }
        assert_expected_namedtuple(actual, expected)

    def test_lookup(self, vqvae, indices):
        actual = vqvae.lookup(indices)
        expected = torch.tensor(
            [[[[3.0, 9.0], [7.0, 21.0]], [[6.0, 12.0], [14.0, 28.0]]]]
        )
        assert_expected(actual, expected)

    def test_latent_shape(self, vqvae):
        actual = vqvae.latent_shape(input_shape=(1, 2, 3))
        expected = (1, 2, 3)
        assert_expected(actual, expected)

    def test_latent_shape_bad_encoder(self, vqvae_bad_encoder):
        with pytest.raises(AttributeError):
            vqvae_bad_encoder.latent_shape(input_shape=(1, 2, 3))

    def test_lookup_bad_codebook(self, vqvae_bad_codebook, indices):
        vqvae, mock_codebook = vqvae_bad_codebook
        with pytest.raises(AttributeError):
            vqvae.lookup(indices)

        mock_codebook.assert_called_once()
