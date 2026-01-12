# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import warnings
from collections import OrderedDict

import pytest
import torch
from tests.test_utils import assert_expected, assert_expected_namedtuple, set_rng_seed
from torch import nn, tensor
from torchmultimodal.modules.layers.codebook import Codebook


@pytest.fixture(autouse=True)
def random_seed():
    set_rng_seed(4)


@pytest.fixture
def num_embeddings():
    return 4


@pytest.fixture
def embedding_dim():
    return 5


@pytest.fixture
def encoded():
    # This is 2x5x3
    encoded = tensor(
        [
            [
                [-1.0, 0.0, 1.0],
                [2.0, 1.0, 0.0],
                [0.0, -1.0, -1.0],
                [0.0, 2.0, -1.0],
                [-2.0, -1.0, 1.0],
            ],
            [
                [2.0, 2.0, -1.0],
                [1.0, -1.0, -2.0],
                [0.0, 0.0, 0.0],
                [1.0, 2.0, 1.0],
                [1.0, 0.0, 0.0],
            ],
        ]
    )
    encoded.requires_grad_()

    return encoded


@pytest.fixture
def embedding_weights():
    # This is 4x5
    return tensor(
        [
            [1.0, 0.0, -1.0, -1.0, 2.0],
            [2.0, -2.0, 0.0, 0.0, 1.0],
            [2.0, 1.0, 0.0, 1.0, 1.0],
            [-1.0, -2.0, 0.0, 2.0, 0.0],
        ]
    )


@pytest.fixture
def input_tensor_flat():
    # This is 4x3
    return tensor([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]])


@pytest.fixture
def codebook(num_embeddings, embedding_dim):
    return Codebook(
        num_embeddings=num_embeddings,
        embedding_dim=embedding_dim,
        decay=0.3,
    )


class TestCodebook:
    def test_quantized_output(self, codebook, embedding_weights, encoded):
        codebook.embedding = embedding_weights
        codebook._is_embedding_init = True
        actual = codebook(encoded)

        # This is shape (2,5,3)
        expected_quantized = tensor(
            [
                [
                    [2.0, 2.0, 1.0],
                    [1.0, 1.0, 0.0],
                    [0.0, 0.0, -1.0],
                    [1.0, 1.0, -1.0],
                    [1.0, 1.0, 2.0],
                ],
                [
                    [2.0, 2.0, -1.0],
                    [1.0, -2.0, -2.0],
                    [0.0, 0.0, 0.0],
                    [1.0, 0.0, 2.0],
                    [1.0, 1.0, 0.0],
                ],
            ]
        )
        expected_quantized_flat = (
            expected_quantized.permute(0, 2, 1).contiguous().view(-1, 5)
        )

        expected = {
            "encoded_flat": encoded.permute(0, 2, 1).contiguous().view(-1, 5),
            "quantized_flat": expected_quantized_flat,
            "codebook_indices": tensor([[2.0, 2.0, 0.0], [2.0, 1.0, 3.0]]).type(
                torch.LongTensor
            ),
            "quantized": expected_quantized,
        }

        assert_expected_namedtuple(actual, expected)

    def test_preprocess(self, codebook, encoded):
        encoded_flat, permuted_shape = codebook._preprocess(encoded)

        expected_flat_shape = torch.tensor([6, 5])
        expected_permuted_shape = torch.tensor([2, 3, 5])

        actual_flat_shape = torch.tensor(encoded_flat.shape)
        actual_permuted_shape = torch.tensor(permuted_shape)

        assert_expected(actual_flat_shape, expected_flat_shape)

        assert_expected(actual_permuted_shape, expected_permuted_shape)

    def test_preprocess_channel_dim_assertion(self, codebook, encoded):
        with pytest.raises(ValueError):
            codebook._preprocess(encoded[:, :4, :])

    def test_postprocess(self, codebook, input_tensor_flat):
        quantized = codebook._postprocess(input_tensor_flat, torch.Size([2, 2, 3]))
        actual_quantized_shape = torch.tensor(quantized.shape)
        expected_quantized_shape = torch.tensor([2, 3, 2])

        assert_expected(actual_quantized_shape, expected_quantized_shape)

    def test_init_embedding(self, codebook, encoded, num_embeddings):
        assert not codebook._is_embedding_init, (
            "embedding init flag not False initially"
        )

        encoded_flat, _ = codebook._preprocess(encoded)
        codebook._init_embedding(encoded_flat)

        assert codebook._is_embedding_init, "embedding init flag not True after init"

        actual_weight = codebook.embedding
        expected_weight = tensor(
            [
                [2.0, -1.0, 0.0, 2.0, 0.0],
                [2.0, 1.0, 0.0, 1.0, 1.0],
                [0.0, 1.0, -1.0, 2.0, -1.0],
                [1.0, 0.0, -1.0, -1.0, 1.0],
            ]
        )
        assert_expected(actual_weight, expected_weight)

        actual_code_avg = codebook.code_avg
        expected_code_avg = actual_weight
        assert_expected(actual_code_avg, expected_code_avg)

        actual_code_usage = codebook.code_usage
        expected_code_usage = torch.ones(num_embeddings)
        assert_expected(actual_code_usage, expected_code_usage)

    def test_ema_update_embedding(self, codebook, encoded):
        encoded_flat, _ = codebook._preprocess(encoded)
        codebook._init_embedding(encoded_flat)
        distances = torch.cdist(encoded_flat, codebook.embedding, p=2.0) ** 2
        codebook_indices = torch.argmin(distances, dim=1)
        codebook._ema_update_embedding(encoded_flat, codebook_indices)

        actual_weight = codebook.embedding
        expected_weight = tensor(
            [
                [0.7647, -1.4118, 0.0000, 1.5882, 0.0000],
                [2.0000, 1.0000, 0.0000, 1.0000, 1.0000],
                [-0.4118, 1.4118, -0.5882, 1.1765, -1.4118],
                [1.0000, 0.0000, -1.0000, -1.0000, 1.0000],
            ]
        )
        assert_expected(actual_weight, expected_weight, rtol=0.0, atol=1e-4)

        actual_code_avg = codebook.code_avg
        expected_code_avg = tensor(
            [
                [1.3000, -2.4000, 0.0000, 2.7000, 0.0000],
                [2.0000, 1.0000, 0.0000, 1.0000, 1.0000],
                [-0.7000, 2.4000, -1.0000, 2.0000, -2.4000],
                [1.0000, 0.0000, -1.0000, -1.0000, 1.0000],
            ]
        )
        assert_expected(actual_code_avg, expected_code_avg, rtol=0.0, atol=1e-4)

        actual_code_usage = codebook.code_usage
        expected_code_usage = tensor([1.7000, 1.0000, 1.7000, 1.0000])
        assert_expected(actual_code_usage, expected_code_usage, rtol=0.0, atol=1e-4)

    def test_register_buffer_tensors(self, codebook, encoded):
        out = codebook(encoded)
        out.quantized.sum().backward()

        msg_has_grad = "tensor assigned to buffer but accumulated grad"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            assert not codebook.code_avg.grad, msg_has_grad
            assert not codebook.code_usage.grad, msg_has_grad
            assert not codebook.embedding.grad, msg_has_grad

        assert not list(codebook.parameters()), (
            "buffer variables incorrectly assigned as params"
        )

    def test_init_embedding_smaller_encoded(self, codebook, encoded):
        encoded_small = encoded[:1, :, :2]
        encoded_small_flat, _ = codebook._preprocess(encoded_small)
        codebook._init_embedding(encoded_small_flat)
        embed = codebook.embedding
        # Check for each embedding vector if there is one equal encoded vector + noise
        for emb in embed:
            assert any(
                [
                    torch.isclose(emb, enc, rtol=0, atol=0.01).all()
                    for enc in encoded_small_flat
                ]
            ), "embedding initialized from encoder output incorrectly"

    def test_codebook_restart(self, codebook, encoded):
        encoded_flat, _ = codebook._preprocess(encoded)
        # First init and diversify embedding
        codebook._init_embedding(encoded_flat)
        # Use only embedding vector at index = 1 and force restarts.
        # Slightly modify encoded_flat to make sure vectors restart to something new
        encoded_flat_noise = encoded_flat + torch.randn_like(encoded_flat)
        codebook_indices_low_usage = torch.ones(encoded_flat.shape[0], dtype=torch.long)
        codebook._ema_update_embedding(encoded_flat_noise, codebook_indices_low_usage)

        # Check if embedding contains restarts
        for i, emb in enumerate(codebook.embedding):
            # We used only emb vector with index = 1, so check it was not restarted
            if i == 1:
                assert_expected(
                    emb,
                    codebook.code_avg[1] / codebook.code_usage[1],
                    rtol=0,
                    atol=1e-4,
                )
            # Compare each embedding vector to each encoded vector.
            # If at least one match, then restart happened.
            else:
                assert any(
                    [
                        torch.isclose(emb, enc, rtol=0, atol=1e-4).all()
                        for enc in encoded_flat_noise
                    ]
                ), "embedding restarted from encoder output incorrectly"

    def test_load_state_dict(self):
        state_dict = OrderedDict(
            [
                ("linear.weight", tensor([[1.0]])),
                ("linear.bias", tensor([2.0])),
                ("codebook.embedding", tensor([[3.0]])),
                ("codebook.code_usage", tensor([4.0])),
                ("codebook.code_avg", tensor([[5.0]])),
            ]
        )

        class DummyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(1, 1)
                self.codebook = Codebook(1, 1)

        model = DummyModel()
        assert not model.codebook._is_embedding_init
        model.load_state_dict(state_dict)
        assert model.codebook._is_embedding_init

        actual = model.codebook.embedding
        expected = state_dict["codebook.embedding"]
        assert_expected(actual, expected)

        actual = model.codebook.code_usage
        expected = state_dict["codebook.code_usage"]
        assert_expected(actual, expected)

        actual = model.codebook.code_avg
        expected = state_dict["codebook.code_avg"]
        assert_expected(actual, expected)

    def test_lookup(self, codebook, embedding_weights):
        codebook.embedding = embedding_weights
        indices_flat = tensor([[0, 1]])  # (b, seq_len)
        indices_shaped = tensor([[[0, 1], [2, 3]]])  # (b, shape)
        actual_quantized_flat = codebook.lookup(indices_flat)
        actual_quantized = codebook.lookup(indices_shaped)
        expected_quantized_flat = tensor(
            [[[1.0, 0.0, -1.0, -1.0, 2.0], [2.0, -2.0, 0.0, 0.0, 1.0]]]
        )
        expected_quantized = tensor(
            [
                [
                    [[1.0, 0.0, -1.0, -1.0, 2.0], [2.0, -2.0, 0.0, 0.0, 1.0]],
                    [[2.0, 1.0, 0.0, 1.0, 1.0], [-1.0, -2.0, 0.0, 2.0, 0.0]],
                ]
            ]
        )
        assert_expected(
            actual_quantized_flat, expected_quantized_flat, rtol=0.0, atol=1e-4
        )
        assert_expected(actual_quantized, expected_quantized, rtol=0.0, atol=1e-4)
