# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
from test.test_utils import assert_expected, set_rng_seed
from torch import nn
from torchmultimodal.modules.layers.quantization import Quantization


class TestQuantization(unittest.TestCase):
    """
    Test the Quantization class
    """

    def setUp(self):
        set_rng_seed(4)
        self.num_embeddings = 4
        self.embedding_dim = 5

        # This is 2x5x3
        self.encoded = torch.Tensor(
            [
                [[-1, 0, 1], [2, 1, 0], [0, -1, -1], [0, 2, -1], [-2, -1, 1]],
                [[2, 2, -1], [1, -1, -2], [0, 0, 0], [1, 2, 1], [1, 0, 0]],
            ]
        )
        # This is 4x5
        self.embedding_weights = torch.Tensor(
            [[1, 0, -1, -1, 2], [2, -2, 0, 0, 1], [2, 1, 0, 1, 1], [-1, -2, 0, 2, 0]]
        )
        # This is 4x3
        self.input_tensor_flat = torch.Tensor(
            [[1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1]]
        )

        self.vq = Quantization(
            num_embeddings=self.num_embeddings,
            embedding_dim=self.embedding_dim,
            decay=0.5,
        )

    def test_quantized_output(self):
        self.vq.embedding = nn.Embedding.from_pretrained(self.embedding_weights)
        self.vq._is_embedding_init = True
        output = self.vq(self.encoded)
        _, actual_quantized_flat, actual_codebook_indices, actual_quantized = output
        # This is shape (2,5,3)
        expected_quantized = torch.Tensor(
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
        expected_codebook_indices = torch.Tensor([2, 2, 0, 2, 1, 3]).type(
            torch.LongTensor
        )

        assert_expected(actual_quantized, expected_quantized)
        assert_expected(actual_quantized_flat, expected_quantized_flat)
        assert_expected(actual_codebook_indices, expected_codebook_indices)

    def test_preprocess(self):
        encoded_flat, permuted_shape = self.vq._preprocess(self.encoded)

        expected_flat_shape = torch.tensor([6, 5])
        expected_permuted_shape = torch.tensor([2, 3, 5])

        actual_flat_shape = torch.tensor(encoded_flat.shape)
        actual_permuted_shape = torch.tensor(permuted_shape)

        assert_expected(actual_flat_shape, expected_flat_shape)

        assert_expected(actual_permuted_shape, expected_permuted_shape)

    def test_preprocess_channel_dim_assertion(self):
        with self.assertRaises(ValueError):
            _, _ = self.vq._preprocess(self.encoded[:, :4, :])

    def test_postprocess(self):
        quantized = self.vq._postprocess(self.input_tensor_flat, torch.Size([2, 2, 3]))
        actual_quantized_shape = torch.tensor(quantized.shape)
        expected_quantized_shape = torch.tensor([2, 3, 2])

        assert_expected(actual_quantized_shape, expected_quantized_shape)

    def test_embed_init(self):
        assert not self.vq._is_embedding_init, "embedding init flag not False initially"

        _, _ = self.vq._init_embedding_and_preprocess(self.encoded)

        assert self.vq._is_embedding_init, "embedding init flag not True after init"

        actual_weight = self.vq.embedding.weight
        expected_weight = torch.Tensor(
            [
                [2.0, -1.0, 0.0, 2.0, 0.0],
                [2.0, 1.0, 0.0, 1.0, 1.0],
                [0.0, 1.0, -1.0, 2.0, -1.0],
                [1.0, 0.0, -1.0, -1.0, 1.0],
            ]
        )
        assert_expected(actual_weight, expected_weight)

        actual_code_avg = self.vq._code_avg
        expected_code_avg = actual_weight
        assert_expected(actual_code_avg, expected_code_avg)

        actual_code_usage = self.vq._code_usage
        expected_code_usage = torch.ones(self.num_embeddings)
        assert_expected(actual_code_usage, expected_code_usage)

    def test_ema_update(self):
        _ = self.vq(self.encoded)

        actual_weight = self.vq.embedding.weight
        expected_weight = torch.Tensor(
            [
                [1.0000, -1.3333, 0.0000, 1.6667, 0.0000],
                [2.0000, 1.0000, 0.0000, 1.0000, 1.0000],
                [-0.3333, 1.3333, -0.6667, 1.3333, -1.3333],
                [1.0000, 0.0000, -1.0000, -1.0000, 1.0000],
            ]
        )
        assert_expected(actual_weight, expected_weight, rtol=0.0, atol=1e-4)

        actual_code_avg = self.vq._code_avg
        expected_code_avg = torch.Tensor(
            [
                [1.5000, -2.0000, 0.0000, 2.5000, 0.0000],
                [2.0000, 1.0000, 0.0000, 1.0000, 1.0000],
                [-0.5000, 2.0000, -1.0000, 2.0000, -2.0000],
                [1.0000, 0.0000, -1.0000, -1.0000, 1.0000],
            ]
        )
        assert_expected(actual_code_avg, expected_code_avg, rtol=0.0, atol=1e-4)

        actual_code_usage = self.vq._code_usage
        expected_code_usage = torch.Tensor([1.5000, 1.0000, 1.5000, 1.0000])
        assert_expected(actual_code_usage, expected_code_usage, rtol=0.0, atol=1e-4)
