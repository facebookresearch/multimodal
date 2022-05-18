# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
from test.test_utils import assert_expected
from torch import nn
from torchmultimodal.modules.layers.quantization import Quantization


class TestQuantization(unittest.TestCase):
    """
    Test the Quantization class
    """

    def setUp(self):
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
            num_embeddings=self.num_embeddings, embedding_dim=self.embedding_dim
        )
        self.vq.embedding = nn.Embedding.from_pretrained(self.embedding_weights)

    def test_quantized_output(self):
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

        assert torch.equal(
            actual_flat_shape, expected_flat_shape
        ), f"actual flattened shape: {actual_flat_shape}, expected flattened shape: {expected_flat_shape}"

        assert torch.equal(
            actual_permuted_shape, expected_permuted_shape
        ), f"actual permuted shape: {actual_permuted_shape}, expected permuted shape: {expected_permuted_shape}"

    def test_preprocess_channel_dim_assertion(self):
        with self.assertRaises(ValueError):
            encoded_flat, permuted_shape = self.vq._preprocess(self.encoded[:, :4, :])

    def test_postprocess(self):
        quantized = self.vq._postprocess(self.input_tensor_flat, torch.Size([2, 2, 3]))
        actual_quantized_shape = torch.tensor(quantized.shape)
        expected_quantized_shape = torch.tensor([2, 3, 2])

        assert torch.equal(
            actual_quantized_shape, expected_quantized_shape
        ), f"actual quantized shape: {actual_quantized_shape}, expected quantized shape: {expected_quantized_shape}"
