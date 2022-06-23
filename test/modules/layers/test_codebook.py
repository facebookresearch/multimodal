# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
from test.test_utils import assert_expected, set_rng_seed
from torchmultimodal.modules.layers.codebook import Codebook


class TestCodebook(unittest.TestCase):
    """
    Test the Codebook class
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
        self.encoded.requires_grad_()
        # This is 4x5
        self.embedding_weights = torch.Tensor(
            [[1, 0, -1, -1, 2], [2, -2, 0, 0, 1], [2, 1, 0, 1, 1], [-1, -2, 0, 2, 0]]
        )
        # This is 4x3
        self.input_tensor_flat = torch.Tensor(
            [[1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1]]
        )

        self.vq = Codebook(
            num_embeddings=self.num_embeddings,
            embedding_dim=self.embedding_dim,
            decay=0.3,
        )

    def test_quantized_output(self):
        self.vq.embedding = self.embedding_weights
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

    def test_init_embedding_and_preprocess(self):
        assert not self.vq._is_embedding_init, "embedding init flag not False initially"

        _, _ = self.vq._init_embedding_and_preprocess(self.encoded)

        assert self.vq._is_embedding_init, "embedding init flag not True after init"

        actual_weight = self.vq.embedding
        expected_weight = torch.Tensor(
            [
                [2.0, -1.0, 0.0, 2.0, 0.0],
                [2.0, 1.0, 0.0, 1.0, 1.0],
                [0.0, 1.0, -1.0, 2.0, -1.0],
                [1.0, 0.0, -1.0, -1.0, 1.0],
            ]
        )
        assert_expected(actual_weight, expected_weight)

        actual_code_avg = self.vq.code_avg
        expected_code_avg = actual_weight
        assert_expected(actual_code_avg, expected_code_avg)

        actual_code_usage = self.vq.code_usage
        expected_code_usage = torch.ones(self.num_embeddings)
        assert_expected(actual_code_usage, expected_code_usage)

    def test_ema_update_embedding(self):
        _ = self.vq(self.encoded)

        actual_weight = self.vq.embedding
        expected_weight = torch.Tensor(
            [
                [0.7647, -1.4118, 0.0000, 1.5882, 0.0000],
                [2.0000, 1.0000, 0.0000, 1.0000, 1.0000],
                [-0.4118, 1.4118, -0.5882, 1.1765, -1.4118],
                [1.0000, 0.0000, -1.0000, -1.0000, 1.0000],
            ]
        )
        assert_expected(actual_weight, expected_weight, rtol=0.0, atol=1e-4)

        actual_code_avg = self.vq.code_avg
        expected_code_avg = torch.Tensor(
            [
                [1.3000, -2.4000, 0.0000, 2.7000, 0.0000],
                [2.0000, 1.0000, 0.0000, 1.0000, 1.0000],
                [-0.7000, 2.4000, -1.0000, 2.0000, -2.4000],
                [1.0000, 0.0000, -1.0000, -1.0000, 1.0000],
            ]
        )
        assert_expected(actual_code_avg, expected_code_avg, rtol=0.0, atol=1e-4)

        actual_code_usage = self.vq.code_usage
        expected_code_usage = torch.Tensor([1.7000, 1.0000, 1.7000, 1.0000])
        assert_expected(actual_code_usage, expected_code_usage, rtol=0.0, atol=1e-4)

    def test_register_buffer_tensors(self):
        out = self.vq(self.encoded)
        out.quantized.sum().backward()

        msg_has_grad = "tensor assigned to buffer but accumulated grad"
        assert not self.vq.code_avg.grad, msg_has_grad
        assert not self.vq.code_usage.grad, msg_has_grad
        assert not self.vq.embedding.grad, msg_has_grad

        assert not list(
            self.vq.parameters()
        ), "buffer variables incorrectly assigned as params"

    def test_init_embedding_smaller_encoded(self):
        encoded_small = self.encoded[:1, :, :2]
        out = self.vq(encoded_small)
        encoded_small_flat = out.encoded_flat
        embed = self.vq.embedding
        # Check for each embedding vector if there is one equal encoded vector + noise
        for emb in embed:
            assert any(
                [
                    torch.isclose(emb, enc, rtol=0, atol=0.01).all()
                    for enc in encoded_small_flat
                ]
            ), "embedding initialized from encoder output incorrectly"
