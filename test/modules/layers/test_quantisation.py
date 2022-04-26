# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
from torch import nn
from torchmultimodal.modules.layers.quantisation import Quantisation
from torchmultimodal.utils.preprocess import (
    flatten_to_channel_vectors,
    reshape_from_channel_vectors,
)


class TestQuantisation(unittest.TestCase):
    """
    Test the Quantisation class
    """

    def setUp(self):
        torch.set_printoptions(precision=10)
        torch.manual_seed(4)
        self.num_embeddings = 4
        self.embedding_dim = 5
        self.encoded = torch.randn((2, self.embedding_dim, 3, 3))
        self.embedding_weights = torch.randn((self.num_embeddings, self.embedding_dim))

    def test_quantised_output(self):
        vq = Quantisation(
            num_embeddings=self.num_embeddings, embedding_dim=self.embedding_dim
        )
        vq.embedding = nn.Embedding.from_pretrained(self.embedding_weights)
        x, permuted_shape = flatten_to_channel_vectors(self.encoded, 1)
        output = vq(x)
        actual = reshape_from_channel_vectors(output, permuted_shape, 1)

        # This is shape (2,5,3,3)
        expected = torch.Tensor(
            [
                [
                    [
                        [-1.1265823841, -0.1376807690, 0.7218121886],
                        [-0.1376807690, 0.7218121886, 0.7218121886],
                        [-1.1265823841, -0.1376807690, -1.1265823841],
                    ],
                    [
                        [-0.5252661109, 0.3244057596, -0.8940765262],
                        [0.3244057596, -0.8940765262, -0.8940765262],
                        [-0.5252661109, 0.3244057596, -0.5252661109],
                    ],
                    [
                        [-0.9950634241, -1.0523844957, 1.7175949812],
                        [-1.0523844957, 1.7175949812, 1.7175949812],
                        [-0.9950634241, -1.0523844957, -0.9950634241],
                    ],
                    [
                        [0.2679379284, -0.4480970800, -0.3190571964],
                        [-0.4480970800, -0.3190571964, -0.3190571964],
                        [0.2679379284, -0.4480970800, 0.2679379284],
                    ],
                    [
                        [-0.6253433824, -0.5198931098, -0.8529881239],
                        [-0.5198931098, -0.8529881239, -0.8529881239],
                        [-0.6253433824, -0.5198931098, -0.6253433824],
                    ],
                ],
                [
                    [
                        [-1.6703201532, -1.1265823841, -0.1376807690],
                        [-1.1265823841, 0.7218121886, -1.1265823841],
                        [-0.1376807690, -0.1376807690, -0.1376807690],
                    ],
                    [
                        [0.8635767698, -0.5252661109, 0.3244057596],
                        [-0.5252661109, -0.8940765262, -0.5252661109],
                        [0.3244057596, 0.3244057596, 0.3244057596],
                    ],
                    [
                        [-1.5300362110, -0.9950634241, -1.0523844957],
                        [-0.9950634241, 1.7175949812, -0.9950634241],
                        [-1.0523844957, -1.0523844957, -1.0523844957],
                    ],
                    [
                        [0.5375117064, 0.2679379284, -0.4480970800],
                        [0.2679379284, -0.3190571964, 0.2679379284],
                        [-0.4480970800, -0.4480970800, -0.4480970800],
                    ],
                    [
                        [-1.6273639202, -0.6253433824, -0.5198931098],
                        [-0.6253433824, -0.8529881239, -0.6253433824],
                        [-0.5198931098, -0.5198931098, -0.5198931098],
                    ],
                ],
            ]
        )

        torch.testing.assert_close(
            actual,
            expected,
            msg=f"actual: {actual}, expected: {expected}",
        )

    def test_quantised_shape(self):
        vq = Quantisation(
            num_embeddings=self.num_embeddings, embedding_dim=self.embedding_dim
        )
        vq.embedding = nn.Embedding.from_pretrained(self.embedding_weights)
        x, permuted_shape = flatten_to_channel_vectors(self.encoded, 1)
        output = vq(x)
        output = reshape_from_channel_vectors(output, permuted_shape, 1)
        actual = torch.tensor(output.shape)
        expected = torch.tensor([2, 5, 3, 3])

        assert torch.equal(
            actual, expected
        ), f"actual shape: {actual}, expected shape: {expected}"
