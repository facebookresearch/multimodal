# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch
from test.test_utils import assert_expected, set_rng_seed

from torchmultimodal.models.video_vqvae import (
    AttentionResidualBlock,
    VideoDecoder,
    VideoEncoder,
)


@pytest.fixture(autouse=True)
def random():
    set_rng_seed(4)


@pytest.fixture(scope="module")
def params():
    in_channel_dims = (2, 2)
    out_channel_dims = (2, 2)
    kernel_sizes = ((2, 2, 2), (2, 2, 2))
    strides = ((1, 1, 1), (1, 1, 1))
    return in_channel_dims, out_channel_dims, kernel_sizes, strides


@pytest.fixture(scope="module")
def input_tensor():
    return torch.ones(1, 2, 2, 2, 2)


class TestAttentionResidualBlock:
    def test_hidden_dim_assertion(self):
        with pytest.raises(ValueError):
            _ = AttentionResidualBlock(1)

    def test_forward(self):
        block = AttentionResidualBlock(4)
        x = 2 * torch.ones(1, 4, 2, 2, 2)
        actual = block(x)
        expected = torch.tensor(
            [
                [
                    [
                        [[1.3809, 1.3809], [1.3809, 1.3809]],
                        [[1.3809, 1.3809], [1.3809, 1.3809]],
                    ],
                    [
                        [[2.2828, 2.2828], [2.2828, 2.2828]],
                        [[2.2828, 2.2828], [2.2828, 2.2828]],
                    ],
                    [
                        [[2.4332, 2.4332], [2.4332, 2.4332]],
                        [[2.4332, 2.4332], [2.4332, 2.4332]],
                    ],
                    [
                        [[1.3369, 1.3369], [1.3369, 1.3369]],
                        [[1.3369, 1.3369], [1.3369, 1.3369]],
                    ],
                ]
            ]
        )
        assert_expected(actual, expected, rtol=0, atol=1e-4)


class TestVideoEncoder:
    @pytest.fixture
    def encoder(self, params):
        in_channel_dims, _, kernel_sizes, strides = params
        enc = VideoEncoder(
            in_channel_dims=in_channel_dims,
            kernel_sizes=kernel_sizes,
            strides=strides,
            output_dim=2,
            n_res_layers=1,
            attn_hidden_dim=2,
        )
        enc.eval()
        return enc

    def test_forward(self, input_tensor, encoder):
        actual = encoder(input_tensor)
        expected = torch.tensor(
            [
                [
                    [
                        [[0.2843, 0.2720], [0.2920, 0.2692]],
                        [[0.2785, 0.2668], [0.2994, 0.2639]],
                    ],
                    [
                        [[0.8453, 0.8817], [0.8226, 0.8899]],
                        [[0.8626, 0.8972], [0.8008, 0.9057]],
                    ],
                ]
            ]
        )
        assert_expected(actual, expected, rtol=0, atol=1e-4)


class TestVideoDecoder:
    @pytest.fixture
    def decoder(self, params):
        _, out_channel_dims, kernel_sizes, strides = params
        dec = VideoDecoder(
            out_channel_dims=out_channel_dims,
            kernel_sizes=kernel_sizes,
            strides=strides,
            input_dim=2,
            n_res_layers=1,
            attn_hidden_dim=2,
        )
        dec.eval()
        return dec

    def test_forward(self, input_tensor, decoder):
        actual = decoder(input_tensor)
        expected = torch.tensor(
            [
                [
                    [
                        [[0.1138, 0.1543], [0.1376, 0.1744]],
                        [[0.1230, 0.1658], [0.0899, 0.1514]],
                    ],
                    [
                        [[0.2163, 0.1884], [0.2364, 0.2448]],
                        [[0.1997, 0.1545], [0.2628, 0.2646]],
                    ],
                ]
            ]
        )
        assert_expected(actual, expected, rtol=0, atol=1e-4)
