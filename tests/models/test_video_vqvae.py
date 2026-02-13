# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from itertools import repeat

import pytest
import torch
from tests.test_utils import assert_expected, assert_expected_namedtuple, set_rng_seed
from torchmultimodal.models.video_gpt.video_vqvae import (
    AttentionResidualBlock,
    AxialAttention,
    AxialAttentionBlock,
    preprocess_int_conv_params,
    video_vqvae,
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


class TestAxialBlock:
    @pytest.fixture
    def hidden_dim(self):
        return 3

    @pytest.fixture
    def n_dim(self):
        return 3

    @pytest.fixture
    def input_shape(self, n_dim):
        return tuple(repeat(2, n_dim))

    @pytest.fixture
    def axial_block(self, input_shape, hidden_dim):
        return AxialAttentionBlock(len(input_shape), hidden_dim, 1)

    @pytest.fixture
    def q(self, input_shape, hidden_dim):
        n_heads = 1
        return torch.randn(1, n_heads, *input_shape, hidden_dim // n_heads)

    @pytest.fixture
    def kv(self, input_shape, hidden_dim):
        n_heads = 1
        return torch.randn(1, n_heads, *input_shape, hidden_dim // n_heads)

    @pytest.fixture
    def axial_attn(self):
        return AxialAttention(1)  # only on second axis of input

    def test_axial_attention(self, axial_attn, q, kv):
        k = v = kv
        actual = axial_attn(q, k, v)
        expected = torch.tensor(
            [
                [
                    [
                        [
                            [[-0.5869, 1.8958, 0.8688], [0.0299, 0.2098, 1.2741]],
                            [[-0.6662, 1.9747, 0.8980], [0.1002, 0.2094, 1.5472]],
                        ],
                        [
                            [[0.5902, -0.3275, -0.8727], [-1.0557, 1.0791, 0.3916]],
                            [[0.6623, -0.3223, -0.8948], [-1.0755, 1.0763, 0.3708]],
                        ],
                    ]
                ]
            ]
        )
        assert_expected(actual, expected, rtol=0, atol=1e-4)

    def test_axial_block_forward(self, axial_block, hidden_dim, input_shape):
        """Test AxialAttentionBlock with sub-components"""
        x = 2 * torch.ones(1, hidden_dim, *input_shape)
        actual = axial_block(x)
        expected = torch.tensor(
            [
                [
                    [
                        [[0.822055, 0.822055], [0.822055, 0.822055]],
                        [[0.822055, 0.822055], [0.822055, 0.822055]],
                    ],
                    [
                        [[-0.767143, -0.767143], [-0.767143, -0.767143]],
                        [[-0.767143, -0.767143], [-0.767143, -0.767143]],
                    ],
                    [
                        [[-0.916860, -0.916860], [-0.916860, -0.916860]],
                        [[-0.916860, -0.916860], [-0.916860, -0.916860]],
                    ],
                ]
            ]
        )
        assert_expected(actual, expected, rtol=0, atol=1e-4)

    def test_axial_block_channel_dim(self, axial_block, hidden_dim, input_shape):
        """Test dim check in forward of AxialAttentionBlock"""
        x = torch.zeros(1, hidden_dim + 1, *input_shape)
        with pytest.raises(ValueError):
            _ = axial_block(x)


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
                        [[2.4492, 2.4492], [2.4492, 2.4492]],
                        [[2.4492, 2.4492], [2.4492, 2.4492]],
                    ],
                    [
                        [[2.3055, 2.3055], [2.3055, 2.3055]],
                        [[2.3055, 2.3055], [2.3055, 2.3055]],
                    ],
                    [
                        [[1.9071, 1.9071], [1.9071, 1.9071]],
                        [[1.9071, 1.9071], [1.9071, 1.9071]],
                    ],
                    [
                        [[1.7587, 1.7587], [1.7587, 1.7587]],
                        [[1.7587, 1.7587], [1.7587, 1.7587]],
                    ],
                ]
            ]
        )
        assert_expected(actual, expected, rtol=0, atol=1e-4)


class TestVideoEncoder:
    @pytest.fixture
    def encoder(self, params):
        in_channel_dims, _, kernel_sizes, _ = params

        def get_encoder(strides):
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

        return get_encoder

    @pytest.fixture
    def uneven_strides(self):
        return ((2, 2, 2), (1, 2, 2))

    @pytest.fixture
    def big_input(self):
        return torch.ones(1, 2, 4, 8, 8)

    def test_forward(self, input_tensor, encoder, params):
        strides = params[-1]
        model = encoder(strides)
        actual = model(input_tensor)
        expected = torch.tensor(
            [
                [
                    [
                        [[-0.6480, -0.5961], [-0.6117, -0.6640]],
                        [[-0.7177, -0.7569], [-0.5477, -0.5710]],
                    ],
                    [
                        [[-0.1906, -0.1636], [-0.2265, -0.1501]],
                        [[-0.1730, -0.1398], [-0.2598, -0.1510]],
                    ],
                ]
            ]
        )
        assert_expected(actual, expected, rtol=0, atol=1e-4)

    def test_latent_shape(self, big_input, encoder, uneven_strides):
        downsampler = encoder(uneven_strides)
        output = downsampler(big_input)
        actual = output.shape[2:]
        expected = downsampler.get_latent_shape(big_input.shape[2:])
        assert_expected(actual, expected)


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
                        [[-0.2129, -0.1894], [-0.2358, -0.2302]],
                        [[-0.2012, -0.1757], [-0.2264, -0.2067]],
                    ],
                    [
                        [[-0.1252, -0.1220], [-0.1235, -0.1280]],
                        [[-0.1502, -0.1264], [-0.1551, -0.1490]],
                    ],
                ]
            ]
        )
        assert_expected(actual, expected, rtol=0, atol=1e-4)


class TestVideoVQVAE:
    @pytest.fixture
    def vv(self):
        model = video_vqvae(
            in_channel_dim=2,
            encoder_hidden_dim=2,
            encoder_kernel_size=2,
            encoder_stride=1,
            encoder_n_layers=2,
            n_res_layers=1,
            attn_hidden_dim=2,
            num_embeddings=8,
            embedding_dim=2,
            decoder_hidden_dim=2,
            decoder_kernel_size=2,
            decoder_stride=1,
            decoder_n_layers=2,
        )
        model.eval()
        return model

    @pytest.fixture
    def expected_decoded(self):
        return torch.tensor(
            [
                [
                    [
                        [[0.1547, 0.1720], [0.1354, 0.1029]],
                        [[0.0828, 0.1086], [0.0837, 0.0637]],
                    ],
                    [
                        [[0.1914, 0.0667], [0.1442, -0.0180]],
                        [[0.0793, -0.0574], [0.0635, -0.0776]],
                    ],
                ]
            ]
        )

    @pytest.fixture
    def expected_codebook_output(self):
        return {
            "encoded_flat": torch.tensor(
                [
                    [-0.6480, -0.1906],
                    [-0.5961, -0.1636],
                    [-0.6117, -0.2265],
                    [-0.6640, -0.1501],
                    [-0.7177, -0.1730],
                    [-0.7569, -0.1398],
                    [-0.5477, -0.2598],
                    [-0.5710, -0.1510],
                ]
            ),
            "quantized_flat": torch.tensor(
                [
                    [-0.1801, 0.2553],
                    [-0.1801, 0.2553],
                    [-0.1801, 0.2553],
                    [-0.1801, 0.2553],
                    [-0.1801, 0.2553],
                    [-0.1801, 0.2553],
                    [-0.1801, 0.2553],
                    [-0.1801, 0.2553],
                ]
            ),
            "codebook_indices": torch.tensor(
                [
                    [
                        [[4, 4], [4, 4]],
                        [[4, 4], [4, 4]],
                    ],
                ]
            ),
            "quantized": torch.tensor(
                [
                    [
                        [
                            [[-0.1801, -0.1801], [-0.1801, -0.1801]],
                            [[-0.1801, -0.1801], [-0.1801, -0.1801]],
                        ],
                        [
                            [[0.2553, 0.2553], [0.2553, 0.2553]],
                            [[0.2553, 0.2553], [0.2553, 0.2553]],
                        ],
                    ]
                ]
            ),
        }

    @pytest.fixture
    def indices(self):
        return torch.tensor(
            [
                [
                    [[4, 4], [4, 4]],
                    [[4, 4], [4, 4]],
                ],
            ]
        )

    def test_encode(self, vv, input_tensor, indices):
        actual_codebook_indices = vv.encode(input_tensor)
        expected_codebook_indices = indices
        assert_expected(actual_codebook_indices, expected_codebook_indices)

    def test_decode(self, vv, indices, expected_decoded):
        actual = vv.decode(indices)
        expected = expected_decoded
        assert_expected(actual, expected, rtol=0, atol=1e-4)

    def test_forward(
        self, vv, input_tensor, expected_decoded, expected_codebook_output
    ):
        actual = vv(input_tensor)
        expected = {
            "decoded": expected_decoded,
            "codebook_output": expected_codebook_output,
        }

        assert_expected_namedtuple(actual, expected, rtol=0, atol=1e-4)


def test_preprocess_int_conv_params():
    channels = (3, 3, 3)
    kernel = 2
    stride = 1
    expected_kernel = torch.tensor(((2, 2, 2), (2, 2, 2), (2, 2, 2)))
    expected_stride = torch.tensor(((1, 1, 1), (1, 1, 1), (1, 1, 1)))
    actual_kernel, actual_stride = preprocess_int_conv_params(channels, kernel, stride)
    actual_kernel = torch.tensor(actual_kernel)
    actual_stride = torch.tensor(actual_stride)
    assert_expected(actual_kernel, expected_kernel)
    assert_expected(actual_stride, expected_stride)

    actual_kernel = preprocess_int_conv_params(channels, kernel_sizes=kernel)
    actual_kernel = torch.tensor(actual_kernel)
    assert_expected(actual_kernel, expected_kernel)

    actual_stride = preprocess_int_conv_params(channels, strides=stride)
    actual_stride = torch.tensor(actual_stride)
    assert_expected(actual_stride, expected_stride)
