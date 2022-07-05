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
    video_vqvae,
    VideoDecoder,
    VideoEncoder,
)
from torchmultimodal.modules.layers.codebook import CodebookOutput


@pytest.fixture(autouse=True)
def random():
    set_rng_seed(4)


@pytest.fixture(scope="module")
def params():
    in_channel_dims = (2, 2)
    out_channel_dims = (2, 2)
    kernel_sizes = (2, 2, 2)
    strides = (1, 1, 1)
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
        return VideoEncoder(
            in_channel_dims=in_channel_dims,
            kernel_sizes=kernel_sizes,
            strides=strides,
            n_res_layers=1,
            attn_hidden_dim=2,
            embedding_dim=2,
        )

    def test_forward(self, input_tensor, encoder):
        actual = encoder(input_tensor)
        expected = torch.tensor(
            [
                [
                    [
                        [[0.1565, 0.3358], [0.2368, 0.3180]],
                        [[0.2150, 0.3358], [0.2223, 0.2448]],
                    ],
                    [
                        [[0.9679, 0.6932], [0.9255, 0.7203]],
                        [[1.0502, 0.6932], [1.0287, 0.9623]],
                    ],
                ]
            ]
        )
        assert_expected(actual, expected, rtol=0, atol=1e-4)


class TestVideoDecoder:
    @pytest.fixture
    def decoder(self, params):
        _, out_channel_dims, kernel_sizes, strides = params
        return VideoDecoder(
            out_channel_dims=out_channel_dims,
            kernel_sizes=kernel_sizes,
            strides=strides,
            n_res_layers=1,
            attn_hidden_dim=2,
            embedding_dim=2,
        )

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


class TestVideoVQVAE:
    @pytest.fixture(scope="function")
    def vv(self, params):
        in_channel_dims, out_channel_dims, kernel_sizes, strides = params
        return video_vqvae(
            encoder_in_channel_dims=in_channel_dims,
            encoder_kernel_sizes=kernel_sizes,
            encoder_strides=strides,
            n_res_layers=1,
            attn_hidden_dim=2,
            num_embeddings=8,
            embedding_dim=2,
            decoder_out_channel_dims=out_channel_dims,
            decoder_kernel_sizes=kernel_sizes,
            decoder_strides=strides,
        )

    @pytest.fixture(scope="class")
    def test_data(self):
        decoded = torch.tensor(
            [
                [
                    [
                        [[0.0608, 0.1219], [0.0420, 0.1930]],
                        [[0.0507, 0.0971], [0.0203, 0.2616]],
                    ],
                    [
                        [[0.1319, 0.1547], [0.1375, 0.1287]],
                        [[0.1141, 0.1053], [0.1308, 0.1655]],
                    ],
                ]
            ]
        )
        out = CodebookOutput(
            encoded_flat=torch.tensor(
                [
                    [0.1565, 0.9679],
                    [0.3358, 0.6932],
                    [0.2368, 0.9255],
                    [0.3180, 0.7203],
                    [0.2150, 1.0502],
                    [0.3358, 0.6932],
                    [0.2223, 1.0287],
                    [0.2448, 0.9623],
                ]
            ),
            quantized_flat=torch.tensor(
                [
                    [0.1565, 0.9679],
                    [0.3358, 0.6932],
                    [0.2368, 0.9255],
                    [0.3180, 0.7203],
                    [0.2150, 1.0502],
                    [0.3358, 0.6932],
                    [0.2223, 1.0287],
                    [0.2448, 0.9623],
                ]
            ),
            codebook_indices=torch.tensor([0, 4, 5, 1, 3, 4, 7, 2]),
            quantized=torch.tensor(
                [
                    [
                        [
                            [[0.1565, 0.3358], [0.2368, 0.3180]],
                            [[0.2150, 0.3358], [0.2223, 0.2448]],
                        ],
                        [
                            [[0.9679, 0.6932], [0.9255, 0.7203]],
                            [[1.0502, 0.6932], [1.0287, 0.9623]],
                        ],
                    ]
                ]
            ),
        )
        return decoded, out

    def test_encode(self, vv, input_tensor, test_data):
        _, expected_out = test_data
        out = vv.encode(input_tensor)
        actual_quantized = out.quantized
        expected_quantized = expected_out.quantized
        assert_expected(actual_quantized, expected_quantized, rtol=0, atol=1e-4)

        actual_encoded_flat = out.encoded_flat
        expected_encoded_flat = expected_out.encoded_flat
        assert_expected(actual_encoded_flat, expected_encoded_flat, rtol=0, atol=1e-4)

        actual_codebook_indices = out.codebook_indices
        expected_codebook_indices = expected_out.codebook_indices
        assert_expected(actual_codebook_indices, expected_codebook_indices)

    def test_decode(self, vv, input_tensor):
        actual = vv.decode(input_tensor)
        expected = torch.tensor(
            [
                [
                    [
                        [[0.0494, 0.1049], [0.0628, 0.1731]],
                        [[0.0704, 0.1552], [0.0495, 0.2263]],
                    ],
                    [
                        [[0.0897, 0.1084], [0.1269, 0.1131]],
                        [[0.0886, 0.1265], [0.0855, 0.1139]],
                    ],
                ]
            ]
        )
        assert_expected(actual, expected, rtol=0, atol=1e-4)

    def test_tokenize(self, vv, input_tensor, test_data):
        expected = test_data[1].quantized_flat
        actual = vv.tokenize(input_tensor)
        assert_expected(actual, expected, rtol=0, atol=1e-4)

    def test_forward(self, vv, input_tensor, test_data):
        expected_decoded, expected_out = test_data
        out = vv(input_tensor)
        actual_decoded = out.decoded
        assert_expected(actual_decoded, expected_decoded, rtol=0, atol=1e-4)

        actual_quantized = out.codebook_output.quantized
        expected_quantized = expected_out.quantized
        assert_expected(actual_quantized, expected_quantized, rtol=0, atol=1e-4)

        actual_encoded_flat = out.codebook_output.encoded_flat
        expected_encoded_flat = expected_out.encoded_flat
        assert_expected(actual_encoded_flat, expected_encoded_flat, rtol=0, atol=1e-4)

        actual_codebook_indices = out.codebook_output.codebook_indices
        expected_codebook_indices = expected_out.codebook_indices
        assert_expected(actual_codebook_indices, expected_codebook_indices)
