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
    @pytest.fixture(scope="function")
    def vv(self, params):
        in_channel_dims, out_channel_dims, kernel_sizes, strides = params
        model = video_vqvae(
            encoder_in_channel_dims=in_channel_dims,
            encoder_kernel_sizes=kernel_sizes[0][0],
            encoder_strides=strides[0][0],
            n_res_layers=1,
            attn_hidden_dim=2,
            num_embeddings=8,
            embedding_dim=2,
            decoder_out_channel_dims=out_channel_dims,
            decoder_kernel_sizes=kernel_sizes[0][0],
            decoder_strides=strides[0][0],
        )
        model.decoder.eval()
        return model

    @pytest.fixture(scope="class")
    def test_data(self):
        decoded = torch.tensor(
            [
                [
                    [
                        [[0.1408, 0.1837], [0.1411, 0.1198]],
                        [[0.1134, 0.1115], [0.0588, 0.0494]],
                    ],
                    [
                        [[0.1898, 0.0849], [0.1486, -0.0408]],
                        [[0.0945, -0.1119], [0.0394, -0.0879]],
                    ],
                ]
            ]
        )
        out = CodebookOutput(
            encoded_flat=torch.tensor(
                [
                    [0.3053, -0.4401],
                    [-0.6024, -0.3483],
                    [-0.2187, -0.3871],
                    [-1.0400, 0.0456],
                    [-0.8254, -0.1475],
                    [-0.8026, -0.1680],
                    [-0.2983, -0.3790],
                    [-0.3530, -0.0553],
                ]
            ),
            quantized_flat=torch.tensor(
                [
                    [0.3053, -0.4401],
                    [-0.6024, -0.3483],
                    [-0.2187, -0.3871],
                    [-1.0400, 0.0456],
                    [-0.8254, -0.1475],
                    [-0.8026, -0.1680],
                    [-0.2983, -0.3790],
                    [-0.3530, -0.0553],
                ]
            ),
            codebook_indices=torch.tensor([3, 1, 6, 2, 7, 4, 0, 5]),
            quantized=torch.tensor(
                [
                    [
                        [
                            [[0.3053, -0.6024], [-0.2187, -1.0400]],
                            [[-0.8254, -0.8026], [-0.2983, -0.3530]],
                        ],
                        [
                            [[-0.4401, -0.3483], [-0.3871, 0.0456]],
                            [[-0.1475, -0.1680], [-0.3790, -0.0553]],
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
                        [[0.1279, 0.1485], [0.1215, 0.1099]],
                        [[0.0885, 0.1093], [0.0869, 0.0764]],
                    ],
                    [
                        [[0.1915, 0.1300], [0.1632, 0.0759]],
                        [[0.1178, 0.0366], [0.1071, 0.0526]],
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
