# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch
from test.test_utils import assert_expected, set_rng_seed

from torchmultimodal.models.video_vqvae import (
    _preprocess_int_conv_params,
    AttentionResidualBlock,
    video_vqvae,
    video_vqvae_mugen,
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

    @pytest.fixture(scope="class")
    def test_data(self):
        decoded = torch.tensor(
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
        out = CodebookOutput(
            encoded_flat=torch.tensor(
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
            quantized_flat=torch.tensor(
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
            codebook_indices=torch.tensor([4, 4, 4, 4, 4, 4, 4, 4]),
            quantized=torch.tensor(
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


class TestVideoVQVAEMUGEN:
    @pytest.fixture
    def vv(self):
        def create_model(model_key):
            model = video_vqvae_mugen(pretrained_model_key=model_key)
            model.eval()
            return model

        return create_model

    @pytest.fixture
    def input_data(self):
        def create_data(seq_len):
            return torch.randn(1, 3, seq_len, 256, 256)

        return create_data

    def test_forward(self, vv, input_data):
        x = input_data(32)
        model = vv(None)
        output = model(x)
        actual = torch.tensor(output.decoded.shape)
        expected = torch.tensor((1, 3, 32, 256, 256))
        assert_expected(actual, expected)

    @pytest.mark.parametrize(
        "seq_len,expected", [(8, 132017.28125), (16, -109636.0), (32, 1193122.0)]
    )
    def test_checkpoint(self, vv, input_data, seq_len, expected):
        x = input_data(seq_len)
        model_key = f"mugen_L{seq_len}"
        model = vv(model_key)
        # ensure embed init flag is turned off
        assert model.codebook._is_embedding_init
        output = model(x)
        actual_tensor = torch.sum(output.decoded)
        expected_tensor = torch.tensor(expected)
        assert_expected(actual_tensor, expected_tensor, rtol=1e-5, atol=1e-8)


def test_preprocess_int_conv_params():
    channels = (3, 3, 3)
    kernel = 2
    stride = 1
    expected_kernel = torch.tensor(((2, 2, 2), (2, 2, 2), (2, 2, 2)))
    expected_stride = torch.tensor(((1, 1, 1), (1, 1, 1), (1, 1, 1)))
    actual_kernel, actual_stride = _preprocess_int_conv_params(channels, kernel, stride)
    actual_kernel = torch.tensor(actual_kernel)
    actual_stride = torch.tensor(actual_stride)
    assert_expected(actual_kernel, expected_kernel)
    assert_expected(actual_stride, expected_stride)

    actual_kernel = _preprocess_int_conv_params(channels, kernel_sizes=kernel)
    actual_kernel = torch.tensor(actual_kernel)
    assert_expected(actual_kernel, expected_kernel)

    actual_stride = _preprocess_int_conv_params(channels, strides=stride)
    actual_stride = torch.tensor(actual_stride)
    assert_expected(actual_stride, expected_stride)
