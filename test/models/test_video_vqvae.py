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
        model.eval()
        return model

    @pytest.fixture(scope="class")
    def test_data(self):
        decoded = torch.tensor(
            [
                [
                    [
                        [[0.0544, 0.1371], [0.0670, 0.2542]],
                        [[0.0806, 0.1892], [0.0287, 0.3012]],
                    ],
                    [
                        [[0.0912, 0.1367], [0.1465, 0.1812]],
                        [[0.0858, 0.1422], [0.0855, 0.1091]],
                    ],
                ]
            ]
        )
        out = CodebookOutput(
            encoded_flat=torch.tensor(
                [
                    [0.2843, 0.8453],
                    [0.2720, 0.8817],
                    [0.2920, 0.8226],
                    [0.2692, 0.8899],
                    [0.2785, 0.8626],
                    [0.2668, 0.8972],
                    [0.2994, 0.8008],
                    [0.2639, 0.9057],
                ]
            ),
            quantized_flat=torch.tensor(
                [
                    [1.1638, 0.5075],
                    [1.1638, 0.5075],
                    [1.1638, 0.5075],
                    [1.1638, 0.5075],
                    [1.1638, 0.5075],
                    [1.1638, 0.5075],
                    [1.1638, 0.5075],
                    [1.1638, 0.5075],
                ]
            ),
            codebook_indices=torch.tensor([3, 3, 3, 3, 3, 3, 3, 3]),
            quantized=torch.tensor(
                [
                    [
                        [
                            [[1.1638, 1.1638], [1.1638, 1.1638]],
                            [[1.1638, 1.1638], [1.1638, 1.1638]],
                        ],
                        [
                            [[0.5075, 0.5075], [0.5075, 0.5075]],
                            [[0.5075, 0.5075], [0.5075, 0.5075]],
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
                        [[0.0570, 0.1541], [0.0692, 0.2967]],
                        [[0.0857, 0.2083], [0.0173, 0.3403]],
                    ],
                    [
                        [[0.0920, 0.1517], [0.1568, 0.2194]],
                        [[0.0867, 0.1550], [0.0895, 0.1156]],
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
