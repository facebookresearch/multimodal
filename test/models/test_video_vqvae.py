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


@pytest.fixture(scope="class")
def random():
    set_rng_seed(4)


@pytest.fixture(scope="class")
def in_channels():
    return [3, 2, 1]


@pytest.fixture(scope="class")
def out_channels():
    return [2, 2, 2]


@pytest.fixture(scope="class")
def input_tensor():
    return 2 * torch.ones(1, 3, 2, 2, 2)


class TestAttentionResidualBlock:
    def test_hidden_dim_assertion(self):
        with pytest.raises(ValueError):
            _ = AttentionResidualBlock(1)

    def test_forward(self, random):
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
    @pytest.fixture(scope="class")
    def encoder(self, random, in_channels, out_channels):
        return VideoEncoder(
            in_channels=in_channels[:2],
            out_channels=out_channels[:2],
            kernel_sizes=[2, 2],
            strides=[1, 1],
            n_res_layers=1,
        )

    def test_invalid_arg_length(self, in_channels, out_channels):
        with pytest.raises(ValueError):
            _ = VideoEncoder(in_channels[:2], out_channels, [2, 2], [1, 1], 1)

    def test_invalid_in_out_channels(self, in_channels, out_channels):
        with pytest.raises(ValueError):
            _ = VideoEncoder(in_channels, out_channels, [2, 2, 2], [1, 1, 1], 1)

    def test_forward(self, input_tensor, encoder):
        actual = encoder(input_tensor)
        expected = torch.tensor(
            [
                [
                    [
                        [[0.0000, 0.3529], [0.3530, 0.3885]],
                        [[0.0000, 1.3704], [0.0000, 0.8605]],
                    ],
                    [
                        [[0.4022, 0.0000], [0.0000, 0.0000]],
                        [[0.0000, 0.0000], [1.1636, 1.7347]],
                    ],
                ]
            ]
        )
        assert_expected(actual, expected, rtol=0, atol=1e-4)

    def test_res_stack_has_attn(self, encoder):
        assert isinstance(
            encoder.res_stack[0], AttentionResidualBlock
        ), "missing attention residual block"

    def test_res_stack_length(self, encoder):
        assert len(encoder.res_stack) == 3, "res stack incorrect size"

    def test_num_convs(self, encoder):
        assert len(encoder.convs) == 2, "incorrect number of conv layers"


class TestVideoDecoder:
    @pytest.fixture(scope="class")
    def decoder(self, random, in_channels, out_channels):
        return VideoDecoder(
            in_channels=in_channels[:2],
            out_channels=out_channels[:2],
            kernel_sizes=[2, 2],
            strides=[1, 1],
            n_res_layers=1,
        )

    def test_invalid_arg_length(self, in_channels, out_channels):
        with pytest.raises(ValueError):
            _ = VideoDecoder(in_channels[:2], out_channels, [2, 2], [1, 1], 1)

    def test_invalid_in_out_channels(self, in_channels, out_channels):
        with pytest.raises(ValueError):
            _ = VideoDecoder(in_channels, out_channels, [2, 2, 2], [1, 1, 1], 1)

    def test_forward(self, input_tensor, decoder):
        actual = decoder(input_tensor)
        expected = torch.tensor(
            [
                [
                    [
                        [[-0.0279, 0.0193], [-0.0379, -0.0398]],
                        [[-0.0080, 0.0026], [-0.0063, 0.0058]],
                    ],
                    [
                        [[-0.0893, -0.1093], [-0.0916, -0.1208]],
                        [[-0.0858, -0.1119], [-0.0372, -0.0419]],
                    ],
                ]
            ]
        )
        assert_expected(actual, expected, rtol=0, atol=1e-4)

    def test_res_stack_has_attn(self, decoder):
        assert isinstance(
            decoder.res_stack[0], AttentionResidualBlock
        ), "missing attention residual block"

    def test_res_stack_length(self, decoder):
        assert len(decoder.res_stack) == 3, "res stack incorrect size"

    def test_num_convs(self, decoder):
        assert len(decoder.convts) == 2, "incorrect number of conv layers"
