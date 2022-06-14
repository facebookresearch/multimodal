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
            embedding_dim=2,
        )

    def test_invalid_arg_length(self, in_channels, out_channels):
        with pytest.raises(ValueError):
            _ = VideoEncoder(in_channels[:2], out_channels, [2, 2], [1, 1], 1, 1)

    def test_invalid_in_out_channels(self, in_channels, out_channels):
        with pytest.raises(ValueError):
            _ = VideoEncoder(in_channels, out_channels, [2, 2, 2], [1, 1, 1], 1, 1)

    def test_forward(self, input_tensor, encoder):
        actual = encoder(input_tensor)
        expected = torch.tensor(
            [
                [
                    [
                        [[0.7367, 0.3994], [0.3994, 0.3890]],
                        [[0.5026, 0.1018], [1.1798, 1.2604]],
                    ],
                    [
                        [[0.4504, 0.2898], [0.2898, 0.2811]],
                        [[0.3761, 0.0409], [0.5910, 0.4858]],
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
            embedding_dim=3,
        )

    def test_invalid_arg_length(self, in_channels, out_channels):
        with pytest.raises(ValueError):
            _ = VideoDecoder(in_channels[:2], out_channels, [2, 2], [1, 1], 1, 1)

    def test_invalid_in_out_channels(self, in_channels, out_channels):
        with pytest.raises(ValueError):
            _ = VideoDecoder(in_channels, out_channels, [2, 2, 2], [1, 1, 1], 1, 1)

    def test_forward(self, input_tensor, decoder):
        actual = decoder(input_tensor)
        expected = torch.tensor(
            [
                [
                    [
                        [[-0.2382, -0.2388], [-0.2311, -0.2184]],
                        [[-0.2304, -0.2518], [-0.1024, -0.1248]],
                    ],
                    [
                        [[0.1624, 0.1575], [0.1722, 0.1427]],
                        [[0.1081, 0.0946], [0.1608, 0.0770]],
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
