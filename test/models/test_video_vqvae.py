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


@pytest.fixture(scope="function")
def random():
    set_rng_seed(4)


@pytest.fixture(scope="module")
def in_channels():
    return (2, 2, 1)


@pytest.fixture(scope="module")
def out_channels():
    return (2, 2, 2)


@pytest.fixture(scope="module")
def kernel_sizes():
    return (2, 2, 2)


@pytest.fixture(scope="module")
def strides():
    return (1, 1, 1)


@pytest.fixture(scope="module")
def input_tensor():
    return 2 * torch.ones(1, 2, 2, 2, 2)


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
    @pytest.fixture(scope="function")
    def encoder(self, random, in_channels, out_channels, kernel_sizes, strides):
        return VideoEncoder(
            in_channels=in_channels[:2],
            out_channels=out_channels[:2],
            kernel_sizes=kernel_sizes[:2],
            strides=strides[:2],
            n_res_layers=1,
            embedding_dim=2,
        )

    def test_invalid_arg_length(self, in_channels, out_channels, kernel_sizes, strides):
        with pytest.raises(ValueError):
            _ = VideoEncoder(
                in_channels[:2], out_channels, kernel_sizes[:2], strides[:2], 1, 1
            )

    def test_invalid_in_out_channels(
        self, in_channels, out_channels, kernel_sizes, strides
    ):
        with pytest.raises(ValueError):
            _ = VideoEncoder(in_channels, out_channels, kernel_sizes, strides, 1, 1)

    def test_forward(self, input_tensor, encoder):
        actual = encoder(input_tensor)
        expected = torch.tensor(
            [
                [
                    [
                        [[0.1621, 0.3358], [0.2430, 0.3202]],
                        [[0.2081, 0.3358], [0.2254, 0.2343]],
                    ],
                    [
                        [[0.9593, 0.6932], [0.8883, 0.7170]],
                        [[1.0707, 0.6932], [1.0197, 0.9931]],
                    ],
                ]
            ]
        )
        assert_expected(actual, expected, rtol=0, atol=1e-4)

    def test_res_stack_length(self, encoder):
        assert len(encoder.res_stack) == 3, "res stack incorrect size"

    def test_num_convs(self, encoder):
        # Account for ReLU layers minus the last one which is removed
        assert len(encoder.convs) == 2 * 2 - 1, "incorrect number of conv layers"


class TestVideoDecoder:
    @pytest.fixture(scope="function")
    def decoder(self, random, in_channels, out_channels, kernel_sizes, strides):
        return VideoDecoder(
            in_channels=in_channels[:2],
            out_channels=out_channels[:2],
            kernel_sizes=kernel_sizes[:2],
            strides=strides[:2],
            n_res_layers=1,
            embedding_dim=2,
        )

    def test_invalid_arg_length(self, in_channels, out_channels, kernel_sizes, strides):
        with pytest.raises(ValueError):
            _ = VideoDecoder(
                in_channels[:2], out_channels, kernel_sizes[:2], strides[:2], 1, 1
            )

    def test_invalid_in_out_channels(
        self, in_channels, out_channels, kernel_sizes, strides
    ):
        with pytest.raises(ValueError):
            _ = VideoDecoder(in_channels, out_channels, kernel_sizes, strides, 1, 1)

    def test_forward(self, input_tensor, decoder):
        actual = decoder(input_tensor)
        expected = torch.tensor(
            [
                [
                    [
                        [[0.1365, 0.1426], [0.2040, 0.0228]],
                        [[0.1400, 0.0682], [0.0854, -0.0561]],
                    ],
                    [
                        [[0.2297, 0.1594], [0.2598, 0.2370]],
                        [[0.1883, 0.0922], [0.2612, 0.1610]],
                    ],
                ]
            ]
        )
        assert_expected(actual, expected, rtol=0, atol=1e-4)

    def test_res_stack_length(self, decoder):
        assert len(decoder.res_stack) == 3, "res stack incorrect size"

    def test_num_convs(self, decoder):
        # Account for ReLU layers minus the last one which is removed
        assert len(decoder.convts) == 2 * 2 - 1, "incorrect number of conv layers"
