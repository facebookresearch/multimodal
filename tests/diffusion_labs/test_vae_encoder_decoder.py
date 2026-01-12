#!/usr/bin/env fbpython
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from functools import partial

import pytest
import torch
from tests.test_utils import assert_expected, set_rng_seed
from torchmultimodal.diffusion_labs.models.vae.encoder_decoder import (
    res_block,
    res_block_stack,
    ResNetDecoder,
    ResNetEncoder,
)


@pytest.fixture(autouse=True)
def set_seed():
    set_rng_seed(54321)


@pytest.fixture
def in_channels():
    return 2


@pytest.fixture
def out_channels():
    return 5


@pytest.fixture
def z_channels():
    return 3


@pytest.fixture
def channels():
    return 4


@pytest.fixture
def num_res_blocks():
    return 2


@pytest.fixture
def channel_multipliers():
    return (1, 2)


@pytest.fixture
def norm_groups():
    return 2


@pytest.fixture
def norm_eps():
    return 1e-05


@pytest.fixture
def x(in_channels):
    bsize = 2
    height = 16
    width = 16
    return torch.randn(bsize, in_channels, height, width)


@pytest.fixture
def z(z_channels):
    bsize = 2
    height = 4
    width = 4
    return torch.randn(bsize, z_channels, height, width)


class TestResNetEncoder:
    @pytest.fixture
    def encoder(
        self,
        in_channels,
        z_channels,
        channels,
        num_res_blocks,
        channel_multipliers,
        norm_groups,
        norm_eps,
    ):
        return partial(
            ResNetEncoder,
            in_channels=in_channels,
            z_channels=z_channels,
            channels=channels,
            num_res_blocks=num_res_blocks,
            channel_multipliers=channel_multipliers,
            norm_groups=norm_groups,
            norm_eps=norm_eps,
        )

    @pytest.mark.parametrize("double_z", [True, False])
    def test_forward_dims(self, encoder, x, z_channels, channel_multipliers, double_z):
        encoder_module = encoder(double_z=double_z)
        output = encoder_module(x)
        assert_expected(
            output.size(),
            torch.Size(
                [
                    x.size(0),
                    z_channels * (2 if double_z else 1),
                    x.size(2) // 2 ** (len(channel_multipliers) - 1),
                    x.size(3) // 2 ** (len(channel_multipliers) - 1),
                ]
            ),
        )

    def test_forward(self, encoder, x):
        encoder_module = encoder()
        actual = encoder_module(x)
        expected = torch.tensor(126.5277)
        assert_expected(actual.sum(), expected, rtol=0, atol=1e-4)

    def test_channel_indivisble_norm_group_error(self, encoder):
        with pytest.raises(ValueError):
            _ = encoder(norm_groups=7)


class TestResNetDecoder:
    @pytest.fixture
    def decoder(
        self,
        out_channels,
        z_channels,
        channels,
        num_res_blocks,
        channel_multipliers,
        norm_groups,
        norm_eps,
    ):
        return partial(
            ResNetDecoder,
            out_channels=out_channels,
            z_channels=z_channels,
            channels=channels,
            num_res_blocks=num_res_blocks,
            channel_multipliers=channel_multipliers,
            norm_groups=norm_groups,
            norm_eps=norm_eps,
        )

    @pytest.mark.parametrize("output_alpha_channel", [True, False])
    def test_forward_dims(
        self, decoder, z, out_channels, channel_multipliers, output_alpha_channel
    ):
        decoder_module = decoder(output_alpha_channel=output_alpha_channel)
        output = decoder_module(z)
        assert_expected(
            output.size(),
            torch.Size(
                [
                    z.size(0),
                    out_channels + (1 if output_alpha_channel else 0),
                    z.size(2) * 2 ** (len(channel_multipliers) - 1),
                    z.size(3) * 2 ** (len(channel_multipliers) - 1),
                ]
            ),
        )

    def test_forward(self, decoder, z):
        decoder_module = decoder()
        actual = decoder_module(z)
        expected = torch.tensor(-10.0260)
        assert_expected(actual.sum(), expected, rtol=0, atol=1e-4)

    def test_forward_alpha_channel(self, decoder, z):
        decoder_module = decoder(output_alpha_channel=True)
        actual = decoder_module(z)
        expected = torch.tensor(-16.2157)
        assert_expected(actual.sum(), expected, rtol=0, atol=1e-4)

    def test_channel_indivisble_norm_group_error(self, decoder):
        with pytest.raises(ValueError):
            _ = decoder(norm_groups=7)


@pytest.mark.parametrize("out_channels,expected_value", [(2, 52.2716), (4, 152.8285)])
def test_res_block(x, out_channels, expected_value):
    in_channels = x.size(1)
    res_block_module = res_block(in_channels, out_channels, dropout=0.3, norm_groups=1)
    actual = res_block_module(x)
    expected = torch.tensor(expected_value)
    assert_expected(
        actual.size(),
        torch.Size(
            [
                x.size(0),
                out_channels,
                x.size(2),
                x.size(3),
            ]
        ),
    )
    assert_expected(actual.sum(), expected, rtol=0, atol=1e-4)


@pytest.mark.parametrize(
    "needs_upsample,needs_downsample,expected_value",
    [(False, True, 28.02428), (False, False, 382.8569), (True, False, 581.62414)],
)
def test_res_block_stack(
    x,
    in_channels,
    channels,
    num_res_blocks,
    needs_upsample,
    needs_downsample,
    expected_value,
):
    res_block_stack_module = res_block_stack(
        in_channels=in_channels,
        out_channels=channels,
        num_blocks=num_res_blocks,
        dropout=0.1,
        needs_upsample=needs_upsample,
        needs_downsample=needs_downsample,
        norm_groups=1,
    )
    actual = res_block_stack_module(x)
    expected = torch.tensor(expected_value)
    if needs_upsample:
        size_multipler = 2
    elif needs_downsample:
        size_multipler = 0.5
    else:
        size_multipler = 1
    assert_expected(
        actual.size(),
        torch.Size(
            [
                x.size(0),
                channels,
                int(x.size(2) * size_multipler),
                int(x.size(3) * size_multipler),
            ]
        ),
    )
    assert_expected(actual.sum(), expected, rtol=0, atol=1e-4)


def test_res_block_stack_exception(
    in_channels,
    channels,
    num_res_blocks,
):
    with pytest.raises(ValueError):
        _ = res_block_stack(
            in_channels=in_channels,
            out_channels=channels,
            num_blocks=num_res_blocks,
            needs_upsample=True,
            needs_downsample=True,
        )
