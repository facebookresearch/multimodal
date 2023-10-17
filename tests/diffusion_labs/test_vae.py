#!/usr/bin/env fbpython
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch
import torch.distributions as tdist
from tests.test_utils import assert_expected, set_rng_seed
from torchmultimodal.diffusion_labs.models.vae.vae import variational_autoencoder


@pytest.fixture(autouse=True)
def set_seed():
    set_rng_seed(98765)


@pytest.fixture
def embedding_channels():
    return 6


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
    return (1, 2, 4)


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
def z(embedding_channels):
    bsize = 2
    height = 4
    width = 4
    return torch.randn(bsize, embedding_channels, height, width)


class TestVariationalAutoencoder:
    @pytest.fixture
    def vae(
        self,
        in_channels,
        out_channels,
        embedding_channels,
        z_channels,
        channels,
        norm_groups,
        norm_eps,
        channel_multipliers,
        num_res_blocks,
    ):
        return variational_autoencoder(
            embedding_channels=embedding_channels,
            in_channels=in_channels,
            out_channels=out_channels,
            z_channels=z_channels,
            channels=channels,
            num_res_blocks=num_res_blocks,
            channel_multipliers=channel_multipliers,
            norm_groups=norm_groups,
            norm_eps=norm_eps,
        )

    def test_encode(self, vae, x, embedding_channels, channel_multipliers):
        posterior = vae.encode(x)
        expected_shape = torch.Size(
            [
                x.size(0),
                embedding_channels,
                x.size(2) // 2 ** (len(channel_multipliers) - 1),
                x.size(3) // 2 ** (len(channel_multipliers) - 1),
            ]
        )
        expected_mean = torch.tensor(-3.4872)
        assert_expected(posterior.mean.size(), expected_shape)
        assert_expected(posterior.mean.sum(), expected_mean, rtol=0, atol=1e-4)

        expected_stddev = torch.tensor(193.3726)
        assert_expected(posterior.stddev.size(), expected_shape)
        assert_expected(posterior.stddev.sum(), expected_stddev, rtol=0, atol=1e-4)

        # compute kl with standard gaussian
        expected_kl = torch.tensor(9.8025)
        torch_kl_divergence = tdist.kl_divergence(
            posterior,
            tdist.Normal(
                torch.zeros_like(posterior.mean), torch.ones_like(posterior.stddev)
            ),
        ).sum()
        assert_expected(torch_kl_divergence, expected_kl, rtol=0, atol=1e-4)

        # compare sample shape
        assert_expected(posterior.rsample().size(), expected_shape)

    def test_decode(self, vae, z, out_channels, channel_multipliers):
        actual = vae.decode(z)
        expected = torch.tensor(-156.1534)
        expected_shape = torch.Size(
            [
                z.size(0),
                out_channels,
                z.size(2) * 2 ** (len(channel_multipliers) - 1),
                z.size(3) * 2 ** (len(channel_multipliers) - 1),
            ]
        )
        assert_expected(actual.size(), expected_shape)
        assert_expected(actual.sum(), expected, rtol=0, atol=1e-4)

    @pytest.mark.parametrize(
        "sample_posterior,expected_value", [(True, -153.6517), (False, -178.8593)]
    )
    def test_forward(self, vae, x, out_channels, sample_posterior, expected_value):
        actual = vae(x, sample_posterior=sample_posterior).decoder_output
        expected = torch.tensor(expected_value)
        expected_shape = torch.Size(
            [
                x.size(0),
                out_channels,
                x.size(2),
                x.size(3),
            ]
        )
        assert_expected(actual.size(), expected_shape)
        assert_expected(actual.sum(), expected, rtol=0, atol=1e-4)
