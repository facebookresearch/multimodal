# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from functools import partial

import pytest
import torch
from tests.test_utils import assert_expected, set_rng_seed
from torch import nn
from torchmultimodal.models.dalle2.adm.adm import ADM, ADMStack, ADMUNet
from torchmultimodal.models.dalle2.adm.attention_block import ADMAttentionBlock
from torchmultimodal.models.dalle2.adm.res_block import ADMResBlock


@pytest.fixture(autouse=True)
def set_seed():
    set_rng_seed(4)


@pytest.fixture
def params():
    in_channels = 3
    s = 4
    embed_dim = 6
    return in_channels, s, embed_dim


@pytest.fixture
def x(params):
    in_ch, s, _ = params
    out = torch.randn((1, in_ch, s, s), dtype=torch.float32)
    return out


@pytest.fixture
def t(params):
    time_dim = params[-1]
    return torch.randn((1, time_dim), dtype=torch.float32)


@pytest.fixture
def c(params):
    in_ch, _, embed_dim = params
    return torch.randn((1, in_ch, embed_dim), dtype=torch.float32)


class TestADM:
    @pytest.fixture
    def cond(self, params):
        embed_dim = params[-1]
        c = torch.ones(1, embed_dim)
        return {"test": c}

    @pytest.fixture
    def timestep(self):
        return torch.ones(1)

    @pytest.fixture
    def time_encoder(self, params):
        embed_dim = params[-1]

        class DummyTime(nn.Module):
            def forward(self, x):
                x = torch.tensor(x)
                return x.repeat(embed_dim).unsqueeze(0)

        return DummyTime()

    @pytest.fixture
    def unet(self):
        class DummyUNet(nn.Module):
            def forward(self, x, t, c):
                return x + t + c

        return DummyUNet()

    @pytest.fixture
    def model(self, unet, time_encoder):
        cond_proj = {"test": nn.Identity()}
        return partial(ADM, unet, time_encoder, cond_proj, cond_proj)

    def test_get_conditional_projections(self, model, params, timestep, cond):
        embed_dim = params[-1]
        adm = model(predict_variance_value=False)
        actual_res, actual_attn = adm._get_conditional_projections(timestep, cond)
        expected_res = 2 * torch.ones(1, embed_dim)
        expected_attn = torch.ones(1, 1, embed_dim)
        assert_expected(actual_res, expected_res)
        assert_expected(actual_attn, expected_attn)

    def test_forward(self, model, params, timestep, cond):
        embed_dim = params[-1]
        x = torch.ones(1, embed_dim)
        adm = model(predict_variance_value=False)
        actual = adm(x, timestep, cond)
        expected = 4 * torch.ones(1, embed_dim)
        assert_expected(actual.prediction, expected.unsqueeze(1))

    def test_cond_proj_incorrect_embed_dim(self, model, timestep, cond):
        cond["test"] = cond["test"][:, :-1]
        adm = model(predict_variance_value=False)
        with pytest.raises(ValueError):
            _ = adm._get_conditional_projections(timestep, cond)

    def test_predict_variance_value_incorrect_channel_dim_error(
        self, model, params, timestep, cond
    ):
        embed_dim = params[-1]
        x = torch.ones(1, embed_dim)
        adm = model(predict_variance_value=True)
        with pytest.raises(ValueError):
            _ = adm(x, timestep, cond)


# All expected values come after first testing the ADMUNet has the exact output
# as the corresponding UNet class in d2go, then simply forward passing
# ADMUNet with params, random seed, and initialization order in this file.
class TestADMUNet:
    @pytest.fixture
    def model(self, params):
        in_dim, _, time_dim = params
        net = ADMUNet(
            channels_per_layer=[32, 32],
            num_resize=1,
            num_res_per_layer=2,
            use_attention_for_layer=[True, True],
            dim_res_cond=time_dim,
            dim_attn_cond=time_dim,
            in_channels=in_dim,
            out_channels=in_dim,
        )
        net.eval()
        return net

    def test_forward(self, model, x, t, c):
        actual = model(x, t, c)
        expected = torch.tensor(2.3899)
        assert_expected(actual.sum(), expected, rtol=0, atol=1e-4)
        assert_expected(actual.shape, x.shape)

    def test_less_channels_attention_error(self, params):
        in_dim, _, time_dim = params
        with pytest.raises(ValueError):
            _ = ADMUNet(
                channels_per_layer=[32, 32],
                num_resize=1,
                num_res_per_layer=2,
                use_attention_for_layer=[True],
                dim_res_cond=time_dim,
                dim_attn_cond=time_dim,
                in_channels=in_dim,
                out_channels=in_dim,
            )

    def test_less_channels_resize_error(self, params):
        in_dim, _, time_dim = params
        with pytest.raises(ValueError):
            _ = ADMUNet(
                channels_per_layer=[32, 32],
                num_resize=3,
                num_res_per_layer=2,
                use_attention_for_layer=[True, True],
                dim_res_cond=time_dim,
                dim_attn_cond=time_dim,
                in_channels=in_dim,
                out_channels=in_dim,
            )


class TestADMStack:
    @pytest.fixture
    def model(self, params):
        in_dim, _, time_dim = params
        stack = ADMStack()
        stack.append(ADMResBlock(in_dim, in_dim, time_dim, norm_groups=in_dim))
        stack.append(ADMAttentionBlock(in_dim, time_dim, norm_groups=in_dim))
        # To use the else statement in ADMStack
        stack.append(nn.Identity())
        return stack

    def test_forward(self, model, x, t, c):
        actual = model(x, t, c)
        expected = torch.tensor(-11.8541)
        assert_expected(actual.sum(), expected, rtol=0, atol=1e-4)
        assert_expected(actual.shape, x.shape)
