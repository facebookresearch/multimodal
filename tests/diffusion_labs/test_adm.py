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
from torchmultimodal.diffusion_labs.models.adm_unet.adm import ADMStack, ADMUNet
from torchmultimodal.diffusion_labs.modules.layers.attention_block import (
    ADMAttentionBlock,
)
from torchmultimodal.diffusion_labs.modules.layers.res_block import ADMResBlock


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


# All expected values come after first testing the ADMUNet has the exact output
# as the corresponding UNet class in d2go, then simply forward passing
# ADMUNet with params, random seed, and initialization order in this file.
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
    def dummy_identity(self):
        class IdentityMultipleArgs(nn.Module):
            def forward(self, x, t, c):
                return x

        return IdentityMultipleArgs()

    @pytest.fixture
    def dummy_up(self):
        class DummyUp(nn.Module):
            def forward(self, x, t, c):
                embed_dim = x.shape[1]
                return x[:, : embed_dim // 2] + t + c

        return DummyUp()

    @pytest.fixture
    def model(self, time_encoder, params):
        in_dim, _, time_dim = params
        cond_proj = {"test": nn.Identity()}
        return partial(
            ADMUNet,
            channels_per_layer=[32, 32],
            num_resize=1,
            num_res_per_layer=2,
            use_attention_for_layer=[True, True],
            dim_res_cond=time_dim,
            dim_attn_cond=time_dim,
            in_channels=in_dim,
            timestep_encoder=time_encoder,
            res_cond_proj=cond_proj,
            attn_cond_proj=cond_proj,
        )

    def test_get_conditional_projections(self, model, params, timestep, cond):
        in_dim = params[0]
        embed_dim = params[-1]
        adm = model(predict_variance_value=False, out_channels=in_dim)
        actual_res, actual_attn = adm._get_conditional_projections(timestep, cond)
        expected_res = 2 * torch.ones(1, embed_dim)
        expected_attn = torch.ones(1, 1, embed_dim)
        assert_expected(actual_res, expected_res)
        assert_expected(actual_attn, expected_attn)

    def test_forward(
        self, model, params, timestep, cond, dummy_identity, dummy_up, mocker
    ):
        in_dim = params[0]
        embed_dim = params[-1]
        x = torch.ones(1, embed_dim)
        adm = model(predict_variance_value=False, out_channels=in_dim)

        adm.down = nn.Sequential(
            dummy_identity,
        )
        adm.bottleneck = dummy_identity
        adm.up = nn.Sequential(
            dummy_up,
        )

        actual = adm(x, timestep, cond)
        expected = 4 * torch.ones(1, embed_dim)
        assert_expected(actual.prediction, expected.unsqueeze(1))

    def test_cond_proj_incorrect_embed_dim(self, params, model, timestep, cond):
        in_dim = params[0]
        cond["test"] = cond["test"][:, :-1]
        adm = model(predict_variance_value=False, out_channels=in_dim)
        with pytest.raises(ValueError):
            _ = adm._get_conditional_projections(timestep, cond)

    def test_predict_variance_value_incorrect_channel_dim_error(
        self, model, params, timestep, cond, dummy_identity, dummy_up, mocker
    ):
        in_dim = params[0]
        embed_dim = params[-1]
        x = torch.ones(1, embed_dim)
        adm = model(predict_variance_value=True, out_channels=in_dim)

        adm.down = nn.Sequential(
            dummy_identity,
        )
        adm.bottleneck = dummy_identity
        adm.up = nn.Sequential(
            dummy_up,
        )

        with pytest.raises(ValueError):
            _ = adm(x, timestep, cond)

    def test_unet_forward(self, params, model, x, t, c, mocker):
        mocker.patch(
            "torchmultimodal.diffusion_labs.models.adm_unet.adm.ADMUNet._get_conditional_projections",
            return_value=(t, c),
        )
        in_dim = params[0]
        adm = model(predict_variance_value=False, out_channels=in_dim)
        adm.eval()

        actual = adm(x, t, c)
        expected = torch.tensor(1.0438)
        assert_expected(
            actual.prediction.sum(),
            expected,
            rtol=0,
            atol=1e-4,
        )
        assert_expected(actual.prediction.shape, x.shape)

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
                time_embed_dim=time_dim,
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
                time_embed_dim=time_dim,
            )


class TestADMStack:
    @pytest.fixture
    def model(self, params):
        in_dim, _, time_dim = params
        stack = ADMStack()
        stack.append_residual_block(
            ADMResBlock(in_dim, in_dim, time_dim, norm_groups=in_dim)
        )
        stack.append_attention_block(
            ADMAttentionBlock(in_dim, time_dim, norm_groups=in_dim)
        )
        stack.append_simple_block(nn.Identity())
        return stack

    def test_forward(self, model, x, t, c):
        actual = model(x, t, c)
        expected = torch.tensor(-11.8541)
        assert_expected(actual.sum(), expected, rtol=0, atol=1e-4)
        assert_expected(actual.shape, x.shape)
