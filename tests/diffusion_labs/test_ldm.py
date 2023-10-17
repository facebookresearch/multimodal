# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from copy import deepcopy
from functools import partial
from typing import Dict, List, Optional, Sequence, Set

import pytest
import torch
from tests.test_utils import assert_expected, set_rng_seed
from torch import nn, Tensor
from torchmultimodal.diffusion_labs.models.ldm.ldm import LDMModel, LDMUNet


@pytest.fixture(autouse=True)
def set_seed():
    set_rng_seed(54321)


@pytest.fixture
def in_channels() -> int:
    return 5


@pytest.fixture
def out_channels() -> int:
    return 6


@pytest.fixture
def model_channels() -> int:
    return 32


@pytest.fixture
def channel_multipliers() -> List[int]:
    return [2, 4]


@pytest.fixture
def attention_resolutions() -> Set[int]:
    return {2}


@pytest.fixture
def context_dim() -> int:
    return 32


@pytest.fixture
def context_dims(context_dim) -> List[int]:
    return [context_dim]


@pytest.fixture
def num_res_blocks() -> int:
    return 2


@pytest.fixture
def num_res_blocks_list() -> List[int]:
    return [2, 2]


@pytest.fixture
def num_attn_heads() -> int:
    return 4


@pytest.fixture
def num_channels_per_head() -> int:
    return 32


@pytest.fixture
def image_size() -> int:
    return 8


@pytest.fixture
def bsize() -> int:
    return 5


@pytest.fixture
def context_seqlen() -> int:
    return 5


@pytest.fixture
def max_time() -> int:
    return 20


@pytest.fixture
def coordinate_embedding_dim() -> int:
    return 10


@pytest.fixture
def embed_input_size() -> bool:
    return True


@pytest.fixture
def embed_target_size() -> bool:
    return True


@pytest.fixture
def embed_crop_params() -> bool:
    return True


@pytest.fixture
def num_coordinates(embed_input_size, embed_target_size, embed_crop_params) -> int:
    return 2 * (embed_input_size + embed_target_size + embed_crop_params)


@pytest.fixture
def pooled_text_embedding_dim() -> int:
    return 20


@pytest.fixture
def x(bsize, in_channels, image_size) -> Tensor:
    return torch.randn(bsize, in_channels, image_size, image_size)


@pytest.fixture
def context(bsize, context_seqlen, context_dim) -> List[Tensor]:
    return [torch.randn(bsize, context_seqlen, context_dim)]


@pytest.fixture
def additional_embeddings(
    bsize, pooled_text_embedding_dim, num_coordinates
) -> Dict[str, Tensor]:
    return {
        "pooled_text_embed": torch.randn(bsize, pooled_text_embedding_dim),
        "coordinates": torch.randn(bsize, num_coordinates),
    }


@pytest.fixture
def time(bsize, max_time) -> Tensor:
    return torch.randint(1, max_time, (bsize,))


@pytest.fixture
def context_keys() -> List[str]:
    return ["a", "b", "c", "d", "e"]


@pytest.fixture
def context_dict(bsize, context_seqlen, context_dim, context_keys) -> Dict[str, Tensor]:
    return {k: torch.randn(bsize, context_seqlen, context_dim) for k in context_keys}


# All expected values come after first testing that LDMUNet has the exact
# output as the corresponding class in d2go, then simply forward passing
# LDMUNet with params, random seed, and initialization order in this file.
class TestLDMUNet:
    @pytest.fixture
    def unet(
        self,
        in_channels,
        out_channels,
        model_channels,
        channel_multipliers,
        attention_resolutions,
        context_dims,
        num_res_blocks,
        num_attn_heads,
    ):
        return partial(
            LDMUNet,
            in_channels=in_channels,
            out_channels=out_channels,
            model_channels=model_channels,
            context_dims=context_dims,
            attention_resolutions=attention_resolutions,
            channel_multipliers=channel_multipliers,
            num_res_blocks_per_level=num_res_blocks,
            num_attention_heads=num_attn_heads,
        )

    def _unzero_unet_params(self, unet: LDMUNet):
        for p in unet.parameters():
            if torch.allclose(p, torch.zeros_like(p)):
                nn.init.normal_(p)

    def test_forward(self, unet, x, time, context, out_channels):
        unet_module = unet()
        self._unzero_unet_params(unet_module)
        expected_shape = torch.Size(
            [x.size()[0], out_channels, x.size()[2], x.size()[3]]
        )
        expected = torch.tensor(-4.60389)
        actual = unet_module(x, time, context)
        assert_expected(actual.size(), expected_shape)
        assert_expected(actual.mean(), expected, rtol=0, atol=1e-4)

    def test_forward_no_context(self, unet, x, time, context):
        unet_module = unet(context_dims=None)
        self._unzero_unet_params(unet_module)
        expected = torch.tensor(-1.93169)
        actual = unet_module(x, time)
        assert_expected(actual.mean(), expected, rtol=0, atol=1e-3)
        actual = unet_module(x, time, context)
        assert_expected(actual.mean(), expected, rtol=0, atol=1e-3)

    def test_forward_additional_embeddings(
        self,
        unet,
        x,
        time,
        context,
        out_channels,
        coordinate_embedding_dim,
        embed_input_size,
        embed_target_size,
        embed_crop_params,
        pooled_text_embedding_dim,
        additional_embeddings,
    ):
        unet_module = unet(
            coordinate_embedding_dim=coordinate_embedding_dim,
            embed_input_size=embed_input_size,
            embed_target_size=embed_target_size,
            embed_crop_params=embed_crop_params,
            pooled_text_embedding_dim=pooled_text_embedding_dim,
        )
        self._unzero_unet_params(unet_module)
        expected_shape = torch.Size(
            [x.size()[0], out_channels, x.size()[2], x.size()[3]]
        )
        expected = torch.tensor(2.99702)
        actual = unet_module(
            x, time, context, additional_embeddings=additional_embeddings
        )
        assert_expected(actual.size(), expected_shape)
        assert_expected(actual.mean(), expected, rtol=0, atol=1e-4)

    def test_forward_num_res_blocks_list(
        self, unet, num_res_blocks_list, x, time, context
    ):
        unet_module = unet(num_res_blocks_per_level=num_res_blocks_list)
        self._unzero_unet_params(unet_module)
        expected = torch.tensor(-4.60389)
        actual = unet_module(x, time, context)
        assert_expected(actual.mean(), expected, rtol=0, atol=1e-3)

    def test_forward_res_updown(self, unet, x, time, context):
        unet_module = unet(use_res_block_updown=True)
        self._unzero_unet_params(unet_module)
        expected = torch.tensor(-5.13025)
        actual = unet_module(x, time, context)
        assert_expected(actual.mean(), expected, rtol=0, atol=1e-4)

    def test_forward_linear_projection(self, unet, x, time, context):
        unet_module = unet(use_linear_projection_in_transformer=True)
        self._unzero_unet_params(unet_module)
        expected = torch.tensor(-4.60389)
        actual = unet_module(x, time, context)
        assert_expected(actual.mean(), expected, rtol=0, atol=1e-3)

    def test_forward_scale_shift_conditional(self, unet, x, time, context):
        unet_module = unet(scale_shift_conditional=True)
        self._unzero_unet_params(unet_module)
        expected = torch.tensor(-0.12192)
        actual = unet_module(x, time, context)
        assert_expected(actual.mean(), expected, rtol=0, atol=1e-4)

    def test_forward_heterogeneous_depths(self, unet, x, time, context):
        unet_module = unet(num_transformer_layers=[1, 2])
        self._unzero_unet_params(unet_module)
        expected = torch.tensor(3.86524)
        actual = unet_module(x, time, context)
        assert_expected(actual.mean(), expected, rtol=0, atol=1e-4)

    def test_unet_num_res_blocks_channels_mismatch_error(self, unet):
        with pytest.raises(ValueError):
            _ = unet(num_res_blocks_per_level=[1, 2, 3])

    def test_unet_norm_group_error(self, unet):
        with pytest.raises(ValueError):
            _ = unet(model_channels=17)

    def test_unet_context_dims_transformer_layers_mismatch_error(
        self, unet, context_dim
    ):
        with pytest.raises(ValueError):
            _ = unet(context_dims=[context_dim] * 2)

        with pytest.raises(ValueError):
            _ = unet(num_transformer_layers=[1, 1, 1])

    def test_unet_context_list_len_error(self, unet, x, time, context):
        unet_module = unet()
        with pytest.raises(RuntimeError):
            unet_module(x, time, context + deepcopy(context))

    def test_unet_context_dim_mismatch(self, unet, x, time, context):
        unet_module = unet()
        with pytest.raises(RuntimeError):
            unet_module(x, time, torch.cat([context[0], context[0]], dim=-1))

    def test_unet_num_heads_channels_errors(
        self,
        unet,
        num_attn_heads,
        num_channels_per_head,
    ):
        # Test when both num_attn_heads and num_channels_per_head are None
        with pytest.raises(ValueError):
            _ = unet(num_attention_heads=None, num_channels_per_attention_head=None)

        with pytest.raises(ValueError):
            _ = unet(
                num_attention_heads=num_attn_heads,
                num_channels_per_attention_head=num_channels_per_head,
            )


class TestLDMModel:
    @pytest.fixture
    def unet(self):
        class SimpleUNet(nn.Module):
            def forward(
                self,
                x: Tensor,
                t: Tensor,
                context_list: Optional[Sequence[Tensor]] = None,
                additional_embeddings: Optional[Dict[str, Tensor]] = None,
            ):
                if additional_embeddings is not None:
                    return torch.stack(
                        [t.mean() for t in additional_embeddings.values()], dim=0
                    ).sum()

                if isinstance(context_list, Sequence) and len(context_list) > 0:
                    return torch.cat(context_list, dim=1)
                else:
                    return torch.zeros(1)

        return SimpleUNet()

    @pytest.fixture
    def model(self, unet):
        return partial(LDMModel, unet=unet)

    @pytest.mark.parametrize(
        "cond_keys,expected_value",
        [([], 0.0), (["a", "b"], 26.7966), (["a", "b", "c"], 48.4302)],
    )
    def test_forward(self, model, x, time, context_dict, cond_keys, expected_value):
        ldm_model = model(cond_keys=cond_keys)
        expected = torch.tensor(expected_value)
        actual = ldm_model(x, time, context_dict)
        assert_expected(actual.prediction.sum(), expected, rtol=0, atol=1e-4)

    @pytest.mark.parametrize(
        "cond_keys,expected_value",
        [(["a"], 45.04681), (["b"], -18.25021)],
    )
    def test_forward_single_context(
        self, model, x, time, context_dict, cond_keys, expected_value
    ):
        ldm_model = model(cond_keys=cond_keys)
        expected = torch.tensor(expected_value)
        actual = ldm_model(x, time, context_dict)
        assert_expected(actual.prediction.sum(), expected, rtol=0, atol=1e-4)

    @pytest.mark.parametrize(
        "additional_cond_keys,expected_value",
        [
            (["pooled_text_embed"], -0.09103),
            (["pooled_text_embed", "coordinates"], -0.1936),
        ],
    )
    def test_forward_additional_embeddings(
        self,
        model,
        x,
        time,
        context_dict,
        additional_cond_keys,
        additional_embeddings,
        expected_value,
    ):
        ldm_model = model(additional_cond_keys=additional_cond_keys)
        expected = torch.tensor(expected_value)
        context_dict.update(additional_embeddings)
        actual = ldm_model(x, time, context_dict)
        assert_expected(actual.prediction.sum(), expected, rtol=0, atol=1e-4)

    def test_forward_context_dim_error(
        self, model, x, time, bsize, context_seqlen, context_dim
    ):
        context_dict = {
            "a": torch.randn(bsize, context_seqlen, context_dim),
            "b": torch.randn(bsize, context_dim),
        }
        with pytest.raises(RuntimeError):
            model(cond_keys=["a", "b"])(x, time, context_dict)

        # Should not raise runtime error because 'b' is not a cond key
        model(cond_keys=["a"])(x, time, context_dict)
