# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from copy import deepcopy
from functools import partial

import pytest
import torch
from tests.test_utils import assert_expected, set_rng_seed
from torch import nn
from torchmultimodal.diffusion_labs.models.ldm.spatial_transformer import (
    SpatialTransformer,
    SpatialTransformerCrossAttentionLayer,
)


@pytest.fixture(autouse=True)
def set_seed():
    set_rng_seed(54321)


@pytest.fixture
def in_channels():
    return 16


@pytest.fixture
def num_heads():
    return 2


@pytest.fixture
def num_layers():
    return 3


@pytest.fixture
def context_dim():
    return 8


@pytest.fixture
def batch_size():
    return 3


@pytest.fixture
def x(batch_size, in_channels):
    return torch.randn(batch_size, 10, in_channels)


@pytest.fixture
def x_img(batch_size, in_channels):
    return torch.randn(batch_size, in_channels, 8, 8)


@pytest.fixture
def context(batch_size, context_dim):
    return torch.randn(batch_size, 6, context_dim)


# All expected values come after first testing that SpatialTransformerCrossAttentionLayer
# has the exact output as the corresponding class in d2go, then simply
# forward passing SpatialTransformerCrossAttentionLayer with params, random seed, and
# initialization order in this file.
class TestSpatialTransformerCrossAttentionLayer:
    @pytest.fixture
    def attn(self, in_channels, num_heads):
        return partial(
            SpatialTransformerCrossAttentionLayer,
            d_model=in_channels,
            num_heads=num_heads,
        )

    def test_cross_attn_forward_with_context(self, attn, x, context_dim, context):
        attn_module = attn(context_dim=context_dim)
        actual = attn_module(x, context)
        expected = torch.tensor(46.95579)
        assert_expected(actual.sum(), expected, rtol=0, atol=1e-4)

    def test_cross_attn_forward_without_context(self, attn, in_channels, x):
        attn_module = attn(context_dim=in_channels)
        actual = attn_module(x)
        expected = torch.tensor(5.83984)
        assert_expected(actual.sum(), expected, rtol=0, atol=1e-4)

    def test_self_attn_forward(self, attn, x, context):
        attn_module = attn()
        actual = attn_module(x)
        expected = torch.tensor(-1.7353)
        assert_expected(actual.sum(), expected, rtol=0, atol=1e-4)
        actual = attn_module(x, context)
        assert_expected(actual.sum(), expected, rtol=0, atol=1e-4)


# All expected values come after first testing that SpatialTransformer
# has the exact output as the corresponding class in d2go, then simply
# forward passing SpatialTransformer with params, random seed, and
# initialization order in this file.
class TestSpatialTransformer:
    @pytest.fixture
    def transformer(self, in_channels, num_heads, num_layers):
        return partial(
            SpatialTransformer,
            in_channels=in_channels,
            num_heads=num_heads,
            num_layers=num_layers,
            norm_groups=2,
        )

    def _unzero_output_proj(self, transformer):
        """
        Output proj is initialized with zero weights due to
        fixup initialization. Change to non-zero proj weights to
        run unit tests with different input combinations.
        """
        for p in transformer.out_projection.parameters():
            nn.init.normal_(p)
        return transformer

    def test_transformer_forward_with_context(
        self, transformer, x_img, context_dim, num_layers, context
    ):
        transformer_module = self._unzero_output_proj(
            transformer(context_dims=[context_dim] * num_layers)
        )
        context_list = [deepcopy(context) for _ in range(num_layers)]
        actual = transformer_module(x_img, context_list)
        expected = torch.tensor(2401.9578)
        assert_expected(actual.sum(), expected, rtol=0, atol=1e-3)

    def test_transformer_forward_without_context(
        self, transformer, x_img, context, num_layers
    ):
        transformer_module = self._unzero_output_proj(transformer())
        expected = torch.tensor(-1634.7414)
        actual = transformer_module(x_img)
        assert_expected(actual.sum(), expected, rtol=0, atol=1e-3)
        context_list = [deepcopy(context) for _ in range(num_layers)]
        actual = transformer_module(x_img, context_list)
        assert_expected(actual.sum(), expected, rtol=0, atol=1e-3)

    def test_transformer_forward_with_auto_repeated_context(
        self, transformer, x_img, context_dim, num_layers, context
    ):
        transformer_module = self._unzero_output_proj(
            transformer(context_dims=[context_dim])
        )
        context_list = [deepcopy(context) for _ in range(num_layers)]
        actual = transformer_module(x_img, context_list)
        expected = torch.tensor(2401.9578)
        assert_expected(actual.sum(), expected, rtol=0, atol=1e-3)

    def test_context_dims_layers_mismatch(self, transformer, context_dim, num_layers):
        with pytest.raises(ValueError):
            transformer(context_dims=[context_dim] * (num_layers - 1))

    def test_forward_context_dims_layers_mismatch(
        self, transformer, context, context_dim, num_layers
    ):
        transformer_module = transformer(context_dims=[context_dim] * num_layers)
        context_list = [deepcopy(context) for _ in range(num_layers - 1)]
        with pytest.raises(RuntimeError):
            transformer_module(x_img, context_list)

    def test_transformer_forward_with_linear_proj(
        self, transformer, x_img, context_dim, num_layers, context
    ):
        transformer_module = self._unzero_output_proj(
            transformer(
                context_dims=[context_dim] * num_layers, use_linear_projections=True
            )
        )
        context_list = [deepcopy(context) for _ in range(num_layers)]
        actual = transformer_module(x_img, context_list)
        expected = torch.tensor(2401.9578)
        assert_expected(actual.sum(), expected, rtol=0, atol=1e-3)
