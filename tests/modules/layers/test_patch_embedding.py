# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest

import torch
from tests.test_utils import assert_expected, set_rng_seed
from torch import nn
from torchmultimodal.modules.layers.patch_embedding import PatchEmbeddings


@pytest.fixture(autouse=True)
def random():
    set_rng_seed(0)


@pytest.fixture
def inputs():
    return torch.ones(2, 3, 2, 2)


@pytest.fixture
def mask():
    return torch.tensor([[1, 1, 0, 1], [0, 1, 1, 0]])


class TestPatchEmbeddings:
    def _init_conv_proj(self, model):
        model.conv_projection.weight = nn.Parameter(
            torch.tensor([[[[0.0]], [[1.0]], [[2.0]]], [[[3.0]], [[4.0]], [[5.0]]]])
        )

    @pytest.fixture
    def embedding(self):
        model = PatchEmbeddings(
            image_size=2,
            patch_size=1,
            hidden_size=2,
            use_image_masking=True,
        )
        assert model.conv_projection.bias.sum().item() == 0
        self._init_conv_proj(model)
        model.eval()
        return model

    @pytest.fixture
    def embedding_patches_dropped(self):
        model = PatchEmbeddings(
            image_size=2,
            patch_size=1,
            hidden_size=2,
            use_image_masking=False,
            patch_drop_rate=0.5,
        )
        self._init_conv_proj(model)
        return model

    def test_forward(self, inputs, embedding):
        actual = embedding(inputs).embeddings
        expected = torch.Tensor(
            [
                [[0.0, 0.0], [3.0, 12.0], [3.0, 12.0], [3.0, 12.0], [3.0, 12.0]],
                [[0.0, 0.0], [3.0, 12.0], [3.0, 12.0], [3.0, 12.0], [3.0, 12.0]],
            ]
        )
        assert_expected(actual, expected, atol=1e-4, rtol=0)

    def test_forward_masked(self, inputs, mask, embedding):
        actual = embedding(inputs, image_patches_mask=mask).embeddings
        expected = torch.Tensor(
            [
                [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [3.0, 12.0], [0.0, 0.0]],
                [[0.0, 0.0], [3.0, 12.0], [0.0, 0.0], [0.0, 0.0], [3.0, 12.0]],
            ]
        )
        assert_expected(actual, expected, atol=1e-4, rtol=0)

    def test_forward_patches_dropped(self, inputs, embedding_patches_dropped):
        actual = embedding_patches_dropped(inputs).embeddings
        expected = torch.Tensor(
            [
                [[0.0, 0.0], [3.0, 12.0], [3.0, 12.0]],
                [[0.0, 0.0], [3.0, 12.0], [3.0, 12.0]],
            ]
        )
        assert_expected(actual, expected, atol=1e-4, rtol=0)

    def test_forward_rectangle_input(self):
        model = PatchEmbeddings(
            image_size=(4, 6),
            patch_size=2,
            hidden_size=2,
            use_image_masking=False,
            num_channels=1,
        )
        model.conv_projection.weight = nn.Parameter(
            torch.tensor([[[[0.0, 0.0], [0.0, 0.0]]], [[[3.0, 3.0], [3.0, 3.0]]]])
        )
        model.eval()
        actual = model(torch.ones(1, 1, 4, 6)).embeddings
        expected = torch.Tensor(
            [
                [
                    [0.0, 0.0],
                    [0.0, 12.0],
                    [0.0, 12.0],
                    [0.0, 12.0],
                    [0.0, 12.0],
                    [0.0, 12.0],
                    [0.0, 12.0],
                ],
            ]
        )
        assert_expected(actual, expected, atol=1e-4, rtol=0)

    def test_forward_no_cls(self, inputs, mask):
        embedding = PatchEmbeddings(
            image_size=2,
            patch_size=1,
            hidden_size=2,
            use_image_masking=True,
            include_cls_embed=False,
        )
        self._init_conv_proj(embedding)
        actual = embedding(inputs).embeddings
        expected = torch.Tensor(
            [
                [[3.0, 12.0], [3.0, 12.0], [3.0, 12.0], [3.0, 12.0]],
                [[3.0, 12.0], [3.0, 12.0], [3.0, 12.0], [3.0, 12.0]],
            ]
        )
        assert_expected(actual, expected, atol=1e-4, rtol=0)
