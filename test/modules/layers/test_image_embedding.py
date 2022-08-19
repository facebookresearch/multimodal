# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest

import torch
from test.test_utils import assert_expected, set_rng_seed
from torch import nn
from torchmultimodal.modules.layers.image_embedding import (
    ImageEmbeddings,
    PatchEmbeddings,
)


@pytest.fixture(autouse=True)
def random():
    set_rng_seed(0)


@pytest.fixture
def inputs():
    return torch.ones(2, 3, 2, 2)


@pytest.fixture
def hi_res_inputs():
    return torch.ones(1, 3, 4, 4)


@pytest.fixture
def mask():
    return torch.tensor([[1, 1, 0, 1], [0, 1, 1, 0]])


@pytest.fixture
def patch_embedding():
    model = PatchEmbeddings(image_size=2, patch_size=1, embed_dim=2)
    model.projection.weight = nn.Parameter(
        torch.tensor([[[[0.0]], [[1.0]], [[2.0]]], [[[3.0]], [[4.0]], [[5.0]]]])
    )
    model.projection.bias = nn.Parameter(torch.tensor([0.0, 0.0]))
    model.eval()
    return model


class TestPatchEmbeddings:
    def test_forward(self, inputs, patch_embedding):
        actual = patch_embedding(inputs)
        expected = torch.Tensor(
            [
                [[3.0, 12.0], [3.0, 12.0], [3.0, 12.0], [3.0, 12.0]],
                [[3.0, 12.0], [3.0, 12.0], [3.0, 12.0], [3.0, 12.0]],
            ]
        )
        assert_expected(actual, expected, atol=1e-4, rtol=0)


class TestImageEmbeddings:
    @pytest.fixture
    def embedding(self, patch_embedding):
        model = ImageEmbeddings(image_size=2, patch_size=1, hidden_size=2)
        model.patch_embeddings = patch_embedding
        model.eval()
        return model

    def test_forward(self, inputs, embedding):
        actual = embedding(inputs)
        expected = torch.Tensor(
            [
                [[0.0, 0.0], [3.0, 12.0], [3.0, 12.0], [3.0, 12.0], [3.0, 12.0]],
                [[0.0, 0.0], [3.0, 12.0], [3.0, 12.0], [3.0, 12.0], [3.0, 12.0]],
            ]
        )
        assert_expected(actual, expected, atol=1e-4, rtol=0)

    def test_forward_interpolate_pos_encoding(self, hi_res_inputs, embedding):
        actual = embedding(hi_res_inputs, interpolate_pos_encoding=True)
        expected = torch.Tensor(
            [
                [
                    [0.0, 0.0],
                    [3.0, 12.0],
                    [3.0, 12.0],
                    [3.0, 12.0],
                    [3.0, 12.0],
                    [3.0, 12.0],
                    [3.0, 12.0],
                    [3.0, 12.0],
                    [3.0, 12.0],
                    [3.0, 12.0],
                    [3.0, 12.0],
                    [3.0, 12.0],
                    [3.0, 12.0],
                    [3.0, 12.0],
                    [3.0, 12.0],
                    [3.0, 12.0],
                    [3.0, 12.0],
                ]
            ]
        )
        assert_expected(actual, expected, atol=1e-4, rtol=0)

    def test_forward_masked(self, inputs, mask, embedding):
        actual = embedding(inputs, image_patches_mask=mask)
        expected = torch.Tensor(
            [
                [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [3.0, 12.0], [0.0, 0.0]],
                [[0.0, 0.0], [3.0, 12.0], [0.0, 0.0], [0.0, 0.0], [3.0, 12.0]],
            ]
        )
        assert_expected(actual, expected, atol=1e-4, rtol=0)
