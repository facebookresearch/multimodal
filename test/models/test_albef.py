# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import copy
from functools import partial

import pytest
import torch
from test.test_utils import assert_expected, set_rng_seed
from torch import nn, Tensor
from torchmultimodal.models.albef import (
    ALBEFModel,
    ALBEFModelWithSimilarity,
    ALBEFSimilarity,
)
from torchmultimodal.modules.encoders.albef_multimodal_encoder import (
    ALBEFMultimodalEncoder,
)
from torchmultimodal.modules.encoders.albef_text_encoder import ALBEFTextEncoder
from torchmultimodal.modules.encoders.albef_vision_encoder import ALBEFVisionEncoder
from torchmultimodal.utils.common import momentum_update, remove_grad


@pytest.fixture
def random(autouse=True):
    set_rng_seed(0)


@pytest.fixture
def vision_encoder():
    return ALBEFVisionEncoder(
        image_size=4,
        patch_size=4,
        num_layers=2,
        num_heads=1,
        hidden_dim=3,
        mlp_dim=6,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
    )


@pytest.fixture
def text_encoder():
    return ALBEFTextEncoder(hidden_size=3, num_attention_heads=1)


@pytest.fixture
def multimodal_encoder():
    return ALBEFMultimodalEncoder(hidden_size=3, num_attention_heads=1)


@pytest.fixture
def albef_model(vision_encoder, text_encoder, multimodal_encoder):
    return ALBEFModel(
        vision_encoder,
        text_encoder,
        multimodal_encoder,
    )


class TestALBEFModel:
    @pytest.fixture
    def albef_model_output(self, albef_model):
        image = torch.randn(2, 3, 4, 4)
        text = torch.randint(10, (2, 2))
        text_atts = Tensor([[1, 1], [1, 0]])
        return albef_model(image, text, text_atts)

    @pytest.fixture(scope="class")
    def expected_shape(self):
        return torch.Size([2, 2, 3])

    def test_albef_image_embeddings(self, albef_model_output, expected_shape):
        actual = albef_model_output.image_embeddings
        assert_expected(actual.shape, expected_shape)

    def test_albef_image_embeddings_momentum(self, albef_model_output, expected_shape):
        actual = albef_model_output.image_embeddings_m
        assert_expected(actual.shape, expected_shape)

    def test_albef_text_embeddings(self, albef_model_output, expected_shape):
        actual = albef_model_output.text_embeddings
        assert_expected(actual.shape, expected_shape)

    def test_albef_text_embeddings_momentum(self, albef_model_output, expected_shape):
        actual = albef_model_output.text_embeddings_m
        assert_expected(actual.shape, expected_shape)

    def test_albef_multimodal_embeddings(self, albef_model_output, expected_shape):
        actual = albef_model_output.multimodal_embeddings
        assert_expected(actual.shape, expected_shape)

    def test_albef_multimodal_embeddings_momentum(
        self, albef_model_output, expected_shape
    ):
        actual = albef_model_output.multimodal_embeddings_m
        assert_expected(actual.shape, expected_shape)


class TestALBEFModelWithSimilarity:
    @pytest.fixture
    def albef_with_sim(self, albef_model):
        return ALBEFModelWithSimilarity(
            albef_model,
            nn.Linear(3, 2),
            nn.Linear(3, 2),
            embed_dim=2,
            queue_size=4,
        )

    def test_dequeue_and_enqueue(self, albef_with_sim):
        image_feat_m = torch.randn(2, 2)
        text_feat_m = torch.randn(2, 2)
        idx = Tensor([[2], [1]]).type(torch.long)
        albef_with_sim._dequeue_and_enqueue(image_feat_m, text_feat_m, idx)
        assert_expected(
            albef_with_sim.image_queue[:, 0:2],
            image_feat_m.T,
            rtol=0,
            atol=1e-4,
        )
        assert_expected(
            albef_with_sim.text_queue[:, 0:2], text_feat_m.T, rtol=0, atol=1e-4
        )
        assert_expected(albef_with_sim.idx_queue[:, 0:2], idx.T, rtol=0, atol=1e-4)

    def test_similarity(self, albef_with_sim):
        albef_with_sim.image_queue = torch.randn(2, 4)
        albef_with_sim.text_queue = torch.randn(2, 4)
        image_embeds = torch.randn(2, 5, 3)
        image_embeds_m = torch.randn(2, 5, 3)
        text_embeds = torch.randn(2, 7, 3)
        text_embeds_m = torch.randn(2, 7, 3)
        idx = Tensor([[2], [1]]).type(torch.long)
        output = albef_with_sim._similarity(
            image_embeds, image_embeds_m, text_embeds, text_embeds_m, idx
        )
        expected = torch.Size([2, 6])
        actual = output.sim_i2t.shape
        assert_expected(actual, expected)
        actual = output.sim_t2i.shape
        assert_expected(actual, expected)
        actual = output.sim_i2t_m.shape
        assert_expected(actual, expected)
        actual = output.sim_t2i_m.shape
        assert_expected(actual, expected)

    def test_neg_embeddings(self, albef_with_sim):
        image_embeds = torch.randn(2, 1, 3)
        text_embeds = torch.randn(2, 1, 3)
        text_atts = torch.randn(2, 1)
        similarity = ALBEFSimilarity(
            sim_i2t=torch.randn(2, 5),
            sim_t2i=torch.randn(2, 5),
            sim_i2t_m=torch.randn(2, 5),
            sim_t2i_m=torch.randn(2, 5),
        )
        (
            image_embeds_neg,
            text_embeds_neg,
            text_atts_neg,
        ) = albef_with_sim._neg_embeddings(
            image_embeds, text_embeds, text_atts, similarity
        )
        expected = torch.Size([2, 1, 3])
        actual = image_embeds_neg.shape
        assert_expected(actual, expected)
        actual = text_embeds_neg.shape
        assert_expected(actual, expected)
        expected = torch.Size([2, 1])
        actual = text_atts_neg.shape
        assert_expected(actual, expected)


def test_copy_params_momentum_models():
    model = nn.Linear(3, 2)
    model_m = copy.deepcopy(model)
    remove_grad(model_m)
    for param, param_m in zip(model.parameters(), model_m.parameters()):
        assert_expected(param, param_m, rtol=0, atol=1e-4)
        assert not param_m.requires_grad


def test_momentum_update():
    init_weight = Tensor([[1, 2, 3], [4, 5, 6]])
    init_weight_m = Tensor([[6, 5, 4], [3, 2, 1]])
    model = nn.Linear(3, 2)
    model_m = nn.Linear(3, 2)
    model.weight = nn.Parameter(init_weight)
    model_m.weight = nn.Parameter(init_weight_m)
    momentum_update(model, model_m, 0.75)
    expected_weight_m = Tensor([[4.75, 4.25, 3.75], [3.25, 2.75, 2.25]])
    assert_expected(model.weight, init_weight, rtol=0, atol=1e-4)
    assert_expected(model_m.weight, expected_weight_m, rtol=0, atol=1e-4)
