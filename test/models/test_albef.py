# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from functools import partial

import pytest
import torch
from test.test_utils import assert_expected, set_rng_seed
from torch import nn, Tensor
from torchmultimodal.models.albef import ALBEFModel, ALBEFSimilarity
from torchmultimodal.modules.encoders.albef_multimodal_encoder import (
    ALBEFMultimodalEncoder,
)
from torchmultimodal.modules.encoders.albef_text_encoder import ALBEFTextEncoder
from torchmultimodal.modules.encoders.albef_vision_encoder import ALBEFVisionEncoder


@pytest.fixture(autouse=True)
def vision_encoder():
    set_rng_seed(0)
    return ALBEFVisionEncoder(
        image_size=4,
        patch_size=4,
        num_layers=2,
        num_heads=1,
        hidden_dim=3,
        mlp_dim=6,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
    )


@pytest.fixture(autouse=True)
def text_encoder():
    return ALBEFTextEncoder(hidden_size=3, num_attention_heads=1)


@pytest.fixture(autouse=True)
def multimodal_encoder():
    return ALBEFMultimodalEncoder(hidden_size=3, num_attention_heads=1)


@pytest.fixture(autouse=True)
def dummy_albef_model():
    return ALBEFModel(
        nn.Linear(3, 2),
        nn.Linear(3, 2),
        nn.Linear(3, 2),
        nn.Linear(3, 2),
        nn.Linear(3, 2),
        embed_dim=2,
        queue_size=4,
        momentum=0.75,
    )


@pytest.fixture(autouse=True)
def albef_model(vision_encoder, text_encoder, multimodal_encoder):
    return ALBEFModel(
        vision_encoder,
        text_encoder,
        multimodal_encoder,
        nn.Linear(3, 2),
        nn.Linear(3, 2),
        embed_dim=2,
        queue_size=4,
    )


@pytest.fixture(autouse=True)
def albef_model_output(albef_model):
    image = torch.randn(2, 3, 4, 4)
    text = torch.randint(10, (2, 2))
    text_atts = Tensor([[1, 1], [1, 0]])
    return albef_model(image, text, text_atts)


def test_albef_image_embeddings(albef_model_output):
    expected = Tensor(
        [
            [[1.401580, -0.537510, -0.864071], [1.378901, -0.417473, -0.961429]],
            [[1.413948, -0.729806, -0.684143], [-1.313033, 1.111438, 0.201595]],
        ]
    )
    assert_expected(albef_model_output.image_embeddings, expected, rtol=0, atol=1e-4)


def test_albef_image_embeddings_momentum(albef_model_output):
    expected = Tensor(
        [
            [[1.401580, -0.537510, -0.864070], [1.378902, -0.417473, -0.961429]],
            [[1.413949, -0.729807, -0.684141], [-1.313033, 1.111438, 0.201595]],
        ]
    )
    assert_expected(albef_model_output.image_embeddings_m, expected, rtol=0, atol=1e-4)


def test_albef_text_embeddings(albef_model_output):
    expected = Tensor(
        [
            [[-1.372420, 0.981756, 0.390664], [1.305102, -1.124284, -0.180818]],
            [[-1.320019, 0.220507, 1.099512], [1.366452, -0.998831, -0.367621]],
        ]
    )
    assert_expected(albef_model_output.text_embeddings, expected, rtol=0, atol=1e-4)


def test_albef_vl_embeddings(albef_model_output):
    expected = Tensor(
        [
            [-1.398094, 0.883438, 0.514656],
            [-1.115577, 1.310528, -0.194951],
            [-0.706023, 1.414213, -0.708190],
            [-1.402520, 0.544084, 0.858435],
            [-1.402520, 0.544084, 0.858435],
            [-0.706023, 1.414213, -0.708190],
        ]
    )
    assert_expected(albef_model_output.vl_embeddings, expected, rtol=0, atol=1e-4)


def test_copy_params_momentum_models(dummy_albef_model):
    dummy_albef_model.models_m = [nn.Linear(3, 2) for _ in range(5)]
    dummy_albef_model._copy_params_momentum_models()
    for model, model_m in zip(dummy_albef_model.models, dummy_albef_model.models_m):
        for param, param_m in zip(model.parameters(), model_m.parameters()):
            assert_expected(param, param_m, rtol=0, atol=1e-4)
            assert not param_m.requires_grad


def test_dequeue_and_enqueue(dummy_albef_model):
    image_feat_m = torch.randn(2, 2)
    text_feat_m = torch.randn(2, 2)
    dummy_albef_model._dequeue_and_enqueue(image_feat_m, text_feat_m)
    assert_expected(
        dummy_albef_model.image_queue[:, 0:2], image_feat_m.T, rtol=0, atol=1e-4
    )
    assert_expected(
        dummy_albef_model.text_queue[:, 0:2], text_feat_m.T, rtol=0, atol=1e-4
    )


def test_momentum_update(dummy_albef_model):
    init_weight = Tensor([[1, 2, 3], [4, 5, 6]])
    init_weight_m = Tensor([[6, 5, 4], [3, 2, 1]])
    dummy_albef_model.vision_encoder.weight = nn.Parameter(init_weight)
    dummy_albef_model.vision_encoder_m.weight = nn.Parameter(init_weight_m)
    dummy_albef_model._momentum_update()
    expected_weight_m = Tensor([[4.75, 4.25, 3.75], [3.25, 2.75, 2.25]])
    assert_expected(
        dummy_albef_model.vision_encoder.weight, init_weight, rtol=0, atol=1e-4
    )
    assert_expected(
        dummy_albef_model.vision_encoder_m.weight, expected_weight_m, rtol=0, atol=1e-4
    )


def test_similarity(dummy_albef_model):
    dummy_albef_model.image_queue = torch.randn(2, 4)
    dummy_albef_model.text_queue = torch.randn(2, 4)
    image_feat = torch.randn(2, 2)
    text_feat = torch.randn(2, 2)
    image_feat_m = torch.randn(2, 2)
    text_feat_m = torch.randn(2, 2)
    output = dummy_albef_model._similarity(
        image_feat, text_feat, image_feat_m, text_feat_m
    )
    expected_sim_i2t = Tensor(
        [
            [6.502346, -7.115936, -2.145788, -6.376165, -24.799065, 9.196653],
            [6.027566, 0.114348, 1.448457, -1.095767, -9.806778, 4.751029],
        ]
    )
    expected_sim_t2i = Tensor(
        [
            [1.810656, 7.657638, 3.690561, -3.923022, -0.312378, -1.590274],
            [-8.485663, -23.672398, -12.058897, 11.247606, 3.449185, 0.094083],
        ]
    )
    expected_sim_i2t_m = Tensor(
        [
            [10.874029, -5.082995, -0.096355, -5.771802, -28.081427, 11.545799],
            [-28.726080, -29.685524, -21.830336, -15.685751, -10.502577, -6.253645],
        ]
    )
    expected_sim_t2i_m = Tensor(
        [
            [10.874029, -28.726080, -9.868037, 20.097771, -14.018656, 35.459442],
            [-5.082995, -29.685524, -13.870997, 15.797729, -0.453870, 9.397278],
        ]
    )
    assert_expected(output.sim_i2t, expected_sim_i2t, rtol=0, atol=1e-4)
    assert_expected(output.sim_t2i, expected_sim_t2i, rtol=0, atol=1e-4)
    assert_expected(output.sim_i2t_m, expected_sim_i2t_m, rtol=0, atol=1e-4)
    assert_expected(output.sim_t2i_m, expected_sim_t2i_m, rtol=0, atol=1e-4)


def test_neg_embeddings(dummy_albef_model):
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
    ) = dummy_albef_model._neg_embeddings(
        image_embeds, text_embeds, text_atts, similarity
    )
    expected_image_embeds_neg = Tensor(
        [[0.360331, 0.022756, -0.614475], [-1.006015, 1.046856, 0.119976]]
    ).unsqueeze(1)
    expected_text_embeds_neg = Tensor(
        [[0.560847, 0.789766, -0.013260], [0.910636, -1.927051, 0.644789]]
    ).unsqueeze(1)
    expected_text_atts_neg = Tensor([0.111645, -0.351305]).unsqueeze(1)
    assert_expected(image_embeds_neg, expected_image_embeds_neg, rtol=0, atol=1e-4)
    assert_expected(text_embeds_neg, expected_text_embeds_neg, rtol=0, atol=1e-4)
    assert_expected(text_atts_neg, expected_text_atts_neg, rtol=0, atol=1e-4)
