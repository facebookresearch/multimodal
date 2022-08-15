# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import copy

import pytest
import torch
from test.test_utils import assert_expected, set_rng_seed
from torch import nn, Tensor
from torchmultimodal.models.albef.image_encoder import albef_image_encoder
from torchmultimodal.models.albef.model import (
    ALBEFModel,
    ALBEFModelWithSimilarity,
    ALBEFSimilarity,
)
from torchmultimodal.models.albef.multimodal_encoder import ALBEFMultimodalEncoder
from torchmultimodal.modules.encoders.text_encoder import text_encoder
from torchmultimodal.utils.common import momentum_update, remove_grad


@pytest.fixture(autouse=True)
def vision_encoder():
    set_rng_seed(0)
    return albef_image_encoder(
        image_size=4,
        patch_size=4,
        num_layers=2,
        num_heads=1,
        hidden_dim=3,
        mlp_dim=6,
    )


@pytest.fixture(autouse=True)
def text_transformer():
    return text_encoder(hidden_size=3, num_attention_heads=1)


@pytest.fixture(autouse=True)
def multimodal_encoder():
    return ALBEFMultimodalEncoder(hidden_size=3, num_attention_heads=1)


@pytest.fixture(autouse=True)
def albef_model(vision_encoder, text_transformer, multimodal_encoder):
    return ALBEFModel(
        vision_encoder,
        text_transformer,
        multimodal_encoder,
    )


@pytest.fixture(autouse=True)
def albef_with_sim(albef_model):
    return ALBEFModelWithSimilarity(
        albef_model,
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
            [[-1.3777, 0.9650, 0.4127], [0.5677, -1.4056, 0.8379]],
            [[-0.9438, -0.4402, 1.3840], [-0.7489, -0.6645, 1.4134]],
        ]
    )
    assert_expected(albef_model_output.image_embeddings, expected, rtol=0, atol=1e-4)


def test_albef_image_embeddings_momentum(albef_model_output):
    expected = Tensor(
        [
            [[-1.3777, 0.9650, 0.4127], [0.5677, -1.4056, 0.8379]],
            [[-0.9438, -0.4402, 1.3840], [-0.7489, -0.6645, 1.4134]],
        ]
    )
    assert_expected(albef_model_output.image_embeddings_m, expected, rtol=0, atol=1e-4)


def test_albef_text_embeddings(albef_model_output):
    expected = Tensor(
        [
            [[0.3589, -1.3641, 1.0052], [0.4500, -1.3861, 0.9361]],
            [[0.5198, -1.3989, 0.8791], [0.6649, -1.4134, 0.7485]],
        ]
    )
    assert_expected(albef_model_output.text_embeddings, expected, rtol=0, atol=1e-4)


def test_albef_text_embeddings_momentum(albef_model_output):
    expected = Tensor(
        [
            [[0.3589, -1.3641, 1.0052], [0.4500, -1.3861, 0.9361]],
            [[0.5198, -1.3989, 0.8791], [0.6649, -1.4134, 0.7485]],
        ]
    )
    assert_expected(albef_model_output.text_embeddings_m, expected, rtol=0, atol=1e-4)


def test_albef_multimodal_embeddings(albef_model_output):
    expected = Tensor(
        [
            [[-1.3968, 0.8900, 0.5068], [-1.4033, 0.8534, 0.5500]],
            [[-1.3705, 0.3831, 0.9874], [-1.3150, 0.2068, 1.1082]],
        ]
    )
    assert_expected(
        albef_model_output.multimodal_embeddings, expected, rtol=0, atol=1e-4
    )


def test_albef_multimodal_embeddings_momentum(albef_model_output):
    expected = Tensor(
        [
            [[-1.3968, 0.8900, 0.5068], [-1.4033, 0.8534, 0.5500]],
            [[-1.3705, 0.3831, 0.9874], [-1.3150, 0.2068, 1.1082]],
        ]
    )
    assert_expected(
        albef_model_output.multimodal_embeddings_m, expected, rtol=0, atol=1e-4
    )


def test_copy_params_momentum_models():
    model = nn.Linear(3, 2)
    model_m = copy.deepcopy(model)
    remove_grad(model_m)
    for param, param_m in zip(model.parameters(), model_m.parameters()):
        assert_expected(param, param_m, rtol=0, atol=1e-4)
        assert not param_m.requires_grad


def test_dequeue_and_enqueue(albef_with_sim):
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
    assert_expected(albef_with_sim.text_queue[:, 0:2], text_feat_m.T, rtol=0, atol=1e-4)
    assert_expected(albef_with_sim.idx_queue[:, 0:2], idx.T, rtol=0, atol=1e-4)


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


def test_similarity(albef_with_sim):
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
    expected_sim_i2t = Tensor(
        [
            [1.4238, 5.9710, 32.5735, -14.0543, -10.3691, -9.0482],
            [12.8806, 14.1899, 14.4615, -5.5730, -5.5790, -2.3470],
        ]
    )
    expected_sim_t2i = Tensor(
        [
            [-12.8780, -6.5612, -26.2800, 19.3357, 13.5714, -19.2627],
            [-12.2216, -5.3067, -25.9683, 19.8362, 12.2456, -20.2635],
        ]
    )
    expected_sim_i2t_m = Tensor(
        [
            [-14.2616, -13.2161, 1.5444, -1.4462, 0.6496, -2.3828],
            [-10.8893, -7.2885, 20.8970, -9.6587, -5.7121, -7.4140],
        ]
    )
    expected_sim_t2i_m = Tensor(
        [
            [-14.2616, -10.8893, -25.0555, 15.5605, 17.5263, -13.5244],
            [-13.2161, -7.2885, -26.3496, 18.9465, 14.3102, -18.5720],
        ]
    )
    assert_expected(output.sim_i2t, expected_sim_i2t, rtol=0, atol=1e-4)
    assert_expected(output.sim_t2i, expected_sim_t2i, rtol=0, atol=1e-4)
    assert_expected(output.sim_i2t_m, expected_sim_i2t_m, rtol=0, atol=1e-4)
    assert_expected(output.sim_t2i_m, expected_sim_t2i_m, rtol=0, atol=1e-4)


def test_neg_embeddings(albef_with_sim):
    image_embeds = torch.randn(2, 1, 3)
    text_embeds = torch.randn(2, 1, 3)
    text_atts = torch.randn(2, 1)
    similarity = ALBEFSimilarity(
        sim_i2t=torch.randn(2, 5),
        sim_t2i=torch.randn(2, 5),
        sim_i2t_m=torch.randn(2, 5),
        sim_t2i_m=torch.randn(2, 5),
    )
    image_embeds_neg, text_embeds_neg, text_atts_neg = albef_with_sim._neg_embeddings(
        image_embeds, text_embeds, text_atts, similarity
    )
    expected_image_embeds_neg = Tensor(
        [[[-0.5263, 1.7818, -1.4097]], [[0.4777, 0.1689, -1.0773]]]
    )
    expected_text_embeds_neg = Tensor(
        [[[-0.8561, -0.5563, -0.6149]], [[-0.7626, 1.4723, 1.9049]]]
    )
    expected_text_atts_neg = Tensor([[0.5052], [-1.2696]])
    assert_expected(image_embeds_neg, expected_image_embeds_neg, rtol=0, atol=1e-4)
    assert_expected(text_embeds_neg, expected_text_embeds_neg, rtol=0, atol=1e-4)
    assert_expected(text_atts_neg, expected_text_atts_neg, rtol=0, atol=1e-4)
