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
from torchmultimodal.modules.encoders.text_encoder import bert_text_encoder
from torchmultimodal.utils.common import momentum_update, remove_grad


@pytest.fixture(autouse=True)
def random():
    set_rng_seed(0)


@pytest.fixture
def vision_encoder():
    return albef_image_encoder(
        image_size=4,
        patch_size=4,
        num_layers=2,
        num_heads=1,
        hidden_dim=3,
        mlp_dim=6,
    )


@pytest.fixture
def text_transformer():
    return bert_text_encoder(hidden_size=3, num_attention_heads=1, dropout=0.0)


@pytest.fixture
def multimodal_encoder():
    return ALBEFMultimodalEncoder(hidden_size=3, num_attention_heads=1)


@pytest.fixture
def albef_model(vision_encoder, text_transformer, multimodal_encoder):
    return ALBEFModel(
        vision_encoder,
        text_transformer,
        multimodal_encoder,
    )


@pytest.fixture
def albef_with_sim(albef_model):
    return ALBEFModelWithSimilarity(
        albef_model,
        nn.Linear(3, 2),
        nn.Linear(3, 2),
        embed_dim=2,
        queue_size=4,
    )


@pytest.fixture
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
            [14.2454, 13.9994, -9.1568, 9.5412, 7.4214, -0.7148],
            [13.0532, 14.1404, -18.5745, 9.8749, 11.7299, 5.2513],
        ]
    )
    expected_sim_t2i = Tensor(
        [
            [12.6796, 10.6270, 0.6630, 27.2049, -15.4273, 4.8052],
            [14.2015, 3.5544, 4.7949, 19.3049, -13.6284, 10.2482],
        ]
    )
    expected_sim_i2t_m = Tensor(
        [
            [13.8611, 12.3967, 0.5949, 8.2271, 2.6205, -6.2077],
            [8.1197, 11.0125, -28.7517, 8.0546, 15.6209, 13.2403],
        ]
    )
    expected_sim_t2i_m = Tensor(
        [
            [13.8611, 8.1197, 2.4225, 25.0621, -15.3354, 7.2918],
            [12.3967, 11.0125, 0.3460, 27.4308, -15.3554, 4.3350],
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
        [[[0.5138, -0.5096, 0.8612]], [[0.1721, 1.7501, -1.0727]]]
    )
    expected_text_embeds_neg = Tensor(
        [[[0.6509, 0.4389, -0.1484]], [[-0.1594, -0.6870, -0.4733]]]
    )
    expected_text_atts_neg = Tensor([[0.2380], [-2.1143]])
    assert_expected(image_embeds_neg, expected_image_embeds_neg, rtol=0, atol=1e-4)
    assert_expected(text_embeds_neg, expected_text_embeds_neg, rtol=0, atol=1e-4)
    assert_expected(text_atts_neg, expected_text_atts_neg, rtol=0, atol=1e-4)
