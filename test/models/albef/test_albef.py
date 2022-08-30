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
from torchmultimodal.models.albef.image_encoder import ALBEFVisionEncoder
from torchmultimodal.models.albef.model import (
    ALBEFModel,
    ALBEFModelWithSimilarity,
    ALBEFSimilarity,
)
from torchmultimodal.models.albef.multimodal_encoder import ALBEFMultimodalEncoder
from torchmultimodal.modules.encoders.bert_text_encoder import bert_text_encoder
from torchmultimodal.utils.common import momentum_update, remove_grad


@pytest.fixture(autouse=True)
def random():
    set_rng_seed(0)


@pytest.fixture
def vision_encoder():
    return ALBEFVisionEncoder(
        image_size=4,
        patch_size=4,
        num_hidden_layers=2,
        num_attention_heads=1,
        hidden_size=3,
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
        embed_size=2,
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
            [[1.364883, -1.003092, -0.361791], [-0.634884, 1.411830, -0.776947]],
            [[1.401580, -0.537510, -0.864071], [1.378901, -0.417473, -0.961429]],
        ]
    )
    assert_expected(albef_model_output.image_embeddings, expected, rtol=0, atol=1e-4)


def test_albef_image_embeddings_momentum(albef_model_output):
    expected = Tensor(
        [
            [[1.364883, -1.003092, -0.361791], [-0.634884, 1.411830, -0.776947]],
            [[1.401580, -0.537510, -0.864070], [1.378902, -0.417473, -0.961429]],
        ]
    )
    assert_expected(albef_model_output.image_embeddings_m, expected, rtol=0, atol=1e-4)


def test_albef_text_embeddings(albef_model_output):
    expected = Tensor(
        [
            [[-0.332726, 1.356729, -1.024002], [1.050448, -1.345235, 0.294787]],
            [[-1.098961, -0.221372, 1.320333], [1.304645, -1.125002, -0.179644]],
        ]
    )
    assert_expected(albef_model_output.text_embeddings, expected, rtol=0, atol=1e-4)


def test_albef_text_embeddings_momentum(albef_model_output):
    expected = Tensor(
        [
            [[-0.332726, 1.356729, -1.024002], [1.050448, -1.345235, 0.294787]],
            [[-1.098961, -0.221372, 1.320333], [1.304645, -1.125002, -0.179644]],
        ]
    )
    assert_expected(albef_model_output.text_embeddings_m, expected, rtol=0, atol=1e-4)


def test_albef_multimodal_embeddings(albef_model_output):
    expected = Tensor(
        [
            [[-0.100506, 1.271901, -1.171395], [1.410639, -0.618296, -0.792343]],
            [[-1.393961, 0.490451, 0.903510], [0.606909, -1.409685, 0.802776]],
        ]
    )
    assert_expected(
        albef_model_output.multimodal_embeddings, expected, rtol=0, atol=1e-4
    )


def test_albef_multimodal_embeddings_momentum(albef_model_output):
    expected = Tensor(
        [
            [[-0.100506, 1.271901, -1.171395], [1.410639, -0.618296, -0.792343]],
            [[-1.393961, 0.490451, 0.903510], [0.606909, -1.409685, 0.802776]],
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
            [-5.128132, -13.669198, -2.814691, 7.166637, 19.930466, 20.275330],
            [9.302484, 11.485555, -5.828896, -7.156259, -17.247587, -26.397799],
        ]
    )
    expected_sim_t2i = Tensor(
        [
            [12.8447, 13.8292, -15.2739, -20.3898, 26.4407, 17.8609],
            [-12.8771, -11.3956, 25.1225, 14.7973, -3.5396, 7.2677],
        ]
    )
    expected_sim_i2t_m = Tensor(
        [
            [2.0358, -13.9559, -14.8056, 5.6649, 19.6189, 7.0686],
            [4.7981, -13.0741, -18.6137, 4.6502, 18.0892, 1.2024],
        ]
    )
    expected_sim_t2i_m = Tensor(
        [
            [2.0358, 4.7981, 7.9365, -9.1906, 28.4402, 29.4093],
            [-13.9559, -13.0741, 24.3506, 17.6918, -10.5707, 0.4952],
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
        [[[1.917750, 1.748151, 0.901075]], [[-0.193372, -1.123208, 2.178921]]]
    )
    expected_text_embeds_neg = Tensor(
        [[[-0.0520, 0.4082, -1.4286]], [[0.0278, 0.7572, -1.7793]]]
    )
    expected_text_atts_neg = Tensor([[-0.5061], [0.0827]])
    assert_expected(image_embeds_neg, expected_image_embeds_neg, rtol=0, atol=1e-4)
    assert_expected(text_embeds_neg, expected_text_embeds_neg, rtol=0, atol=1e-4)
    assert_expected(text_atts_neg, expected_text_atts_neg, rtol=0, atol=1e-4)
