# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import pytest
import torch
import torch.nn as nn
from tests.test_utils import assert_expected, init_weights_with_constant
from torchmultimodal.models.blip2.blip2 import BLIP2
from torchmultimodal.models.blip2.qformer_model import QformerForCLM
from torchmultimodal.modules.encoders.vision_transformer import VisionTransformer
from torchmultimodal.modules.layers.patch_embedding import PatchEmbeddings
from torchmultimodal.modules.layers.transformer import TransformerEncoder


@pytest.fixture
def dim_q():
    return 4


@pytest.fixture
def dim_kv():
    return 2


@pytest.fixture
def dim_feedforward():
    return 6


@pytest.fixture
def num_hidden_layers():
    return 2


@pytest.fixture
def num_heads():
    return 2


@pytest.fixture
def vocab_size():
    return 20


@pytest.fixture
def qformer_model_for_clm(
    dim_q,
    dim_kv,
    dim_feedforward,
    num_hidden_layers,
    num_heads,
    vocab_size,
):
    qformer_for_clm = QformerForCLM(
        dim_q=dim_q,
        dim_kv=dim_kv,
        dim_feedforward=dim_feedforward,
        num_heads=num_heads,
        attn_dropout=0.0,
        dropout=0.0,
        num_hidden_layers=num_hidden_layers,
        max_position_embeddings=512,
        vocab_size=vocab_size,
    )
    return qformer_for_clm


@pytest.fixture
def vit():
    embedding = PatchEmbeddings(image_size=2, patch_size=1, hidden_size=2)
    encoder = TransformerEncoder(
        n_layer=1,
        d_model=2,
        n_head=1,
        dim_feedforward=1,
        activation=nn.GELU,
        norm_first=True,
        final_layer_norm_eps=1e-5,
    )
    image_encoder = VisionTransformer(
        embeddings=embedding,
        encoder=encoder,
    )
    init_weights_with_constant(image_encoder)
    image_encoder.eval()
    return image_encoder


@pytest.fixture
def blip2(dim_q, dim_kv, qformer_model_for_clm, vit):
    blip2 = BLIP2(
        dim_q=dim_q,
        image_encoder_embedding_dim=dim_kv,
        qformer=qformer_model_for_clm,
        vision_encoder=vit,
        embedding_dim=4,
        decoder_bos_token_id=19,
    )
    init_weights_with_constant(blip2)
    blip2.eval()
    return blip2


@pytest.fixture
def attn_mask():
    return torch.Tensor([[1.0, 0.0, 1.0, 1.0], [0.0, 1.0, 1.0, 1.0]])


class TestBLIP2:
    def test_blip2(self, blip2, attn_mask):
        image = torch.ones(2, 3, 2, 2)
        input_ids = torch.ones(2, 4).long()
        output = blip2(image, input_ids, attn_mask)
        assert_expected(
            output.image_features, torch.ones([2, 32, 4]) * 0.5, rtol=0, atol=1e-4
        )
        assert_expected(
            output.text_features, torch.ones([2, 4]) * 0.5, rtol=0, atol=1e-4
        )
        assert_expected(
            output.image_embeddings, torch.ones([2, 5, 2]), rtol=0, atol=1e-4
        )
        assert_expected(
            output.prediction_scores, torch.ones([2, 4, 20]) * 5, rtol=0, atol=1e-4
        )

    def test_blip2_scripting(self, blip2, attn_mask):
        image = torch.ones(2, 3, 2, 2)
        input_ids = torch.ones(2, 4).long()
        scripted_model = torch.jit.script(blip2)
        actual = scripted_model(image, input_ids, attn_mask)
        expected = blip2(image, input_ids, attn_mask)
        assert_expected(actual, expected)
