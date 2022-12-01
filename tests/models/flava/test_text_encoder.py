# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from functools import partial

import pytest

import torch
from tests.test_utils import assert_expected, set_rng_seed
from torch import nn
from torchmultimodal.models.flava.transformer import init_transformer_weights
from torchmultimodal.modules.layers.transformer import TransformerEncoder
from torchtext.models.bert.model import BERTTextEncoder
from torchtext.models.bert.text_embedding import BERTTextEmbeddings


@pytest.fixture(autouse=True)
def random():
    set_rng_seed(0)


class TestFlavaTextEncoder:
    @pytest.fixture
    def emb_weights(self):
        return torch.Tensor([[0, 1], [1, 0], [1, 1]])

    @pytest.fixture
    def text_encoder_components(self, emb_weights):
        text_embedding = BERTTextEmbeddings(
            hidden_size=2,
            vocab_size=3,
            max_position_embeddings=2,
            dropout=0,
        )
        text_embedding.word_embeddings = nn.Embedding.from_pretrained(emb_weights)
        text_embedding.position_embeddings = nn.Embedding.from_pretrained(emb_weights)
        text_embedding.token_type_embeddings = nn.Embedding.from_pretrained(emb_weights)
        text_embedding.eval()

        encoder = TransformerEncoder(
            n_layer=1,
            d_model=2,
            n_head=1,
            dim_feedforward=1,
            activation=nn.GELU,
            norm_first=True,
        )
        weight_init_fn = partial(init_transformer_weights, initializer_range=0.02)
        text_encoder = BERTTextEncoder(
            embeddings=text_embedding,
            encoder=encoder,
            layernorm=nn.LayerNorm(2),
            pooler=nn.Identity(),
            weight_init_fn=weight_init_fn,
        )
        return text_encoder, text_embedding

    @pytest.fixture
    def input_ids(self):
        return torch.IntTensor([[0, 1]])

    @pytest.fixture
    def attn_mask(self):
        return torch.IntTensor([[1, 0]])

    def test_embedding(self, text_encoder_components, input_ids):
        _, text_embedding = text_encoder_components
        out = text_embedding(input_ids)
        expected = torch.Tensor([[[1.0, -1.0], [-1.0, 1.0]]])
        assert_expected(out, expected)

    def test_text_transformer(self, text_encoder_components, input_ids):
        text_encoder, _ = text_encoder_components
        out = text_encoder(
            input_ids,
            return_attn_weights=True,
            return_hidden_states=True,
        )

        assert_expected(
            out.last_hidden_state, torch.Tensor([[[1.0, -1.0], [-1.0, 1.0]]])
        )

        assert_expected(
            out.hidden_states,
            (
                torch.Tensor([[[1.0000, -1.0000], [-1.0000, 1.0000]]]),
                torch.Tensor([[[1.0008, -0.9994], [-0.9997, 1.0012]]]),
            ),
            atol=1e-4,
            rtol=0.0,
        )

        assert_expected(out.attentions, (torch.Tensor([[[[0, 1.0], [0.0, 1.0]]]]),))

    def test_text_transformer_attn_mask(
        self, text_encoder_components, input_ids, attn_mask
    ):
        text_encoder, _ = text_encoder_components
        out = text_encoder(
            input_ids,
            attention_mask=attn_mask,
            return_attn_weights=True,
            return_hidden_states=True,
        )

        assert_expected(
            out.last_hidden_state, torch.Tensor([[[1.0, -1.0], [-1.0, 1.0]]])
        )

        assert_expected(
            out.hidden_states,
            (
                torch.Tensor([[[1.0, -1.0], [-1.0, 1.0]]]),
                torch.Tensor([[[0.9997, -1.0012], [-1.0008, 0.9994]]]),
            ),
            atol=1e-4,
            rtol=0.0,
        )

        assert_expected(out.pooler_output, torch.Tensor([[[1.0, -1.0], [-1.0, 1.0]]]))
        assert_expected(out.attentions, (torch.Tensor([[[[1.0, 0], [1.0, 0]]]]),))
