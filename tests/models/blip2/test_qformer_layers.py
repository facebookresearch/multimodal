# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch
from tests.test_utils import assert_expected, init_weights_with_constant, set_rng_seed
from torch import nn
from torchmultimodal.models.blip2.qformer_layers import (
    QformerEmbedding,
    QformerEncoder,
    QformerLayer,
)


@pytest.fixture(autouse=True)
def random():
    set_rng_seed(0)


class TestQformerWithMHA:
    @pytest.fixture
    def dim_q(self):
        return 4

    @pytest.fixture
    def dim_kv(self):
        return 2

    @pytest.fixture
    def dim_feedforward(self):
        return 6

    @pytest.fixture
    def cross_attention_freq(self):
        return 2

    @pytest.fixture
    def num_hidden_layers(self):
        return 2

    @pytest.fixture
    def num_heads(self):
        return 2

    @pytest.fixture()
    def input_ids(self):
        return torch.LongTensor([[0, 1], [2, 3]])

    @pytest.fixture()
    def query_embeddings(self):
        return torch.Tensor(
            [
                [
                    [0.6424, 0.6182, 0.5110, 0.7867],
                    [0.3907, 0.2057, 0.6909, 0.6334],
                ],
                [
                    [0.6904, 0.4445, 0.4336, 0.4603],
                    [0.6318, 0.1163, 0.0340, 0.6871],
                ],
            ]
        )

    @pytest.fixture
    def q(self):
        return torch.Tensor([[[1, 2, 3, 1], [4, 3, 2, 1], [1, 1, 1, 1]]])

    @pytest.fixture
    def kv(self):
        return torch.Tensor([[[3, 2], [1, 1]]])

    @pytest.fixture
    def current_key_value(self):
        return torch.Tensor(
            [
                [
                    [[8.0, 8.0], [11.0, 11.0], [5.0, 5.0]],
                    [[8.0, 8.0], [11.0, 11.0], [5.0, 5.0]],
                ]
            ]
        )

    @pytest.fixture
    def past_key_value(self):
        return torch.Tensor(
            [
                [
                    [[7.0, 7.0], [9.0, 9.0], [4.0, 4.0]],
                    [[7.0, 7.0], [9.0, 9.0], [4.0, 4.0]],
                ]
            ]
        )

    @pytest.fixture
    def past_key_values(self, past_key_value, num_hidden_layers):
        past_key_values = []
        for i in range(num_hidden_layers):
            past_key_values.append((past_key_value, past_key_value))
        return past_key_values

    @pytest.fixture
    def qformer_layer_self_attention_only(self, dim_q, dim_feedforward, num_heads):
        qformer_layer = QformerLayer(
            dim_q=dim_q,
            dim_feedforward=dim_feedforward,
            num_heads=num_heads,
            attn_dropout=0.0,
            dropout=0.0,
            has_cross_attention=False,
        )
        init_weights_with_constant(qformer_layer)
        qformer_layer.eval()
        return qformer_layer

    @pytest.fixture
    def qformer_layer_with_cross_attention(
        self,
        dim_q,
        dim_kv,
        dim_feedforward,
        num_heads,
    ):
        qformer_layer = QformerLayer(
            dim_q=dim_q,
            dim_kv=dim_kv,
            dim_feedforward=dim_feedforward,
            num_heads=num_heads,
            attn_dropout=0.0,
            dropout=0.0,
            activation=nn.ReLU,
            has_cross_attention=True,
        )
        init_weights_with_constant(qformer_layer)
        # modify query feedforward params to test cross attention case with different query lengths
        init_weights_with_constant(qformer_layer.feedforward_query, 2.0)
        init_weights_with_constant(qformer_layer.feedforward_layernorm_query, 2.0)
        qformer_layer.eval()
        return qformer_layer

    @pytest.fixture
    def qformer_encoder(
        self,
        dim_q,
        dim_kv,
        dim_feedforward,
        cross_attention_freq,
        num_hidden_layers,
        num_heads,
    ):
        qformer_encoder = QformerEncoder(
            dim_q=dim_q,
            dim_kv=dim_kv,
            dim_feedforward=dim_feedforward,
            num_heads=num_heads,
            attn_dropout=0.0,
            dropout=0.0,
            cross_attention_freq=cross_attention_freq,
            num_hidden_layers=num_hidden_layers,
        )
        init_weights_with_constant(qformer_encoder)
        qformer_encoder.eval()
        return qformer_encoder

    def test_qformer_layer_self_attention_only(
        self, qformer_layer_self_attention_only, current_key_value, past_key_value, q
    ):
        actual = qformer_layer_self_attention_only(
            q, past_key_value=(past_key_value, past_key_value), use_cache=True
        )
        expected = torch.Tensor(
            [
                [
                    [0.0955, 1.3015, 2.5076, 0.0955],
                    [2.3416, 1.4472, 0.5528, -0.3416],
                    [1.0000, 1.0000, 1.0000, 1.0000],
                ]
            ]
        )
        assert_expected(actual[0], expected, rtol=0, atol=1e-4)
        assert_expected(
            actual[1][0],
            torch.cat([past_key_value, current_key_value], dim=2),
        )
        assert_expected(
            actual[1][1],
            torch.cat([past_key_value, current_key_value], dim=2),
        )

    def test_qformer_layer_with_cross_attention_only_query(
        self,
        qformer_layer_with_cross_attention,
        current_key_value,
        past_key_value,
        q,
        kv,
    ):
        # test with query length < attn_residual.shape[1]
        actual = qformer_layer_with_cross_attention(
            q,
            kv,
            past_key_value=(past_key_value, past_key_value),
            query_length=2,
            use_cache=True,
        )
        expected = torch.Tensor(
            [
                [
                    [0.1909, 2.6030, 5.0151, 0.1909],
                    [4.6833, 2.8944, 1.1056, -0.6833],
                    [1.0000, 1.0000, 1.0000, 1.0000],
                ]
            ]
        )
        assert_expected(actual[0], expected, rtol=0, atol=1e-4)
        assert_expected(
            actual[1][0],
            torch.cat([past_key_value, current_key_value], dim=2),
        )
        assert_expected(
            actual[1][1],
            torch.cat([past_key_value, current_key_value], dim=2),
        )

    def test_qformer_layer_with_cross_attention_query_and_text(
        self,
        qformer_layer_with_cross_attention,
        current_key_value,
        past_key_value,
        q,
        kv,
    ):
        # test with query length >= attn_residual.shape[1]
        actual = qformer_layer_with_cross_attention(
            q,
            kv,
            past_key_value=(past_key_value, past_key_value),
            query_length=3,
            use_cache=True,
        )
        expected = torch.Tensor(
            [
                [
                    [0.1909, 2.6030, 5.0151, 0.1909],
                    [4.6833, 2.8944, 1.1056, -0.6833],
                    [2.0000, 2.0000, 2.0000, 2.0000],
                ]
            ]
        )
        assert_expected(actual[0], expected, rtol=0, atol=1e-4)
        assert_expected(
            actual[1][0],
            torch.cat([past_key_value, current_key_value], dim=2),
        )
        assert_expected(
            actual[1][1],
            torch.cat([past_key_value, current_key_value], dim=2),
        )

    def test_qformer_encoder(
        self,
        qformer_encoder,
        past_key_values,
        current_key_value,
        past_key_value,
        q,
        kv,
    ):
        actual = qformer_encoder(
            q, kv, past_key_values=past_key_values, query_length=2, use_cache=True
        )
        expected_hidden_state = torch.Tensor(
            [
                [
                    [0.0955, 1.3015, 2.5076, 0.0955],
                    [2.3416, 1.4472, 0.5528, -0.3416],
                    [1.0000, 1.0000, 1.0000, 1.0000],
                ]
            ]
        )
        expected_key_value = torch.Tensor(
            [
                [
                    [[5.0, 5.0], [5.0, 5.0], [5.0, 5.0]],
                    [[5.0, 5.0], [5.0, 5.0], [5.0, 5.0]],
                ]
            ]
        )
        assert_expected(actual[0], expected_hidden_state, rtol=0, atol=1e-4)
        assert_expected(
            actual[1][0][0],
            torch.cat([past_key_value, current_key_value], dim=2),
        )
        assert_expected(
            actual[1][0][1],
            torch.cat([past_key_value, current_key_value], dim=2),
        )
        assert_expected(
            actual[1][1][0],
            torch.cat([past_key_value, expected_key_value], dim=2),
        )
        assert_expected(
            actual[1][1][1],
            torch.cat([past_key_value, expected_key_value], dim=2),
        )

    def test_layer_scripting(
        self,
        qformer_layer_with_cross_attention,
        current_key_value,
        past_key_value,
        q,
        kv,
    ):
        scripted_model = torch.jit.script(qformer_layer_with_cross_attention)
        actual = scripted_model(
            q,
            kv,
            past_key_value=(past_key_value, past_key_value),
            query_length=3,
            use_cache=True,
        )
        expected = torch.Tensor(
            [
                [
                    [0.1909, 2.6030, 5.0151, 0.1909],
                    [4.6833, 2.8944, 1.1056, -0.6833],
                    [2.0000, 2.0000, 2.0000, 2.0000],
                ]
            ]
        )
        assert_expected(actual[0], expected, rtol=0, atol=1e-4)
        assert_expected(
            actual[1][0],
            torch.cat([past_key_value, current_key_value], dim=2),
        )
        assert_expected(
            actual[1][1],
            torch.cat([past_key_value, current_key_value], dim=2),
        )

    def test_encoder_scripting(
        self,
        qformer_encoder,
        past_key_values,
        current_key_value,
        past_key_value,
        q,
        kv,
    ):
        scripted_encoder = torch.jit.script(qformer_encoder)
        actual = scripted_encoder(
            q, kv, past_key_values=past_key_values, query_length=2, use_cache=True
        )
        expected = qformer_encoder(
            q, kv, past_key_values=past_key_values, query_length=2, use_cache=True
        )
        assert_expected(actual[0], expected[0])
        assert_expected(actual[1], expected[1])
        assert len(actual) == len(expected)

    @pytest.fixture
    def qformer_emb(self, dim_q):
        return QformerEmbedding(
            embedding_dim=dim_q,
            max_position_embeddings=512,
            vocab_size=20,
        )

    def test_qformer_embedding(self, input_ids, query_embeddings, qformer_emb):
        actual = qformer_emb(
            input_ids=input_ids,
            query_embeddings=query_embeddings,
        )
        expected_value = torch.Tensor(
            [
                [
                    [0.0287, -0.2175, -1.3081, 1.4969],
                    [-0.4602, -1.4116, 1.0838, 0.7880],
                    [-0.0600, 1.3838, -1.4382, 0.1144],
                    [1.1554, 0.0435, 0.3865, -1.5855],
                ],
                [
                    [1.7251, -0.5904, -0.6931, -0.4416],
                    [0.8989, -0.8530, -1.1327, 1.0868],
                    [0.8951, -1.1037, -0.8854, 1.0940],
                    [-0.0748, -0.2439, 1.5529, -1.2342],
                ],
            ]
        )
        # expected dim [bsz, num_token, embed_dim]
        assert_expected(actual, expected_value, atol=1e-4, rtol=1e-4)

    def test_qformer_embedding_empty_input_ids(
        self,
        query_embeddings,
        qformer_emb,
    ):
        actual = qformer_emb(
            query_embeddings=query_embeddings,
        )
        expected_value = torch.Tensor(
            [
                [
                    [0.0287, -0.2175, -1.3081, 1.4969],
                    [-0.4602, -1.4116, 1.0838, 0.7880],
                ],
                [
                    [1.7251, -0.5904, -0.6931, -0.4416],
                    [0.8989, -0.8530, -1.1327, 1.0868],
                ],
            ]
        )
        assert_expected(actual, expected_value, atol=1e-4, rtol=1e-4)

    def test_qformer_embedding_empty_query_embs(
        self,
        input_ids,
        qformer_emb,
    ):
        actual = qformer_emb(
            input_ids=input_ids,
        )
        expected_value = torch.Tensor(
            [
                [
                    [-0.0600, 1.3838, -1.4382, 0.1144],
                    [1.1554, 0.0435, 0.3865, -1.5855],
                ],
                [
                    [0.8951, -1.1037, -0.8854, 1.0940],
                    [-0.0748, -0.2439, 1.5529, -1.2342],
                ],
            ]
        )
        assert_expected(actual, expected_value, atol=1e-4, rtol=1e-4)

    def test_qformer_embedding_invalid_input(
        self,
        qformer_emb,
    ):
        with pytest.raises(ValueError):
            qformer_emb()

    def test_embedding_scripting(self, input_ids, qformer_emb, query_embeddings):
        scripted_emb = torch.jit.script(qformer_emb)
        actual = scripted_emb(input_ids=input_ids, query_embeddings=query_embeddings)
        assert_expected(
            actual, qformer_emb(input_ids=input_ids, query_embeddings=query_embeddings)
        )
