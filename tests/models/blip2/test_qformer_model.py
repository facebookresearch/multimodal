# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch
from tests.test_utils import assert_expected, init_weights_with_constant, set_rng_seed
from torch.nn import CrossEntropyLoss
from torchmultimodal.models.blip2.qformer_model import QformerForCLM, QformerModel


@pytest.fixture(autouse=True)
def random():
    set_rng_seed(0)


class TestQformerModel:
    @pytest.fixture
    def attn_mask(self):
        return torch.Tensor([[1.0, 0.0, 1.0, 1.0], [0.0, 1.0, 1.0, 1.0]])

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

    @pytest.fixture
    def vocab_size(self):
        return 20

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
    def past_key_value(self):
        return torch.Tensor(
            [
                [
                    [[7.0, 7.0], [9.0, 9.0], [4.0, 4.0]],
                    [[7.0, 7.0], [9.0, 9.0], [4.0, 4.0]],
                ],
                [
                    [[7.0, 7.0], [9.0, 9.0], [4.0, 4.0]],
                    [[7.0, 7.0], [9.0, 9.0], [4.0, 4.0]],
                ],
            ]
        )

    @pytest.fixture
    def past_key_values(self, past_key_value, num_hidden_layers):
        past_key_values = []
        for i in range(num_hidden_layers):
            past_key_values.append((past_key_value, past_key_value))
        return past_key_values

    @pytest.fixture
    def kv(self):
        return torch.Tensor([[[3, 2], [1, 1]], [[3, 2], [1, 1]]])

    @pytest.fixture
    def labels(self):
        labels = torch.ones([2, 2]).long()
        return labels[:, 1:].contiguous()

    @pytest.fixture
    def loss_fct(self):
        return CrossEntropyLoss(reduction="mean", label_smoothing=0.1)

    @pytest.fixture
    def qformer_model(
        self,
        dim_q,
        dim_kv,
        dim_feedforward,
        cross_attention_freq,
        num_hidden_layers,
        num_heads,
        vocab_size,
    ):
        qformer_model = QformerModel(
            dim_q=dim_q,
            dim_kv=dim_kv,
            dim_feedforward=dim_feedforward,
            num_heads=num_heads,
            attn_dropout=0.0,
            dropout=0.0,
            num_hidden_layers=num_hidden_layers,
            max_position_embeddings=512,
            vocab_size=vocab_size,
            query_length=2,
        )
        init_weights_with_constant(qformer_model)
        qformer_model.eval()
        return qformer_model

    @pytest.fixture
    def qformer_model_for_clm(
        self,
        dim_q,
        dim_kv,
        dim_feedforward,
        cross_attention_freq,
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
        init_weights_with_constant(qformer_for_clm)
        qformer_for_clm.eval()
        return qformer_for_clm

    def test_qformer_model_with_attn_mask(
        self,
        input_ids,
        attn_mask,
        qformer_model,
        query_embeddings,
        num_hidden_layers,
        kv,
    ):
        actual = qformer_model(
            input_ids=input_ids,
            encoder_hidden_states=kv,
            attention_mask=attn_mask,
            query_embeds=query_embeddings,
            use_cache=True,
        )
        expected_hidden_states = torch.Tensor(
            [
                [
                    [1.0287, 0.7825, -0.3081, 2.4969],
                    [0.5398, -0.4116, 2.0838, 1.7880],
                    [1.0000, 1.0000, 1.0000, 1.0000],
                    [1.0000, 1.0000, 1.0000, 1.0000],
                ],
                [
                    [2.7251, 0.4096, 0.3069, 0.5584],
                    [1.8989, 0.1470, -0.1327, 2.0868],
                    [1.0000, 1.0000, 1.0000, 1.0000],
                    [1.0000, 1.0000, 1.0000, 1.0000],
                ],
            ]
        )
        assert_expected(actual[0], expected_hidden_states, atol=1e-4, rtol=1e-4)

        assert_expected(len(actual[1]), num_hidden_layers)
        assert_expected(len(actual[1][0]), 2)  # 2-element tuple includes key and value
        assert_expected(
            actual[1][0][0].shape, torch.Size([2, 2, 4, 2])
        )  # bsz x num_heads x seq_len x head_dim
        expected_cached_values = torch.Tensor(
            [
                [
                    [
                        [5.0000, 5.0000],
                        [5.0000, 5.0000],
                        [5.0000, 5.0000],
                        [5.0000, 5.0000],
                    ],
                    [
                        [5.0000, 5.0000],
                        [5.0000, 5.0000],
                        [5.0000, 5.0000],
                        [5.0000, 5.0000],
                    ],
                ],
                [
                    [
                        [5.0000, 5.0000],
                        [5.0000, 5.0000],
                        [5.0000, 5.0000],
                        [5.0000, 5.0000],
                    ],
                    [
                        [5.0000, 5.0000],
                        [5.0000, 5.0000],
                        [5.0000, 5.0000],
                        [5.0000, 5.0000],
                    ],
                ],
            ]
        )
        assert_expected(actual[1][0][0], expected_cached_values, atol=1e-4, rtol=1e-4)

    def test_qformer_model_with_past_key_values(
        self,
        input_ids,
        qformer_model,
        query_embeddings,
        num_hidden_layers,
        kv,
        past_key_values,
    ):
        actual = qformer_model(
            input_ids=input_ids,
            encoder_hidden_states=kv,
            query_embeds=query_embeddings,
            past_key_values=past_key_values,
            use_cache=True,
        )
        expected_hidden_states = torch.Tensor(
            [
                [
                    [1.0287, 0.7825, -0.3081, 2.4969],
                    [0.5398, -0.4116, 2.0838, 1.7880],
                    [1.0000, 1.0000, 1.0000, 1.0000],
                    [1.0000, 1.0000, 1.0000, 1.0000],
                ],
                [
                    [2.7251, 0.4096, 0.3069, 0.5584],
                    [1.8989, 0.1470, -0.1327, 2.0868],
                    [1.0000, 1.0000, 1.0000, 1.0000],
                    [1.0000, 1.0000, 1.0000, 1.0000],
                ],
            ]
        )
        assert_expected(actual[0], expected_hidden_states, atol=1e-4, rtol=1e-4)

        assert_expected(len(actual[1]), num_hidden_layers)
        assert_expected(len(actual[1][0]), 2)  # 2-element tuple includes key and value
        assert_expected(
            actual[1][0][0].shape, torch.Size([2, 2, 7, 2])
        )  # bsz x num_heads x seq_len x head_dim
        expected_cached_values = torch.Tensor(
            [
                [
                    [
                        [7.0, 7.0],
                        [9.0, 9.0],
                        [4.0, 4.0],
                        [5.0, 5.0],
                        [5.0, 5.0],
                        [5.0, 5.0],
                        [5.0, 5.0],
                    ],
                    [
                        [7.0, 7.0],
                        [9.0, 9.0],
                        [4.0, 4.0],
                        [5.0, 5.0],
                        [5.0, 5.0],
                        [5.0, 5.0],
                        [5.0, 5.0],
                    ],
                ],
                [
                    [
                        [7.0, 7.0],
                        [9.0, 9.0],
                        [4.0, 4.0],
                        [5.0, 5.0],
                        [5.0, 5.0],
                        [5.0, 5.0],
                        [5.0, 5.0],
                    ],
                    [
                        [7.0, 7.0],
                        [9.0, 9.0],
                        [4.0, 4.0],
                        [5.0, 5.0],
                        [5.0, 5.0],
                        [5.0, 5.0],
                        [5.0, 5.0],
                    ],
                ],
            ]
        )
        assert_expected(actual[1][0][0], expected_cached_values, atol=1e-4, rtol=1e-4)

    def test_qformer_model_with_causal_mask(
        self,
        input_ids,
        attn_mask,
        kv,
        qformer_model,
        query_embeddings,
        num_hidden_layers,
    ):
        actual = qformer_model(
            input_ids=input_ids,
            encoder_hidden_states=kv,
            attention_mask=attn_mask,
            query_embeds=query_embeddings,
            use_cache=True,
            use_causal_mask=True,
        )
        expected_hidden_states = torch.Tensor(
            [
                [
                    [1.0287, 0.7825, -0.3081, 2.4969],
                    [0.5398, -0.4116, 2.0838, 1.7880],
                    [1.0000, 1.0000, 1.0000, 1.0000],
                    [1.0000, 1.0000, 1.0000, 1.0000],
                ],
                [
                    [2.7251, 0.4096, 0.3069, 0.5584],
                    [1.8989, 0.1470, -0.1327, 2.0868],
                    [1.0000, 1.0000, 1.0000, 1.0000],
                    [1.0000, 1.0000, 1.0000, 1.0000],
                ],
            ]
        )
        assert_expected(actual[0], expected_hidden_states, atol=1e-4, rtol=1e-4)

    def test_qformer_model_scripting(
        self, qformer_model, input_ids, attn_mask, query_embeddings, kv
    ):
        scripted_model = torch.jit.script(qformer_model)
        scripted_output = scripted_model(
            input_ids=input_ids,
            encoder_hidden_states=kv,
            attention_mask=attn_mask,
            query_embeds=query_embeddings,
            use_cache=True,
        )
        actual = qformer_model(
            input_ids=input_ids,
            encoder_hidden_states=kv,
            attention_mask=attn_mask,
            query_embeds=query_embeddings,
            use_cache=True,
        )
        assert_expected(scripted_output[0], actual[0], atol=1e-4, rtol=1e-4)
        assert_expected(scripted_output[1], actual[1], atol=1e-4, rtol=1e-4)

    def test_qformer_for_clm(
        self,
        qformer_model_for_clm,
        query_embeddings,
        input_ids,
        kv,
        attn_mask,
        labels,
        loss_fct,
        vocab_size,
    ):
        actual_pred = qformer_model_for_clm(
            input_ids=input_ids,
            encoder_hidden_states=kv,
            attention_mask=attn_mask,
            query_embeds=query_embeddings,
            use_cache=False,
        )
        expected = torch.ones([2, 2, 20]) * 5
        assert_expected(actual_pred, expected, atol=1e-4, rtol=1e-4)

    def test_qformer_for_clm_scripting(
        self,
        qformer_model_for_clm,
        query_embeddings,
        input_ids,
        kv,
        attn_mask,
        labels,
        loss_fct,
        vocab_size,
    ):
        scripted_model = torch.jit.script(qformer_model_for_clm)
        actual_pred = scripted_model(
            input_ids=input_ids,
            encoder_hidden_states=kv,
            attention_mask=attn_mask,
            query_embeds=query_embeddings,
            use_cache=False,
        )
        expected = torch.ones([2, 2, 20]) * 5
        assert_expected(actual_pred, expected, atol=1e-4, rtol=1e-4)
