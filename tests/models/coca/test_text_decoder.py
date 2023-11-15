# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch
from tests.test_utils import assert_expected, init_weights_with_constant, set_rng_seed
from torch import nn, Tensor
from torchmultimodal.models.coca.text_decoder import CoCaTextDecoder, CoCaTextEmbeddings


class TestCoCaTextEmbeddings:
    @pytest.fixture
    def vocab_size(self) -> int:
        return 15

    @pytest.fixture
    def num_positions(self) -> int:
        return 7

    @pytest.fixture
    def embedding_dim(self) -> int:
        return 10

    @pytest.fixture
    def batch_size(self) -> int:
        return 2

    @pytest.fixture
    def text_embeddings(self, vocab_size, num_positions, embedding_dim):
        embeddings = CoCaTextEmbeddings(
            vocab_size=vocab_size,
            num_positions=num_positions,
            embedding_dim=embedding_dim,
            pad_idx=1,
        )
        init_weights_with_constant(embeddings)

        # Set CLS embedding and token embeddings to ranges
        embeddings.cls_embedding = nn.Parameter(
            torch.arange(embeddings.cls_embedding.shape[0], dtype=torch.float)
        )
        embeddings.token_embeddings.weight = nn.Parameter(
            torch.arange(vocab_size, dtype=torch.float)
            .unsqueeze(1)
            .expand_as(embeddings.token_embeddings.weight)
        )

        return embeddings

    @pytest.fixture
    def inputs(self):
        return torch.LongTensor([[4, 5, 6, 7, 0, 1, 2], [11, 12, 13, 14, 0, 2, 1]])

    @pytest.fixture
    def expected(self, inputs, embedding_dim, batch_size):
        embeds = (inputs[:, :-1] + 1).unsqueeze(-1).repeat(1, 1, embedding_dim)
        cls_embeds = (
            torch.arange(1, embedding_dim + 1)
            .unsqueeze(0)
            .unsqueeze(0)
            .repeat(batch_size, 1, 1)
        )
        expected = torch.cat([embeds, cls_embeds], dim=1).to(dtype=torch.float)
        return expected

    def test_coca_text_embeddings(self, inputs, text_embeddings, expected):
        actual = text_embeddings(inputs[:, :-1])
        assert_expected(actual, expected)


class TestCoCaTextDecoder:
    @pytest.fixture
    def batch_size(self):
        return 2

    @pytest.fixture
    def embedding_dim(self):
        return 8

    @pytest.fixture
    def vocab_size(self):
        return 12

    @pytest.fixture
    def num_positions(self):
        return 6

    @pytest.fixture
    def default_pad_idx(self):
        return 0

    @pytest.fixture
    def output_dim(self):
        return 3

    @pytest.fixture
    def get_text_decoder(
        self,
        batch_size,
        vocab_size,
        num_positions,
        embedding_dim,
        output_dim,
    ):
        def create_text_decoder(
            pad_idx: int = 0,
            embed_cls: bool = True,
            custom_init: bool = False,
            num_positions: int = num_positions,
        ):
            decoder = CoCaTextDecoder(
                vocab_size=vocab_size,
                num_positions=num_positions,
                embedding_dim=embedding_dim,
                n_layer=2,
                n_head=2,
                dim_feedforward=4 * embedding_dim,
                output_dim=output_dim,
                pad_idx=pad_idx,
                embed_cls=embed_cls,
            )

            init_weights_with_constant(decoder)
            if custom_init:
                set_rng_seed(0)
                nn.init.normal_(decoder.embeddings.token_embeddings.weight)
                nn.init.normal_(decoder.text_projection.weight)
                for block in decoder.transformer_decoder.layer:
                    nn.init.normal_(block.attention.q_proj.weight)
                    nn.init.normal_(block.attention.k_proj.weight)
                    nn.init.normal_(block.attention.v_proj.weight)
                    nn.init.normal_(block.attention.output_proj.weight)
                if decoder.embeddings.cls_embedding is not None:
                    nn.init.normal_(decoder.embeddings.cls_embedding)

            decoder.eval()
            return decoder

        return create_text_decoder

    @pytest.fixture
    def input_ids(self):
        return torch.LongTensor([[2, 4, 5, 7, 9, 1], [6, 8, 1, 0, 0, 0]])

    @pytest.fixture
    def padding_mask(self, input_ids, default_pad_idx):
        return input_ids != default_pad_idx

    @pytest.fixture
    def expected(self, batch_size, output_dim, num_positions, embedding_dim):
        return (
            Tensor([8]).repeat(batch_size, output_dim),
            Tensor([726]).repeat(batch_size, num_positions - 1, embedding_dim),
        )

    @pytest.fixture
    def expected_attention_mask(self, batch_size, num_positions):
        return torch.BoolTensor(
            [
                [
                    [
                        [True, False, False, False, False, False, False],
                        [True, True, False, False, False, False, False],
                        [True, True, True, False, False, False, False],
                        [True, True, True, True, False, False, False],
                        [True, True, True, True, True, False, False],
                        [True, True, True, True, True, True, False],
                        [True, True, True, True, True, True, True],
                    ]
                ],
                [
                    [
                        [True, False, False, False, False, False, False],
                        [True, True, False, False, False, False, False],
                        [True, True, True, False, False, False, False],
                        [True, True, True, True, False, False, False],
                        [True, True, True, True, True, False, False],
                        [True, True, True, True, True, True, False],
                        [True, True, True, True, False, False, False],
                    ],
                ],
            ]
        )

    @pytest.mark.parametrize(
        "pad_idx, embed_cls, expected_pooled, expected_tokens_shape, expected_tokens_mean",
        [
            (
                0,
                True,
                torch.Tensor([[5.5019, -4.5114, 3.0416], [3.4487, -6.2877, 3.1439]]),
                torch.Size([2, 5, 8]),
                torch.Tensor(
                    [
                        [585.0038, 587.7021, 588.5288, 585.5997, 588.6697],
                        [586.2949, 585.1484, 588.0995, 590.9081, 591.0029],
                    ]
                ),
            ),
            (
                None,
                True,
                torch.Tensor([[5.5019, -4.5114, 3.0416], [3.4142, -6.3097, 3.1282]]),
                torch.Size([2, 5, 8]),
                torch.Tensor(
                    [
                        [585.0038, 587.7021, 588.5288, 585.5997, 588.6697],
                        [586.2949, 585.1484, 588.0995, 590.9081, 591.0029],
                    ]
                ),
            ),
            (
                None,
                False,
                torch.Tensor([[5.8831, -4.7312, 3.1304], [4.3524, -5.0214, 2.7832]]),
                torch.Size([2, 6, 8]),
                torch.Tensor(
                    [
                        [1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000],
                        [1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000],
                    ]
                ),
            ),
        ],
    )
    def test_coca_text_decoder(
        self,
        input_ids,
        get_text_decoder,
        pad_idx,
        embed_cls,
        expected_pooled,
        expected_tokens_shape,
        expected_tokens_mean,
    ):
        text_decoder = get_text_decoder(
            pad_idx=pad_idx, embed_cls=embed_cls, custom_init=True
        )
        actual_pooled, actual_tokens = text_decoder(input_ids)
        assert_expected(actual_pooled, expected_pooled, rtol=0, atol=1e-3)
        assert_expected(actual_tokens.size(), expected_tokens_shape)
        assert_expected(
            actual_tokens.mean(dim=-1), expected_tokens_mean, rtol=0, atol=1e-3
        )

    def test_coca_text_decoder_with_padding_mask(
        self, input_ids, padding_mask, get_text_decoder, expected
    ):
        text_decoder = get_text_decoder()
        actual = text_decoder(input_ids)
        actual_with_padding_mask = text_decoder(input_ids, padding_mask)
        assert_expected(actual, expected)
        assert_expected(actual_with_padding_mask, expected)

    def test_build_attention_mask(
        self,
        num_positions,
        input_ids,
        get_text_decoder,
        padding_mask,
        expected_attention_mask,
    ):
        # Since embed_cls is True the mask will contain an extra token
        text_decoder = get_text_decoder(num_positions=num_positions + 1)

        inferred_padding_mask = text_decoder.build_mask(input_ids)
        explicit_padding_mask = text_decoder.build_mask(input_ids, padding_mask)
        assert_expected(inferred_padding_mask, expected_attention_mask)
        assert_expected(explicit_padding_mask, expected_attention_mask)

    def test_scripting(self, get_text_decoder, input_ids):
        text_decoder = get_text_decoder()
        scripted_text_decoder = torch.jit.script(text_decoder)
        assert_expected(
            scripted_text_decoder(input_ids),
            text_decoder(input_ids),
            rtol=0,
            atol=1e-4,
        )
