# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest

import torch

from test.test_utils import assert_expected, set_rng_seed
from torchmultimodal.models.mdetr.text_encoder import ModifiedTransformerEncoder


@pytest.fixture(autouse=True)
def random():
    set_rng_seed(0)


class TestModifiedTransformerEncoder:
    @pytest.fixture
    def hidden_size(self):
        return 768

    @pytest.fixture
    def batch_size(self):
        return 2

    @pytest.fixture
    def input_length(self):
        return 16

    @pytest.fixture
    def encoder_input(self, batch_size, input_length, hidden_size):
        return torch.rand((batch_size, input_length, hidden_size))

    @pytest.fixture
    def attention_mask(self, batch_size, input_length):
        return torch.randint(0, 2, (batch_size, input_length), dtype=bool)

    @pytest.fixture
    def encoder(self, hidden_size):
        return ModifiedTransformerEncoder(
            embedding_dim=hidden_size,
            ffn_dimension=3072,
            num_attention_heads=12,
            num_encoder_layers=12,
            dropout=0.1,
            normalize_before=False,
        )

    def test_mdetr_modified_transformer(
        self,
        batch_size,
        input_length,
        hidden_size,
        encoder_input,
        attention_mask,
        encoder,
    ):
        expected = torch.Tensor(
            [
                0.6401,
                0.2591,
                0.7217,
                0.5619,
                0.3337,
                0.2425,
                0.3801,
                0.3394,
                0.2731,
                0.2023,
                0.2436,
                0.1918,
                0.6731,
                0.3916,
                0.5608,
                0.1991,
            ]
        )
        out = encoder(encoder_input, attention_mask)
        actual = out.last_hidden_state[1, :, 1]
        assert_expected(
            out.last_hidden_state.size(),
            torch.Size((batch_size, input_length, hidden_size)),
        )
        assert_expected(actual, expected, rtol=0.0, atol=1e-4)
