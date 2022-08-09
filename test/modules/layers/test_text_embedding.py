# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch
from test.test_utils import assert_expected, set_rng_seed
from torchmultimodal.modules.layers.text_embedding import BERTTextEmbeddings


@pytest.fixture(autouse=True)
def random():
    set_rng_seed(4)


class TestTextEmbeddings:
    @pytest.fixture
    def text_embedding(self):
        return BERTTextEmbeddings(hidden_size=3, vocab_size=3)

    @pytest.fixture
    def input_ids(self):
        return torch.tensor([[1, 2], [0, 2]])

    def test_forward(self, input_ids, text_embedding):
        embs = text_embedding(input_ids)
        actual = embs.shape
        expected = torch.Size([2, 2, 3])
        assert_expected(actual, expected)

    def test_invalid_input(self, text_embedding):
        with pytest.raises(ValueError):
            _ = text_embedding()

    def test_create_position_ids_from_input_ids(self, input_ids, text_embedding):
        actual = text_embedding.create_position_ids_from_input_ids(input_ids)
        expected = torch.tensor([[1, 2], [0, 1]])
        assert_expected(actual, expected)
