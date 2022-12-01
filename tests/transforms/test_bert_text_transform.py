# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch

from tests.test_utils import assert_expected, set_rng_seed
from torchtext.models.bert.bundler import BertTextTransform


class TestBertTextTransform:
    @pytest.fixture(autouse=True)
    def set_seed(self):
        set_rng_seed(1234)

    @pytest.fixture
    def utils(self, set_seed):
        tokenizer = BertTextTransform()
        return tokenizer

    def test_single_transform(self, utils):
        tokenizer = utils
        text = "raw text sample for testing tokenizer"
        out = tokenizer(text)
        assert_expected(
            actual=out,
            expected=torch.as_tensor(
                [101, 6315, 3793, 7099, 2005, 5604, 19204, 17629, 102]
            ),
        )

    def test_multi_transform(self, utils):
        tokenizer = utils
        text = ["raw text sample for testing tokenizer", "second shorter text"]
        out = tokenizer(text)
        assert_expected(
            actual=out,
            expected=torch.as_tensor(
                [
                    [101, 6315, 3793, 7099, 2005, 5604, 19204, 17629, 102],
                    [101, 2117, 7820, 3793, 102, 0, 0, 0, 0],
                ]
            ),
        )
