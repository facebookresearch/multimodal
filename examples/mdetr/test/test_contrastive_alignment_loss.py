# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch
from examples.mdetr.contrastive_alignment_loss import contrastive_alignment_loss
from test.test_utils import assert_expected, set_rng_seed
from transformers import RobertaTokenizerFast


@pytest.fixture(scope="class", autouse=True)
def rng():
    set_rng_seed(0)


class TestContrastiveAlignmentLoss:
    @pytest.fixture(scope="class")
    def batch_size(self):
        return 2

    @pytest.fixture(scope="class")
    def num_queries(self):
        return 20

    @pytest.fixture(scope="class")
    def num_tokens(self):
        return 255

    @pytest.fixture(scope="class")
    def contrastive_dim(self):
        return 8

    @pytest.fixture(scope="class")
    def projected_tokens(self, batch_size, num_tokens, contrastive_dim):
        return torch.randn(batch_size, num_tokens, contrastive_dim)

    @pytest.fixture(scope="class")
    def projected_queries(self, batch_size, num_queries, contrastive_dim):
        return torch.randn(batch_size, num_queries, contrastive_dim)

    @pytest.fixture(scope="class")
    def target_tokens(self):
        return [
            [
                [[39, 44]],
                [[39, 44]],
                [[39, 44]],
                [[39, 44]],
                [[39, 44]],
                [[39, 44]],
                [[39, 44]],
                [[39, 44]],
                [[39, 44]],
                [[39, 44]],
                [[48, 57]],
                [[39, 44]],
                [[15, 22]],
                [[39, 44]],
                [[39, 44]],
                [[0, 3]],
                [[39, 44]],
            ],
            [
                [[33, 48]],
                [[33, 48]],
                [[33, 48]],
                [[33, 48]],
                [[33, 48]],
                [[33, 48]],
                [[33, 48]],
                [[33, 48]],
                [[33, 48]],
                [[33, 48]],
                [[33, 48]],
                [[9, 18]],
                [[33, 48]],
                [[33, 48]],
                [[0, 5]],
                [[33, 48]],
            ],
        ]

    @pytest.fixture(scope="class")
    def indices(self):
        indices = [
            (torch.Tensor([5, 7, 9]), torch.Tensor([2, 1, 0])),
            (
                torch.Tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
                torch.Tensor([9, 8, 4, 6, 0, 2, 3, 1, 5, 7]),
            ),
        ]
        return [(x[0].to(dtype=torch.int), x[1].to(dtype=torch.int)) for x in indices]

    @pytest.fixture(scope="class")
    def num_boxes(self):
        return 25

    @pytest.fixture(scope="class")
    def tokenized(self):
        captions = [
            "Man talking on a phone , surrounded by books in an office .",
            "A man on the phone surrounded by stacks of books .",
        ]
        tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")
        tokenized = tokenizer.batch_encode_plus(
            captions, padding="longest", return_tensors="pt"
        )
        return tokenized

    def test_contrastive_alignment_loss(
        self,
        projected_queries,
        projected_tokens,
        target_tokens,
        indices,
        num_boxes,
        tokenized,
    ):
        expected = torch.tensor(30.3021)
        actual = contrastive_alignment_loss(
            projected_queries,
            projected_tokens,
            target_tokens,
            indices,
            num_boxes,
            tokenized,
        )
        assert_expected(expected, actual, rtol=0, atol=1e-3)
