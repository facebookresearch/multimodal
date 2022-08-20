# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch
from examples.mdetr.loss import construct_positive_map, contrastive_alignment_loss
from test.test_utils import assert_expected, set_rng_seed
from transformers import RobertaTokenizerFast


@pytest.fixture(autouse=True)
def rng():
    set_rng_seed(0)


class TestContrastiveAlignmentLoss:
    @pytest.fixture()
    def batch_size(self):
        return 2

    @pytest.fixture()
    def num_queries(self):
        return 20

    @pytest.fixture()
    def num_tokens(self):
        return 255

    @pytest.fixture()
    def contrastive_dim(self):
        return 8

    @pytest.fixture()
    def projected_tokens(self, batch_size, num_tokens, contrastive_dim):
        return torch.randn(batch_size, num_tokens, contrastive_dim)

    @pytest.fixture()
    def projected_queries(self, batch_size, num_queries, contrastive_dim):
        return torch.randn(batch_size, num_queries, contrastive_dim)

    @pytest.fixture()
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

    @pytest.fixture()
    def indices(self):
        indices = [
            (torch.Tensor([5, 7, 9]), torch.Tensor([2, 1, 0])),
            (
                torch.Tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
                torch.Tensor([9, 8, 4, 6, 0, 2, 3, 1, 5, 7]),
            ),
        ]
        return [(x[0].to(dtype=torch.int), x[1].to(dtype=torch.int)) for x in indices]

    @pytest.fixture()
    def num_boxes(self):
        return 25

    @pytest.fixture()
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

    def test_construct_positive_map(
        self, batch_size, num_queries, num_tokens, target_tokens, indices, tokenized
    ):
        logits = torch.ones(batch_size, num_queries, num_tokens)

        actual = construct_positive_map(logits, target_tokens, indices, tokenized)
        actual_nonzero_entries = torch.nonzero(actual)
        expected_size = (batch_size, num_queries, num_tokens)
        expected_nonzero_entries = torch.LongTensor(
            [
                [0, 5, 9],
                [0, 7, 9],
                [0, 9, 9],
                [1, 0, 8],
                [1, 0, 9],
                [1, 0, 10],
                [1, 1, 8],
                [1, 1, 9],
                [1, 1, 10],
                [1, 2, 8],
                [1, 2, 9],
                [1, 2, 10],
                [1, 3, 8],
                [1, 3, 9],
                [1, 3, 10],
                [1, 4, 8],
                [1, 4, 9],
                [1, 4, 10],
                [1, 5, 8],
                [1, 5, 9],
                [1, 5, 10],
                [1, 6, 8],
                [1, 6, 9],
                [1, 6, 10],
                [1, 7, 8],
                [1, 7, 9],
                [1, 7, 10],
                [1, 8, 8],
                [1, 8, 9],
                [1, 8, 10],
                [1, 9, 8],
                [1, 9, 9],
                [1, 9, 10],
            ]
        )
        assert actual.size() == expected_size
        assert_expected(actual_nonzero_entries, expected_nonzero_entries)
