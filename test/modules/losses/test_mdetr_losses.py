# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import random

import pytest
import torch
from test.test_utils import assert_expected, set_rng_seed
from torch import nn
from torchmultimodal.modules.losses.mdetr import (
    box_losses,
    mdetr_gqa_loss,
    soft_token_prediction_loss,
)
from torchvision.ops.boxes import box_convert


@pytest.fixture(autouse=True)
def rng():
    set_rng_seed(1)


class TestMDETRLosses:
    @pytest.fixture()
    def batch_size(self):
        return 2

    @pytest.fixture()
    def num_queries(self):
        return 10

    @pytest.fixture()
    def num_classes(self):
        return 15

    @pytest.fixture()
    def n_boxes_per_sample(self, batch_size, num_queries):
        return [random.randint(1, num_queries) for _ in range(batch_size)]

    @pytest.fixture()
    def total_boxes(self, n_boxes_per_sample):
        return sum(n_boxes_per_sample)

    @pytest.fixture()
    def pred_logits(self, batch_size, num_queries, num_classes):
        return torch.randn(batch_size, num_queries, num_classes + 1)

    @pytest.fixture()
    def max_slice_len(self):
        return 3

    @pytest.fixture()
    def positive_map(self, max_slice_len, total_boxes, num_classes):
        positive_map = torch.zeros(total_boxes, num_classes + 1)
        for i in range(total_boxes):
            start_idx = random.randint(0, num_classes - max_slice_len)
            increment = random.randint(2, max_slice_len)
            positive_map[i, start_idx : start_idx + increment] = 1
        return positive_map

    @pytest.fixture()
    def indices(self, num_queries, n_boxes_per_sample):
        return [
            tuple(
                torch.sort(
                    torch.multinomial(
                        torch.arange(num_queries, dtype=torch.float), n_boxes
                    )
                )
            )
            for n_boxes in n_boxes_per_sample
        ]

    @pytest.fixture()
    def num_boxes(self, total_boxes):
        return int(random.uniform(0.5 * total_boxes, 2 * total_boxes))

    @pytest.fixture()
    def construct_valid_boxes(self):
        def _construct_valid_boxes(n_boxes):
            boxes = []
            for _ in range(n_boxes):
                x1, y1 = torch.rand(2).unbind(-1)
                x2 = random.uniform(x1.item(), 1)
                y2 = random.uniform(y1.item(), 1)
                box = box_convert(
                    torch.Tensor([x1, y1, x2, y2]), in_fmt="xyxy", out_fmt="cxcywh"
                )
                boxes.append(box)
            return torch.stack(boxes)

        return _construct_valid_boxes

    @pytest.fixture()
    def pred_boxes(self, construct_valid_boxes, batch_size, num_queries):
        return construct_valid_boxes(batch_size * num_queries).reshape(
            batch_size, num_queries, -1
        )

    @pytest.fixture()
    def target_boxes(self, construct_valid_boxes, n_boxes_per_sample):
        return [construct_valid_boxes(n_boxes) for n_boxes in n_boxes_per_sample]

    def test_soft_token_prediction_loss(
        self, pred_logits, n_boxes_per_sample, positive_map, indices, num_boxes
    ):
        actual = torch.Tensor(
            soft_token_prediction_loss(
                pred_logits, n_boxes_per_sample, positive_map, indices, num_boxes
            )
        )
        expected = torch.tensor(5.0893)
        assert_expected(actual, expected, rtol=0, atol=1e-3)

    def test_box_losses(self, pred_boxes, target_boxes, indices, num_boxes):
        actual = box_losses(pred_boxes, target_boxes, indices, num_boxes)
        expected_l1_loss = torch.tensor(0.7721)
        expected_giou_loss = torch.tensor(1.1768)
        assert_expected(actual.l1_loss, expected_l1_loss, rtol=0, atol=1e-3)
        assert_expected(actual.giou_loss, expected_giou_loss, rtol=0, atol=1e-3)


class TestGQALoss:
    @pytest.fixture()
    def batch_size(self):
        return 2

    @pytest.fixture()
    def embedding_dim(self):
        return 7

    @pytest.fixture()
    def num_question_types(self):
        return 5

    @pytest.fixture()
    def loss(self, embedding_dim):
        loss = mdetr_gqa_loss(embedding_dim)
        loss.answer_type_head.weight = nn.Parameter(
            torch.ones_like(loss.answer_type_head.weight)
        )
        loss.answer_type_head.bias = nn.Parameter(
            torch.zeros_like(loss.answer_type_head.bias)
        )
        for v in loss.specialized_heads.values():
            v.weight = nn.Parameter(torch.ones_like(v.weight))
            v.bias = nn.Parameter(torch.zeros_like(v.bias))
        return loss

    @pytest.fixture()
    def answer_type_embeddings(self, batch_size, embedding_dim):
        return torch.randn(batch_size, embedding_dim)

    @pytest.fixture()
    def specialized_embeddings(self, batch_size, embedding_dim, num_question_types):
        return torch.randn(batch_size, embedding_dim, num_question_types)

    @pytest.fixture()
    def answer_types(self):
        return torch.LongTensor([0, 3])

    @pytest.fixture()
    def answer_specific_labels(self):
        return [
            torch.LongTensor([1, -100]),
            torch.LongTensor([-100, -100]),
            torch.LongTensor([-100, -100]),
            torch.LongTensor([-100, 5]),
            torch.LongTensor([-100, -100]),
        ]

    def test_invalid_inputs(
        self,
        answer_type_embeddings,
        specialized_embeddings,
        answer_types,
        answer_specific_labels,
        loss,
    ):
        with pytest.raises(ValueError):
            actual = loss(
                answer_type_embeddings,
                specialized_embeddings[:, :, :-1],
                answer_types,
                answer_specific_labels,
            )

    def test_valid_inputs(
        self,
        answer_type_embeddings,
        specialized_embeddings,
        answer_types,
        answer_specific_labels,
        loss,
    ):
        actual = loss(
            answer_type_embeddings,
            specialized_embeddings,
            answer_types,
            answer_specific_labels,
        )
        expected = {
            "answer_type": torch.tensor(1.6094),
            "answer_obj": torch.tensor(1.0986),
            "answer_attr": torch.tensor(0.0),
            "answer_rel": torch.tensor(0.0),
            "answer_global": torch.tensor(4.7095),
            "answer_cat": torch.tensor(0.0),
        }
        for k in actual.keys():
            assert_expected(actual[k], expected[k], rtol=0, atol=1e-3)
