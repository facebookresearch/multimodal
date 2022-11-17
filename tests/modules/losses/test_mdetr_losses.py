# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import random

import pytest
import torch
from tests.test_utils import assert_expected, set_rng_seed
from torchmultimodal.modules.losses.mdetr import box_losses, soft_token_prediction_loss
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
        pytest.skip(
            "temp skip until PT side PR lands: https://github.com/pytorch/pytorch/pull/69967"
        )
        actual = torch.Tensor(
            soft_token_prediction_loss(
                pred_logits, n_boxes_per_sample, positive_map, indices, num_boxes
            )
        )
        expected = torch.tensor(5.2867)
        assert_expected(actual, expected, rtol=0, atol=1e-3)

    def test_box_losses(self, pred_boxes, target_boxes, indices, num_boxes):
        pytest.skip(
            "temp skip until PT side PR lands: https://github.com/pytorch/pytorch/pull/69967"
        )
        actual = box_losses(pred_boxes, target_boxes, indices, num_boxes)
        expected_l1_loss = torch.tensor(0.8463)
        expected_giou_loss = torch.tensor(1.2569)
        assert_expected(actual.l1_loss, expected_l1_loss, rtol=0, atol=1e-3)
        assert_expected(actual.giou_loss, expected_giou_loss, rtol=0, atol=1e-3)
