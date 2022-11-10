# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import random

import pytest
import torch
from examples.mdetr.matcher import HungarianMatcher
from tests.test_utils import assert_expected, set_rng_seed
from torchvision.ops.boxes import box_convert


@pytest.fixture(autouse=True)
def rng():
    set_rng_seed(0)


class TestMatcher:
    @pytest.fixture()
    def batch_size(self):
        return 2

    @pytest.fixture()
    def num_classes(self):
        return 7

    @pytest.fixture()
    def max_slice_len(self):
        return 3

    @pytest.fixture()
    def n_boxes_per_sample(self):
        return [3, 8]

    @pytest.fixture()
    def total_boxes(self, n_boxes_per_sample):
        return sum(n_boxes_per_sample)

    @pytest.fixture()
    def positive_map(self, max_slice_len, total_boxes, num_classes):
        positive_map = torch.zeros(total_boxes, num_classes)
        for i in range(total_boxes):
            start_idx = random.randint(0, num_classes - max_slice_len)
            increment = random.randint(2, max_slice_len)
            positive_map[i, start_idx : start_idx + increment] = 1
        return positive_map

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
    def target_boxes(self, construct_valid_boxes, n_boxes_per_sample):
        return [construct_valid_boxes(n_boxes) for n_boxes in n_boxes_per_sample]

    @pytest.fixture()
    def matcher(self):
        return HungarianMatcher(cost_class=1, cost_bbox=5, cost_giou=2)

    @pytest.mark.parametrize(
        "num_queries,expected",
        [
            (
                12,
                [
                    (torch.LongTensor([7, 10, 11]), torch.LongTensor([0, 1, 2])),
                    (
                        torch.LongTensor([0, 1, 3, 4, 5, 6, 9, 10]),
                        torch.LongTensor([0, 7, 3, 6, 5, 4, 2, 1]),
                    ),
                ],
            ),
            (
                5,
                [
                    (torch.LongTensor([0, 1, 4]), torch.LongTensor([2, 1, 0])),
                    (
                        torch.LongTensor([0, 1, 2, 3, 4]),
                        torch.LongTensor([1, 5, 4, 6, 2]),
                    ),
                ],
            ),
        ],
    )
    def test_matcher(
        self,
        batch_size,
        num_classes,
        construct_valid_boxes,
        target_boxes,
        positive_map,
        matcher,
        num_queries,
        expected,
    ):
        pred_logits = torch.randn(batch_size, num_queries, num_classes)
        pred_boxes = construct_valid_boxes(batch_size * num_queries).reshape(
            batch_size, num_queries, -1
        )
        actual = matcher(pred_logits, pred_boxes, target_boxes, positive_map)
        for actual_sample, expected_sample in zip(actual, expected):
            assert_expected(actual_sample[0], expected_sample[0])
            assert_expected(actual_sample[1], expected_sample[1])
