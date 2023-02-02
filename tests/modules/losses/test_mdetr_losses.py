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
    def pred_logits(self, batch_size, num_queries, num_classes):
        return torch.randn(batch_size, num_queries, num_classes + 1)

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

    def test_soft_token_prediction_loss(self, pred_logits):
        indices = [
            (torch.LongTensor([4, 5, 9]), torch.LongTensor([1, 0, 2])),
            (
                torch.LongTensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
                torch.LongTensor([9, 8, 3, 4, 5, 6, 7, 1, 0, 2]),
            ),
        ]
        n_boxes_per_sample = [3, 10]
        num_boxes = 19
        positive_map = torch.Tensor(
            [
                [
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                    1.0,
                    0.0,
                    0.0,
                ],
                [
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ],
                [
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                    1.0,
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ],
                [
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                    1.0,
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ],
                [
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                    1.0,
                    0.0,
                    0.0,
                ],
                [
                    0.0,
                    1.0,
                    1.0,
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ],
                [
                    1.0,
                    1.0,
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ],
                [
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ],
                [
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                    1.0,
                    1.0,
                    0.0,
                    0.0,
                ],
                [
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ],
                [
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ],
                [
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ],
                [
                    1.0,
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ],
            ]
        )

        actual = torch.Tensor(
            soft_token_prediction_loss(
                pred_logits, n_boxes_per_sample, positive_map, indices, num_boxes
            )
        )
        expected = torch.tensor(5.2867)
        assert_expected(actual, expected, rtol=0, atol=1e-3)

    def test_box_losses(self, pred_boxes):
        indices = [
            (torch.LongTensor([4, 6, 7, 8, 9]), torch.LongTensor([3, 0, 4, 2, 1])),
            (torch.LongTensor([1, 8]), torch.LongTensor([1, 0])),
        ]
        num_boxes = 8
        target_boxes = [
            torch.Tensor(
                [
                    [0.9941, 0.6071, 0.0070, 0.6372],
                    [0.9358, 0.6296, 0.1217, 0.2474],
                    [0.6058, 0.8187, 0.7384, 0.1234],
                    [0.5829, 0.6806, 0.6967, 0.0670],
                    [0.4472, 0.7152, 0.1831, 0.5401],
                ]
            ),
            torch.Tensor(
                [[0.2642, 0.6090, 0.4897, 0.6948], [0.8163, 0.6436, 0.0900, 0.5304]]
            ),
        ]

        actual = box_losses(pred_boxes, target_boxes, indices, num_boxes)
        expected_l1_loss = torch.tensor(0.8463)
        expected_giou_loss = torch.tensor(1.2569)
        assert_expected(actual.l1_loss, expected_l1_loss, rtol=0, atol=1e-3)
        assert_expected(actual.giou_loss, expected_giou_loss, rtol=0, atol=1e-3)
