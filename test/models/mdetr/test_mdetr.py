# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch

from test.test_utils import assert_expected, set_rng_seed
from torchmultimodal.models.mdetr.model import mdetr_resnet101


class TestMDETR:
    @pytest.fixture(autouse=True)
    def rng(self):
        set_rng_seed(0)

    @pytest.fixture(autouse=True)
    def batch_size(self):
        return 2

    @pytest.fixture(autouse=True)
    def num_queries(self):
        return 100

    @pytest.fixture(autouse=True)
    def num_classes(self):
        return 255

    @pytest.fixture()
    def test_tensors(self):
        return torch.rand(2, 3, 64, 64).unbind(dim=0)

    @pytest.fixture()
    def input_ids(self):
        return torch.Tensor(
            [
                [0, 100, 64, 192, 5, 3778, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [
                    0,
                    1708,
                    190,
                    114,
                    38,
                    1395,
                    192,
                    5,
                    3778,
                    6,
                    38,
                    216,
                    14,
                    24,
                    8785,
                    2,
                ],
            ]
        ).to(dtype=torch.long)

    @pytest.fixture()
    def mdetr(self, num_queries, num_classes):
        mdetr = mdetr_resnet101(num_queries=num_queries, num_classes=num_classes)
        mdetr.eval()
        return mdetr

    def test_mdetr_model(
        self,
        mdetr,
        test_tensors,
        input_ids,
        batch_size,
        num_queries,
        num_classes,
    ):
        out = mdetr(test_tensors, input_ids)
        logits_actual = out.pred_logits
        boxes_actual = out.pred_boxes
        logits_expected = torch.Tensor(
            [
                -0.0145,
                0.0121,
                0.0270,
                0.0310,
                0.0072,
                -0.0002,
                0.0100,
                0.0012,
                0.0290,
                0.0067,
            ]
        )
        boxes_expected = torch.Tensor(
            [
                0.4896,
                0.4898,
                0.4897,
                0.4900,
                0.4894,
                0.4895,
                0.4897,
                0.4908,
                0.4902,
                0.4899,
            ]
        )
        assert logits_actual.size() == (
            batch_size,
            num_queries,
            num_classes + 1,
        )
        assert boxes_actual.size() == (batch_size, num_queries, 4)
        assert_expected(logits_actual[1, :10, 1], logits_expected, rtol=0, atol=1e-3)
        assert_expected(boxes_actual[1, :10, 1], boxes_expected, rtol=0, atol=1e-3)
