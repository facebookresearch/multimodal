# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
from tests.test_utils import assert_expected
from torch import Tensor
from torchmultimodal.models.blip2.qformer_utils import get_causal_mask


class TestExtendedAttnMaskForDecoder:
    @pytest.fixture
    def attention_mask(self):
        return Tensor([[1.0, 1.0, 1.0, 0.0], [1.0, 1.0, 0.0, 1.0]])

    @pytest.fixture
    def input_shape(self):
        return (2, 2)

    def test_extended_attention_mask(self, attention_mask):
        actual_mask = get_causal_mask(attention_mask, attention_mask.shape)
        expected = Tensor(
            [
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [1.0, 1.0, 0.0, 0.0],
                    [1.0, 1.0, 1.0, 0.0],
                    [1.0, 1.0, 1.0, 1.0],
                ],
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [1.0, 1.0, 0.0, 0.0],
                    [1.0, 1.0, 1.0, 0.0],
                    [1.0, 1.0, 1.0, 1.0],
                ],
            ]
        )
        assert_expected(actual_mask, expected, rtol=0, atol=1e-4)

    def test_extended_attention_mask_diff_input_size(self, attention_mask, input_shape):
        actual_mask = get_causal_mask(
            attention_mask,
            input_shape,
        )
        expected = Tensor(
            Tensor(
                [
                    [[1.0, 1.0, 1.0, 0.0], [1.0, 1.0, 1.0, 1.0]],
                    [[1.0, 1.0, 1.0, 0.0], [1.0, 1.0, 1.0, 1.0]],
                ]
            )
        )
        assert_expected(actual_mask, expected, rtol=0, atol=1e-4)

    def test_extended_attention_mask_with_query_embs(self, attention_mask, input_shape):
        actual_mask = get_causal_mask(attention_mask, input_shape, has_query=True)
        expected = Tensor(
            Tensor(
                [
                    [
                        [1.0, 1.0, 0.0, 0.0],
                        [1.0, 1.0, 0.0, 0.0],
                        [1.0, 1.0, 1.0, 0.0],
                        [1.0, 1.0, 1.0, 1.0],
                    ],
                    [
                        [1.0, 1.0, 0.0, 0.0],
                        [1.0, 1.0, 0.0, 0.0],
                        [1.0, 1.0, 1.0, 0.0],
                        [1.0, 1.0, 1.0, 1.0],
                    ],
                ]
            )
        )
        assert_expected(actual_mask, expected, rtol=0, atol=1e-4)
