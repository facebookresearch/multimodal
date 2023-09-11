# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

import torch
from tests.test_utils import assert_expected, init_weights_with_constant
from torchmultimodal.models.masked_auto_encoder.swin_decoder import (
    SwinTransformer,
    SwinTransformerBlock,
    WindowMultiHeadAttention,
)


class TestWindowMultiHeadAttention:
    def test_forward(self):
        attn = WindowMultiHeadAttention(
            input_dim=3, num_heads=3, window_size=(2, 2), meta_hidden_dim=4
        )
        init_weights_with_constant(attn)
        attn.eval()
        inputs = torch.ones(1, 4, 3)
        actual = attn(inputs)
        assert_expected(
            actual,
            torch.Tensor(
                [
                    [
                        [13.0, 13.0, 13.0],
                        [13.0, 13.0, 13.0],
                        [13.0, 13.0, 13.0],
                        [13.0, 13.0, 13.0],
                    ]
                ]
            ),
        )


class TestSwinTransformerBlock:
    def test_forward_no_shift(self):
        model = SwinTransformerBlock(
            input_dim=3,
            num_heads=3,
            input_size=(4, 4),
            window_size=(2, 2),
            feedforward_dim=12,
        )
        model.eval()
        init_weights_with_constant(model)
        actual = model(torch.ones(1, 16, 3))
        assert_expected(
            actual,
            torch.Tensor(
                [
                    [
                        [3.0, 3.0, 3.0],
                        [3.0, 3.0, 3.0],
                        [3.0, 3.0, 3.0],
                        [3.0, 3.0, 3.0],
                        [3.0, 3.0, 3.0],
                        [3.0, 3.0, 3.0],
                        [3.0, 3.0, 3.0],
                        [3.0, 3.0, 3.0],
                        [3.0, 3.0, 3.0],
                        [3.0, 3.0, 3.0],
                        [3.0, 3.0, 3.0],
                        [3.0, 3.0, 3.0],
                        [3.0, 3.0, 3.0],
                        [3.0, 3.0, 3.0],
                        [3.0, 3.0, 3.0],
                        [3.0, 3.0, 3.0],
                    ]
                ]
            ),
        )

    def test_forward_shift(self):
        model = SwinTransformerBlock(
            input_dim=3,
            num_heads=3,
            input_size=(4, 4),
            window_size=(2, 2),
            shift_size=(2, 0),
            feedforward_dim=12,
        )
        model.eval()
        init_weights_with_constant(model)
        actual = model(torch.ones(1, 16, 3))
        assert_expected(
            actual,
            torch.Tensor(
                [
                    [
                        [3.0, 3.0, 3.0],
                        [3.0, 3.0, 3.0],
                        [3.0, 3.0, 3.0],
                        [3.0, 3.0, 3.0],
                        [3.0, 3.0, 3.0],
                        [3.0, 3.0, 3.0],
                        [3.0, 3.0, 3.0],
                        [3.0, 3.0, 3.0],
                        [3.0, 3.0, 3.0],
                        [3.0, 3.0, 3.0],
                        [3.0, 3.0, 3.0],
                        [3.0, 3.0, 3.0],
                        [3.0, 3.0, 3.0],
                        [3.0, 3.0, 3.0],
                        [3.0, 3.0, 3.0],
                        [3.0, 3.0, 3.0],
                    ]
                ]
            ),
        )


class TestSwinTransformer:
    def test_forward(self):
        model = SwinTransformer(
            n_layer=2,
            input_dim=3,
            num_heads=3,
            input_size=(4, 4),
            window_size=(2, 2),
            feedforward_dim=12,
        )
        init_weights_with_constant(model)
        model.eval()
        actual = model(torch.ones(1, 16, 3))
        assert_expected(
            actual.last_hidden_state,
            torch.Tensor(
                [
                    [
                        [1.0, 1.0, 1.0],
                        [1.0, 1.0, 1.0],
                        [1.0, 1.0, 1.0],
                        [1.0, 1.0, 1.0],
                        [1.0, 1.0, 1.0],
                        [1.0, 1.0, 1.0],
                        [1.0, 1.0, 1.0],
                        [1.0, 1.0, 1.0],
                        [1.0, 1.0, 1.0],
                        [1.0, 1.0, 1.0],
                        [1.0, 1.0, 1.0],
                        [1.0, 1.0, 1.0],
                        [1.0, 1.0, 1.0],
                        [1.0, 1.0, 1.0],
                        [1.0, 1.0, 1.0],
                        [1.0, 1.0, 1.0],
                    ]
                ]
            ),
        )
