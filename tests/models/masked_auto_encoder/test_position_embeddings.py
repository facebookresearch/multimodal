# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch
from tests.test_utils import assert_expected
from torchmultimodal.models.masked_auto_encoder.position_embeddings import (
    get_2d_sin_cos_embeddings,
)


class TestPositionEmbeddings:
    def test_2d_pos_embeds(self):
        pos_embeds = get_2d_sin_cos_embeddings(embed_dim=8, input_size=(2, 2))
        assert_expected(
            pos_embeds,
            torch.Tensor(
                [
                    [
                        [
                            0.0000,
                            0.0000,
                            0.0000,
                            0.0000,
                            0.0000,
                            0.0000,
                            0.0000,
                            0.0000,
                        ],
                        [
                            0.0000,
                            0.0000,
                            1.0000,
                            1.0000,
                            0.0000,
                            0.0000,
                            1.0000,
                            1.0000,
                        ],
                        [
                            0.8415,
                            0.0100,
                            0.5403,
                            0.9999,
                            0.0000,
                            0.0000,
                            1.0000,
                            1.0000,
                        ],
                        [
                            0.0000,
                            0.0000,
                            1.0000,
                            1.0000,
                            0.8415,
                            0.0100,
                            0.5403,
                            0.9999,
                        ],
                        [
                            0.8415,
                            0.0100,
                            0.5403,
                            0.9999,
                            0.8415,
                            0.0100,
                            0.5403,
                            0.9999,
                        ],
                    ]
                ]
            ),
            atol=0,
            rtol=1e-4,
        )

    def test_2d_pos_embeds_rectangle(self):
        pos_embeds = get_2d_sin_cos_embeddings(embed_dim=8, input_size=(2, 1))
        assert_expected(
            pos_embeds,
            torch.Tensor(
                [
                    [
                        [
                            0.0000,
                            0.0000,
                            0.0000,
                            0.0000,
                            0.0000,
                            0.0000,
                            0.0000,
                            0.0000,
                        ],
                        [
                            0.0000,
                            0.0000,
                            1.0000,
                            1.0000,
                            0.0000,
                            0.0000,
                            1.0000,
                            1.0000,
                        ],
                        [
                            0.0000,
                            0.0000,
                            1.0000,
                            1.0000,
                            0.8415,
                            0.0100,
                            0.5403,
                            0.9999,
                        ],
                    ]
                ]
            ),
            atol=0,
            rtol=1e-4,
        )

    def test_invalid_input(self):
        with pytest.raises(ValueError):
            get_2d_sin_cos_embeddings(embed_dim=3, input_size=(2, 1))
