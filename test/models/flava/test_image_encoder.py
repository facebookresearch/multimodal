# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
from test.test_utils import assert_expected, set_rng_seed
from torch import nn
from torchmultimodal.models.flava.image_encoder import ImageEmbeddings, ImageTransformer
from torchmultimodal.modules.layers.transformer import TransformerEncoder


class TestFlavaImageEncoder(unittest.TestCase):
    def setUp(self):
        set_rng_seed(0)
        self.image_embedding = ImageEmbeddings(
            image_size=2, patch_size=1, hidden_size=2
        )

        encoder = TransformerEncoder(
            n_layer=1,
            d_model=2,
            n_head=1,
            dim_feedforward=1,
            activation=nn.GELU,
            norm_first=True,
        )

        self.image_encoder = ImageTransformer(
            embeddings=self.image_embedding,
            encoder=encoder,
            layernorm=nn.LayerNorm(2),
            pooler=nn.Identity(),
        )

    def test_embedding(self):
        input = torch.ones(2, 3, 2, 2)
        out = self.image_embedding(input)
        assert_expected(
            out,
            torch.Tensor(
                [
                    [
                        [0.0000, 0.0000],
                        [0.0224, 0.0573],
                        [0.0224, 0.0573],
                        [0.0224, 0.0573],
                        [0.0224, 0.0573],
                    ],
                    [
                        [0.0000, 0.0000],
                        [0.0224, 0.0573],
                        [0.0224, 0.0573],
                        [0.0224, 0.0573],
                        [0.0224, 0.0573],
                    ],
                ]
            ),
            atol=1e-4,
            rtol=0,
        )

    def test_image_encoder(self):
        input = torch.ones(2, 3, 2, 2)
        out = self.image_encoder(input)
        assert_expected(
            out.last_hidden_state,
            torch.Tensor(
                [
                    [
                        [-0.0040, 0.0040],
                        [-0.9840, 0.9840],
                        [-0.9840, 0.9840],
                        [-0.9840, 0.9840],
                        [-0.9840, 0.9840],
                    ],
                    [
                        [-0.0040, 0.0040],
                        [-0.9840, 0.9840],
                        [-0.9840, 0.9840],
                        [-0.9840, 0.9840],
                        [-0.9840, 0.9840],
                    ],
                ]
            ),
            atol=1e-4,
            rtol=0,
        )
        assert_expected(out.pooler_output, out.last_hidden_state)
        assert_expected(
            out.hidden_states,
            (
                torch.Tensor(
                    [
                        [
                            [0.0000, 0.0000],
                            [0.0224, 0.0573],
                            [0.0224, 0.0573],
                            [0.0224, 0.0573],
                            [0.0224, 0.0573],
                        ],
                        [
                            [0.0000, 0.0000],
                            [0.0224, 0.0573],
                            [0.0224, 0.0573],
                            [0.0224, 0.0573],
                            [0.0224, 0.0573],
                        ],
                    ]
                ),
                torch.Tensor(
                    [
                        [
                            [0.0008, 0.0008],
                            [0.0232, 0.0581],
                            [0.0232, 0.0581],
                            [0.0232, 0.0581],
                            [0.0232, 0.0581],
                        ],
                        [
                            [0.0008, 0.0008],
                            [0.0232, 0.0581],
                            [0.0232, 0.0581],
                            [0.0232, 0.0581],
                            [0.0232, 0.0581],
                        ],
                    ]
                ),
            ),
            atol=1e-4,
            rtol=0,
        )
        assert_expected(
            out.attentions,
            (
                torch.Tensor(
                    [
                        [
                            [
                                [0.2000, 0.2000, 0.2000, 0.2000, 0.2000],
                                [0.1999, 0.2000, 0.2000, 0.2000, 0.2000],
                                [0.1999, 0.2000, 0.2000, 0.2000, 0.2000],
                                [0.1999, 0.2000, 0.2000, 0.2000, 0.2000],
                                [0.1999, 0.2000, 0.2000, 0.2000, 0.2000],
                            ]
                        ],
                        [
                            [
                                [0.2000, 0.2000, 0.2000, 0.2000, 0.2000],
                                [0.1999, 0.2000, 0.2000, 0.2000, 0.2000],
                                [0.1999, 0.2000, 0.2000, 0.2000, 0.2000],
                                [0.1999, 0.2000, 0.2000, 0.2000, 0.2000],
                                [0.1999, 0.2000, 0.2000, 0.2000, 0.2000],
                            ]
                        ],
                    ]
                ),
            ),
            atol=1e-4,
            rtol=0,
        )
