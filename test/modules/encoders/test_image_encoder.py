# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest

import torch
from test.test_utils import assert_expected, set_rng_seed
from torch import nn
from torchmultimodal.modules.encoders.image_encoder import VisionTransformer
from torchmultimodal.modules.layers.image_embedding import ImageEmbeddings
from torchmultimodal.modules.layers.transformer import TransformerEncoder


@pytest.fixture(autouse=True)
def random():
    set_rng_seed(0)


@pytest.fixture
def inputs():
    return torch.ones(2, 3, 2, 2)


class TestVisionTransformer:
    @pytest.fixture
    def vit(self):
        embedding = ImageEmbeddings(image_size=2, patch_size=1, hidden_size=2)
        encoder = TransformerEncoder(
            n_layer=1,
            d_model=2,
            n_head=1,
            dim_feedforward=1,
            activation=nn.GELU,
            norm_first=True,
        )
        image_encoder = VisionTransformer(
            embeddings=embedding,
            encoder=encoder,
            layernorm=nn.LayerNorm(2),
            pooler=nn.Identity(),
        )
        image_encoder.eval()
        return image_encoder

    def test_image_encoder(self, inputs, vit):
        out = vit(inputs)
        assert_expected(
            out.last_hidden_state,
            torch.Tensor(
                [
                    [
                        [-0.9999, 0.9999],
                        [-1.0000, 1.0000],
                        [-1.0000, 1.0000],
                        [-1.0000, 1.0000],
                        [-1.0000, 1.0000],
                    ],
                    [
                        [-0.9999, 0.9999],
                        [-1.0000, 1.0000],
                        [-1.0000, 1.0000],
                        [-1.0000, 1.0000],
                        [-1.0000, 1.0000],
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
                            [-0.1812, -0.0347],
                            [-0.1812, -0.0347],
                            [-0.1812, -0.0347],
                            [-0.1812, -0.0347],
                        ],
                        [
                            [0.0000, 0.0000],
                            [-0.1812, -0.0347],
                            [-0.1812, -0.0347],
                            [-0.1812, -0.0347],
                            [-0.1812, -0.0347],
                        ],
                    ]
                ),
                torch.Tensor(
                    [
                        [
                            [0.4063, 0.9650],
                            [0.2225, 0.9273],
                            [0.2225, 0.9273],
                            [0.2225, 0.9273],
                            [0.2225, 0.9273],
                        ],
                        [
                            [0.4063, 0.9650],
                            [0.2225, 0.9273],
                            [0.2225, 0.9273],
                            [0.2225, 0.9273],
                            [0.2225, 0.9273],
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
                                [0.2339, 0.1915, 0.1915, 0.1915, 0.1915],
                                [0.2226, 0.1943, 0.1943, 0.1943, 0.1943],
                                [0.2226, 0.1943, 0.1943, 0.1943, 0.1943],
                                [0.2226, 0.1943, 0.1943, 0.1943, 0.1943],
                                [0.2226, 0.1943, 0.1943, 0.1943, 0.1943],
                            ]
                        ],
                        [
                            [
                                [0.2339, 0.1915, 0.1915, 0.1915, 0.1915],
                                [0.2226, 0.1943, 0.1943, 0.1943, 0.1943],
                                [0.2226, 0.1943, 0.1943, 0.1943, 0.1943],
                                [0.2226, 0.1943, 0.1943, 0.1943, 0.1943],
                                [0.2226, 0.1943, 0.1943, 0.1943, 0.1943],
                            ]
                        ],
                    ]
                ),
            ),
            atol=1e-4,
            rtol=0,
        )
