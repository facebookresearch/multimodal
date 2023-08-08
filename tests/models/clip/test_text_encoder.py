# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from math import inf

import pytest
import torch
from tests.test_utils import assert_expected, set_rng_seed
from torchmultimodal.models.clip.text_encoder import CLIPTextEncoder


class TestCLIPTextEncoder:
    @pytest.fixture
    def start(self):
        set_rng_seed(1234)
        context_length = 77
        batch_size, embedding_dim, heads, width = 2, 4, 2, 512

        def build_text(text_length):
            return torch.randint(1, 10, (batch_size, text_length), dtype=torch.long)

        def build_encoder(
            embedding_dim=embedding_dim,
            use_clip_init=True,
            context_length=context_length,
            heads=heads,
            width=width,
        ):
            return CLIPTextEncoder(
                embedding_dim=embedding_dim,
                use_clip_init=use_clip_init,
                context_length=context_length,
                width=width,
                heads=heads,
            )

        return build_encoder, build_text

    def test_clip_parameters(self, start):
        build_encoder, _ = start
        # Use larger embedding size for stability in std
        text_encoder = build_encoder(embedding_dim=50)

        assert_expected(
            actual=torch.std(text_encoder.token_embedding.weight).item(),
            expected=0.02,
            rtol=0,
            atol=1e-4,
        )
        assert_expected(
            actual=torch.std(text_encoder.positional_embedding).item(),
            expected=0.01,
            rtol=0,
            atol=1e-3,
        )

        proj_std = 0.0090
        attn_std = 0.0442
        fc_std = 0.0313
        for layer in text_encoder.encoder.layers:
            assert_expected(
                actual=torch.std(layer.self_attn.in_proj_weight).item(),
                expected=attn_std,
                rtol=0,
                atol=5e-3,
            )
            assert_expected(
                actual=torch.std(layer.self_attn.out_proj.weight).item(),
                expected=proj_std,
                rtol=0,
                atol=5e-3,
            )
            assert_expected(
                actual=torch.std(layer.linear1.weight).item(),
                expected=fc_std,
                rtol=0,
                atol=5e-3,
            )
            assert_expected(
                actual=torch.std(layer.linear2.weight).item(),
                expected=proj_std,
                rtol=0,
                atol=5e-3,
            )

        assert_expected(
            actual=torch.std(text_encoder.projection.weight).item(),
            expected=attn_std,
            rtol=0,
            atol=5e-3,
        )

    def test_attention_mask(self, start):
        build_encoder, _ = start
        text_encoder = build_encoder(context_length=4)
        assert isinstance(text_encoder, torch.nn.Module)

        actual = text_encoder.build_attention_mask()
        expected = torch.Tensor(
            [[0, -inf, -inf, -inf], [0, 0, -inf, -inf], [0, 0, 0, -inf], [0, 0, 0, 0]]
        )
        assert_expected(actual=actual, expected=expected, rtol=0, atol=0)

    def test_forward(self, start):
        build_encoder, build_text = start
        text = build_text(text_length=77)

        text_encoder = build_encoder()
        assert isinstance(text_encoder, torch.nn.Module)

        actual_clip_init = text_encoder(text)
        expected_clip_init = torch.Tensor(
            [[-1.3103, -0.6713, -0.9614, 0.7010], [1.1780, 0.1888, 0.8019, 0.7287]]
        )
        assert_expected(
            actual=actual_clip_init, expected=expected_clip_init, rtol=0, atol=1e-4
        )

    def test_forward_return_hidden_state(self, start):
        build_encoder, build_text = start
        text = build_text(text_length=3)

        text_encoder = build_encoder(context_length=3, width=4)
        assert isinstance(text_encoder, torch.nn.Module)

        out = text_encoder(text, return_hidden_state=True)
        assert (
            hasattr(out, "projected_embeddings")
            and hasattr(out, "hidden_state")
            and len(out) == 2
        )

        actual_projected_embeddings = out.projected_embeddings
        actual_hidden_state = out.hidden_state
        expected_projected_embeddings = torch.Tensor(
            [
                [-0.3668, -1.5966, -0.3304, -0.5938],
                [-0.7904, 0.8768, -0.9707, -0.7271],
            ]
        )
        expected_hidden_state = torch.Tensor(
            [
                [
                    [0.6348, -0.0414, -1.6042, 1.0108],
                    [0.6205, -0.0303, -1.6066, 1.0164],
                    [0.5916, -0.0017, -1.6133, 1.0234],
                ],
                [
                    [0.5911, -0.0152, -1.6079, 1.0320],
                    [0.1468, -1.6758, 0.7402, 0.7888],
                    [0.6721, -0.2897, -1.4934, 1.1109],
                ],
            ]
        )
        assert_expected(
            actual=actual_projected_embeddings,
            expected=expected_projected_embeddings,
            rtol=0,
            atol=1e-4,
        )
        assert_expected(
            actual=actual_hidden_state,
            expected=expected_hidden_state,
            rtol=0,
            atol=1e-4,
        )

    def test_forward_over_context_length(self, start):
        build_encoder, build_text = start

        text_encoder = build_encoder()
        assert isinstance(text_encoder, torch.nn.Module)

        text = build_text(text_encoder.context_length + 1)

        with pytest.raises(ValueError):
            text_encoder(text)

    def test_scripting(self, start):
        build_encoder, build_text = start
        text = build_text(text_length=77)

        text_encoder = build_encoder()
        assert isinstance(text_encoder, torch.nn.Module)
        scripted_encoder = torch.jit.script(text_encoder)

        actual = scripted_encoder(text)
        expected = torch.Tensor(
            [[-1.3103, -0.6713, -0.9614, 0.7010], [1.1780, 0.1888, 0.8019, 0.7287]]
        )
        assert_expected(actual=actual, expected=expected, rtol=0, atol=1e-4)
