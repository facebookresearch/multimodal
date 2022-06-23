# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch
from test.test_utils import assert_expected, set_rng_seed
from torchmultimodal.modules.encoders.clip_text_encoder import CLIPTextEncoder


class TestCLIPTextEncoder:
    @pytest.fixture
    def start(self):
        set_rng_seed(1234)
        context_length = 77
        batch_size, embedding_dim = 2, 4

        def build_text(text_length):
            return torch.randint(1, 10, (batch_size, text_length), dtype=torch.long)

        def build_encoder(
            embedding_dim=embedding_dim,
            use_clip_init=True,
            context_length=context_length,
        ):
            return CLIPTextEncoder(
                embedding_dim=embedding_dim,
                use_clip_init=use_clip_init,
                context_length=context_length,
            )

        return build_encoder, build_text

    def test_initialization(self, start):
        """Test whether the ``use_clip_init`` parameter changes the initialized parameters."""
        build_encoder, _ = start

        # First, initialize without CLIP parameters and get the standard deviation of token embedding weights
        # Use larger embedding size for stability in std
        text_encoder = build_encoder(use_clip_init=False, embedding_dim=50)
        assert isinstance(text_encoder, torch.nn.Module)
        actual_std_no_init = torch.std(
            text_encoder.encoder.token_embedding.weight
        ).item()

        # Get the standard deviation with CLIP params
        text_encoder = build_encoder(embedding_dim=50)
        actual_std_clip_init = torch.std(
            text_encoder.encoder.token_embedding.weight
        ).item()

        assert abs(actual_std_clip_init - actual_std_no_init) > 1e-1

    def test_clip_parameters(self, start):
        build_encoder, _ = start
        # Use larger embedding size for stability in std
        text_encoder = build_encoder(embedding_dim=50)

        assert_expected(
            actual=torch.std(text_encoder.encoder.token_embedding.weight).item(),
            expected=0.02,
            rtol=0,
            atol=1e-4,
        )
        assert_expected(
            actual=torch.std(
                text_encoder.encoder.positional_embedding.embedding.weight
            ).item(),
            expected=0.01,
            rtol=0,
            atol=1e-4,
        )

        proj_std = 0.0090
        attn_std = 0.0442
        fc_std = 0.0313
        for layer in text_encoder.encoder.layers.layers:
            assert_expected(
                actual=torch.std(layer.self_attn.in_proj_weight).item(),
                expected=attn_std,
                rtol=0,
                atol=1e-4,
            )
            assert_expected(
                actual=torch.std(layer.self_attn.out_proj.weight).item(),
                expected=proj_std,
                rtol=0,
                atol=1e-4,
            )
            assert_expected(
                actual=torch.std(layer.linear1.weight).item(),
                expected=fc_std,
                rtol=0,
                atol=1e-4,
            )
            assert_expected(
                actual=torch.std(layer.linear2.weight).item(),
                expected=proj_std,
                rtol=0,
                atol=1e-4,
            )

        assert_expected(
            actual=torch.std(text_encoder.projection.weight).item(),
            expected=0.0442,
            rtol=0,
            atol=1e-4,
        )

    def test_attention_mask(self, start):
        build_encoder, _ = start
        text_encoder = build_encoder(context_length=4)
        assert isinstance(text_encoder, torch.nn.Module)

        actual = text_encoder.build_attention_mask()
        expected = torch.Tensor(
            [[0, 1, 1, 1], [0, 0, 1, 1], [0, 0, 0, 1], [0, 0, 0, 0]]
        ).to(dtype=bool)
        assert_expected(actual=actual, expected=expected, rtol=0, atol=0)

    def test_forward(self, start):
        build_encoder, build_text = start
        text = build_text(text_length=3)

        text_encoder = build_encoder()
        assert isinstance(text_encoder, torch.nn.Module)

        actual_clip_init = text_encoder(text)
        expected_clip_init = torch.Tensor(
            [[-1.9412, -0.8044, -0.7616, -0.3185], [0.2408, -0.3241, -0.2544, -0.3608]]
        )
        assert_expected(
            actual=actual_clip_init, expected=expected_clip_init, rtol=0, atol=1e-4
        )

    def test_forward_over_context_length(self, start):
        build_encoder, build_text = start

        text_encoder = build_encoder()
        assert isinstance(text_encoder, torch.nn.Module)

        text = build_text(text_encoder.context_length + 1)

        with pytest.raises(IndexError):
            text_encoder(text)

    def test_scripting(self, start):
        build_encoder, build_text = start
        text = build_text(text_length=3)

        text_encoder = build_encoder()
        assert isinstance(text_encoder, torch.nn.Module)
        scripted_encoder = torch.jit.script(text_encoder)

        actual = scripted_encoder(text)
        expected = torch.Tensor(
            [[-1.9412, -0.8044, -0.7616, -0.3185], [0.2408, -0.3241, -0.2544, -0.3608]]
        )
        assert_expected(actual=actual, expected=expected, rtol=0, atol=1e-4)
