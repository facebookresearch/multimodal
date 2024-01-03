# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch
from tests.test_utils import assert_expected, init_weights_with_constant, set_rng_seed
from torchmultimodal.models.coca.coca_model import (
    coca_vit,
    CoCaForPretraining,
    CoCaModelOutput,
)


class TestCoCaModel:
    @pytest.fixture(autouse=True)
    def random(self):
        set_rng_seed(0)

    @pytest.fixture
    def batch_size(self):
        return 2

    @pytest.fixture
    def vocab_size(self):
        return 50

    @pytest.fixture
    def num_text_positions(self):
        return 11

    @pytest.fixture
    def attention_pooler_output_dim(self):
        return 8

    @pytest.fixture
    def text_output_dim(self):
        return 8

    @pytest.fixture
    def image_size(self):
        return 12

    @pytest.fixture
    def get_coca_model(
        self,
        vocab_size,
        num_text_positions,
        attention_pooler_output_dim,
        text_output_dim,
        image_size,
    ):
        def create_coca_model(cascaded_pooler: bool = False):
            coca_model = coca_vit(
                vision_patch_size=4,
                vision_dim_feedforward=24,
                vision_n_layer=2,
                vision_n_head=2,
                vocab_size=vocab_size,
                num_text_positions=num_text_positions,
                text_hidden_dim=8,
                text_n_layer=2,
                text_n_head=2,
                text_dim_feedforward=32,
                text_output_dim=text_output_dim,
                fusion_n_layer=2,
                fusion_n_head=2,
                fusion_dim_feedforward=32,
                multimodal_output_projection_dim=vocab_size,
                pooler_input_embed_dim=6,
                pooler_output_embed_dim=attention_pooler_output_dim,
                image_size=image_size,
                pooler_n_head=2,
                cascaded_pooler=cascaded_pooler,
            )
            init_weights_with_constant(coca_model)
            coca_model.eval()
            return coca_model

        return create_coca_model

    @pytest.fixture
    def text_inputs(self):
        return torch.LongTensor(
            [
                [1, 3, 4, 5, 6, 7, 8, 2, 0, 0, 0],
                [1, 25, 28, 34, 39, 45, 40, 5, 12, 6, 2],
            ]
        )

    @pytest.fixture
    def image_inputs(self, batch_size, image_size):
        return torch.randn(batch_size, 3, image_size, image_size)

    @pytest.fixture
    def expected(
        self,
        batch_size,
        vocab_size,
        num_text_positions,
        attention_pooler_output_dim,
        text_output_dim,
    ):
        pooled_val = 0.3536
        logit_val = 8.0
        return CoCaModelOutput(
            image_pooled_output=pooled_val
            * torch.ones(batch_size, attention_pooler_output_dim),
            text_pooled_output=pooled_val * torch.ones(batch_size, text_output_dim),
            multimodal_embeddings=logit_val
            * torch.ones(batch_size, num_text_positions - 1, vocab_size),
        )

    @pytest.fixture
    def coca_for_pretraining(self, get_coca_model):
        coca_model = get_coca_model()
        coca_for_pretraining = CoCaForPretraining(coca_model)
        init_weights_with_constant(coca_for_pretraining)
        coca_for_pretraining.eval()
        return coca_for_pretraining

    def test_coca_model(self, text_inputs, image_inputs, get_coca_model, expected):
        coca_model = get_coca_model()
        actual = coca_model(image_inputs, text_inputs)
        assert_expected(actual, expected, rtol=0, atol=1e-4)

    def test_coca_model_cascaded_pooler(
        self,
        text_inputs,
        image_inputs,
        get_coca_model,
        batch_size,
        attention_pooler_output_dim,
    ):
        coca_model_cascaded_pooler = get_coca_model(cascaded_pooler=True)
        actual = coca_model_cascaded_pooler(image_inputs, text_inputs)
        assert_expected(
            actual.image_pooled_output.shape,
            (batch_size, 1, attention_pooler_output_dim),
            rtol=0,
            atol=1e-4,
        )

    def test_scripting(self, text_inputs, image_inputs, get_coca_model):
        coca_model = get_coca_model()
        scripted_model = torch.jit.script(coca_model)
        assert_expected(
            scripted_model(image_inputs, text_inputs),
            coca_model(image_inputs, text_inputs),
            rtol=0,
            atol=1e-4,
        )

    def test_coca_for_pretraining(
        self, text_inputs, image_inputs, coca_for_pretraining
    ):
        actual_losses = coca_for_pretraining(image_inputs, text_inputs)
        expected_losses = {
            "contrastive": torch.tensor(0.6931),
            "captioning": torch.tensor(3.9120),
        }
        assert_expected(actual_losses, expected_losses, rtol=0, atol=1e-4)
