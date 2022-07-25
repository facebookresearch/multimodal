# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch
from test.test_utils import assert_expected, set_rng_seed
from torchmultimodal.models.flava.flava_model import (
    flava_model_for_classification,
    flava_model_for_pretraining,
)


@pytest.fixture(autouse=True)
def random():
    set_rng_seed(4)


class TestFLAVACheckpoint:
    @pytest.fixture
    def text_input(self):
        text = torch.randint(0, 30500, (2, 77), dtype=torch.long)
        return text

    @pytest.fixture
    def image_input(self):
        image = torch.rand((2, 3, 224, 224))
        return image

    @pytest.fixture
    def inputs_classification(self, image_input, text_input):
        def gather_inputs(required_embedding):
            labels = torch.randint(0, 2, (2,), dtype=torch.long)
            return image_input, text_input, required_embedding, labels

        return gather_inputs

    @pytest.fixture
    def inputs_pretraining(self, image_input, text_input):
        def gather_inputs(required_embedding):
            image_for_codebook = torch.rand(2, 3, 112, 112)
            image_patches_mask = torch.randint(0, 2, (2, 196), dtype=torch.long)
            text_masked = text_input.detach().clone()
            text_masked[:, 1:3] = 100
            mlm_labels = text_input.detach().clone()
            mlm_labels[:, :] = -1
            mlm_labels[:, 1:3] = text_input[:, 1:3]
            itm_labels = torch.tensor((0, 1), dtype=torch.long)
            skip_unmasked_mm_encoder = True
            return (
                image_input,
                text_input,
                image_for_codebook,
                image_patches_mask,
                text_masked,
                required_embedding,
                skip_unmasked_mm_encoder,
                itm_labels,
                mlm_labels,
            )

        return gather_inputs

    @pytest.fixture
    def classification_model(self):
        return flava_model_for_classification(
            num_classes=3, pretrained_model_key="flava_full"
        )

    @pytest.fixture
    def pretraining_model(self):
        return flava_model_for_pretraining(pretrained_model_key="flava_full")

    def _assert_tensor_dicts_equal(self, dict_actual, dict_expected):
        for key in dict_expected:
            actual = (
                torch.zeros(1)
                if dict_actual[key] is None
                else torch.tensor(dict_actual[key])
            )
            expected = (
                torch.zeros(1)
                if dict_expected[key] is None
                else torch.tensor(dict_expected[key])
            )
            assert_expected(actual, expected, rtol=0, atol=1e-4)

    def test_flava_model_for_classification(
        self, classification_model, inputs_classification
    ):
        output = classification_model(*inputs_classification("mm"))
        actual = output.loss
        expected = torch.tensor(1.1017)
        assert_expected(actual, expected, rtol=0, atol=1e-4)

        output = classification_model(*inputs_classification("image"))
        actual = output.loss
        expected = torch.tensor(1.0912)
        assert_expected(actual, expected, rtol=0, atol=1e-4)

        output = classification_model(*inputs_classification("text"))
        actual = output.loss
        expected = torch.tensor(1.1136)
        assert_expected(actual, expected, rtol=0, atol=1e-4)

    def test_flava_model_for_pretraining(self, pretraining_model, inputs_pretraining):
        output = pretraining_model(*inputs_pretraining("mm"))
        actual = output.losses
        expected = dict(
            mmm_text_loss=15.8243,
            mmm_image_loss=11.9815,
            mim_loss=None,
            mlm_loss=None,
            itm_loss=0.7904,
            global_contrastive_loss=0.6195,
        )
        self._assert_tensor_dicts_equal(actual, expected)

        output = pretraining_model(*inputs_pretraining("image"))
        actual = output.losses
        expected = dict(
            mmm_text_loss=None,
            mmm_image_loss=None,
            mim_loss=10.5971,
            mlm_loss=None,
            itm_loss=None,
            global_contrastive_loss=None,
        )
        self._assert_tensor_dicts_equal(actual, expected)

        output = pretraining_model(*inputs_pretraining("text"))
        actual = output.losses
        expected = dict(
            mmm_text_loss=None,
            mmm_image_loss=None,
            mim_loss=None,
            mlm_loss=18.9983,
            itm_loss=None,
            global_contrastive_loss=None,
        )
        self._assert_tensor_dicts_equal(actual, expected)
