# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest

import torch
from test.test_utils import assert_expected, assert_expected_namedtuple, set_rng_seed
from torch import nn
from torchmultimodal.models.flava.model import (
    flava_image_encoder,
    flava_model_for_classification,
    flava_model_for_pretraining,
    flava_text_encoder,
    FLAVAModel,
    FLAVAOutput,
)
from torchmultimodal.modules.layers.transformer import TransformerOutput

NUM_CLASSES = 2


@pytest.fixture(autouse=True)
def random():
    set_rng_seed(1234)


class TestFLAVA:
    @pytest.fixture
    def classification_inputs(self):
        text = torch.randint(0, 30500, (2, 77), dtype=torch.long)
        image = torch.rand((2, 3, 224, 224))
        labels = torch.randint(0, 2, (2,), dtype=torch.long)
        return text, image, labels

    @pytest.fixture
    def pretraining_inputs(self):
        text = torch.randint(0, 30500, (2, 77), dtype=torch.long)
        image = torch.rand((2, 3, 224, 224))
        image_for_codebook = torch.rand(2, 3, 112, 112)
        image_patches_mask = torch.randint(0, 2, (2, 196), dtype=torch.long)
        text_masked = text.detach().clone()
        text_masked[:, 1:3] = 100
        mlm_labels = text.detach().clone()
        mlm_labels[:, :] = -1
        mlm_labels[:, 1:3] = text[:, 1:3]
        itm_labels = torch.tensor((0, 1), dtype=torch.long)
        return (
            text,
            image,
            image_for_codebook,
            image_patches_mask,
            text_masked,
            mlm_labels,
            itm_labels,
        )

    @torch.no_grad()
    def test_forward_classification(self, classification_inputs):
        text, image, labels = classification_inputs

        flava = flava_model_for_classification(NUM_CLASSES, pretrained_model_key=None)
        flava.eval()

        # Test multimodal scenario
        output = flava(image, text, "mm", labels)
        assert_expected(output.loss.item(), 0.7180, rtol=0, atol=1e-4)

        # Test unimodal image scenario
        output = flava(image, text, "image", labels)
        assert_expected(output.loss.item(), 0.7020, rtol=0, atol=1e-4)

        # Test unimodal text scenario
        output = flava(image, text, "text", labels)
        assert_expected(output.loss.item(), 0.6663, rtol=0, atol=1e-4)

    @torch.no_grad()
    def test_forward_pretraining(self, pretraining_inputs):
        (
            text,
            image,
            image_for_codebook,
            image_patches_mask,
            text_masked,
            mlm_labels,
            itm_labels,
        ) = pretraining_inputs
        flava = flava_model_for_pretraining()
        flava.eval()
        output = flava(
            image=image,
            text=text,
            image_for_codebook=image_for_codebook,
            image_patches_mask=image_patches_mask,
            text_masked=text_masked,
            required_embedding="mm",
            itm_labels=itm_labels,
            mlm_labels=mlm_labels,
        )
        assert output.mlm_output is None
        assert output.mim_output is None
        assert output.global_contrastive_output is not None
        assert output.mmm_text_output is not None
        assert output.mmm_image_output is not None
        assert output.itm_output is not None
        assert_expected(
            sum(
                value if value is not None else 0 for value in output.losses.values()
            ).item(),
            21.5150,
            rtol=0,
            atol=1e-4,
        )

        output = flava(
            image=image,
            text=text,
            image_for_codebook=image_for_codebook,
            image_patches_mask=image_patches_mask,
            text_masked=text_masked,
            required_embedding="image",
            itm_labels=itm_labels,
            mlm_labels=mlm_labels,
        )
        assert output.mlm_output is None
        assert output.mim_output is not None
        assert output.global_contrastive_output is None
        assert output.mmm_text_output is None
        assert output.mmm_image_output is None
        assert output.itm_output is None
        assert_expected(
            sum(
                value if value is not None else 0 for value in output.losses.values()
            ).item(),
            8.9674,
            rtol=0,
            atol=1e-4,
        )

        output = flava(
            image=image,
            text=text,
            image_for_codebook=image_for_codebook,
            image_patches_mask=image_patches_mask,
            text_masked=text_masked,
            required_embedding="text",
            itm_labels=itm_labels,
            mlm_labels=mlm_labels,
        )
        assert output.mlm_output is not None
        assert output.mim_output is None
        assert output.global_contrastive_output is None
        assert output.mmm_text_output is None
        assert output.mmm_image_output is None
        assert output.itm_output is None
        assert_expected(
            sum(
                value if value is not None else 0 for value in output.losses.values()
            ).item(),
            10.0305,
            rtol=0,
            atol=1e-4,
        )


class TestFLAVAModel:
    @pytest.fixture
    def text_encoder(self):
        return flava_text_encoder(
            hidden_size=2,
            num_attention_heads=1,
            num_hidden_layers=1,
            intermediate_size=2,
        )

    @pytest.fixture
    def image_encoder(self):
        return flava_image_encoder(
            hidden_size=2,
            num_attention_heads=1,
            num_hidden_layers=1,
            intermediate_size=2,
            image_size=2,
            patch_size=1,
            num_channels=3,
            use_image_masking=True,
        )

    @pytest.fixture
    def flava(
        self,
        image_encoder,
        text_encoder,
    ):
        flava_model = FLAVAModel(
            image_encoder=image_encoder,
            text_encoder=text_encoder,
            mm_encoder=nn.Identity(),
            image_to_mm_projection=nn.Identity(),
            text_to_mm_projection=nn.Identity(),
            text_projection=nn.Identity(),
            image_projection=nn.Identity(),
        )
        flava_model.eval()
        return flava_model

    @pytest.fixture
    def inputs(self):
        image = torch.zeros(2, 3, 2, 2)
        masked_image = torch.ones(2, 1)
        text = torch.ones(2, 3, dtype=torch.int32)
        masked_text = torch.ones(2, 3, dtype=torch.int32)
        return image, masked_image, text, masked_text

    def test_forward_image_text(self, image_encoder, text_encoder, flava, inputs):
        image, _, text, _ = inputs
        actual = flava(image, text)
        expected_image = image_encoder(image)
        expected_text = text_encoder(
            text, return_attn_weights=True, return_hidden_states=True
        )
        expected = FLAVAOutput(
            text_masked=TransformerOutput(),
            multimodal_masked=TransformerOutput(),
            multimodal=TransformerOutput(),
            text=expected_text,
            image=expected_image,
            image_masked=expected_image,
            projected_image_embeddings=expected_image.last_hidden_state[:, 0, :],
            projected_text_embeddings=expected_text.last_hidden_state[:, 0, :],
        )
        assert_expected_namedtuple(actual, expected, rtol=0, atol=1e-4)

    def test_forward_masked_image_and_text(
        self, image_encoder, text_encoder, flava, inputs
    ):
        image, masked_image, text, masked_text = inputs
        actual = flava(
            text=text,
            image=image,
            image_patches_mask=masked_image,
            text_masked=masked_text,
        )
        expected_image = image_encoder(image)
        expected_image_masked = image_encoder(image, masked_image)
        expected_text = text_encoder(
            text, return_attn_weights=True, return_hidden_states=True
        )
        expected_text_masked = text_encoder(
            masked_text, return_attn_weights=True, return_hidden_states=True
        )
        expected = FLAVAOutput(
            text_masked=expected_text_masked,
            multimodal_masked=torch.cat(
                [
                    expected_image_masked.hidden_states[-1],
                    expected_text_masked.hidden_states[-1],
                ],
                1,
            ),
            multimodal=TransformerOutput(),
            text=expected_text,
            image=expected_image,
            image_masked=expected_image_masked,
            projected_image_embeddings=expected_image.last_hidden_state[:, 0, :],
            projected_text_embeddings=expected_text.last_hidden_state[:, 0, :],
        )
        assert_expected_namedtuple(actual, expected, rtol=0, atol=1e-4)

    def test_forward_masked_text(self, text_encoder, flava, inputs):
        _, _, text, masked_text = inputs
        text = torch.ones(2, 3, dtype=torch.int32)
        masked_text = torch.ones(2, 3, dtype=torch.int32)
        actual = flava(text=text, text_masked=masked_text)
        expected_text = text_encoder(
            text, return_attn_weights=True, return_hidden_states=True
        )
        expected = FLAVAOutput(
            multimodal_masked=TransformerOutput(),
            multimodal=TransformerOutput(),
            text=expected_text,
            image=TransformerOutput(),
            image_masked=TransformerOutput(),
            text_masked=text_encoder(
                masked_text, return_attn_weights=True, return_hidden_states=True
            ),
            projected_image_embeddings=None,
            projected_text_embeddings=expected_text.last_hidden_state[:, 0, :],
        )
        assert_expected_namedtuple(actual, expected, rtol=0, atol=1e-4)

    def test_forward_text(self, text_encoder, flava, inputs):
        _, _, text, _ = inputs
        actual = flava(text=text)
        expected_text = text_encoder(
            text, return_attn_weights=True, return_hidden_states=True
        )
        expected = FLAVAOutput(
            multimodal_masked=TransformerOutput(),
            multimodal=TransformerOutput(),
            text=expected_text,
            image=TransformerOutput(),
            image_masked=TransformerOutput(),
            text_masked=TransformerOutput(),
            projected_image_embeddings=None,
            projected_text_embeddings=expected_text.last_hidden_state[:, 0, :],
        )
        assert_expected_namedtuple(actual, expected, rtol=0, atol=1e-4)

    def test_forward_masked_image(self, image_encoder, flava, inputs):
        image, masked_image, _, _ = inputs
        actual = flava(image=image, image_patches_mask=masked_image)
        expected_image = image_encoder(image)
        expected = FLAVAOutput(
            multimodal_masked=TransformerOutput(),
            multimodal=TransformerOutput(),
            text=TransformerOutput(),
            image=expected_image,
            image_masked=image_encoder(image, masked_image),
            text_masked=TransformerOutput(),
            projected_image_embeddings=expected_image.last_hidden_state[:, 0, :],
            projected_text_embeddings=None,
        )
        assert_expected_namedtuple(actual, expected, rtol=0, atol=1e-4)

    def test_forward_image(self, image_encoder, flava, inputs):
        image, _, _, _ = inputs
        actual = flava(image=image)
        expected_image = image_encoder(image)
        expected = FLAVAOutput(
            multimodal_masked=TransformerOutput(),
            multimodal=TransformerOutput(),
            text=TransformerOutput(),
            image=expected_image,
            image_masked=image_encoder(image),
            text_masked=TransformerOutput(),
            projected_image_embeddings=expected_image.last_hidden_state[:, 0, :],
            projected_text_embeddings=None,
        )
        assert_expected_namedtuple(actual, expected, rtol=0, atol=1e-4)
