# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
from test.test_utils import assert_expected_namedtuple
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


class TestFLAVA(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(1234)

    @torch.no_grad()
    def test_forward_classification(self):
        text = torch.randint(0, 30500, (2, 77), dtype=torch.long)
        image = torch.rand((2, 3, 224, 224))

        labels = torch.randint(0, 2, (2,), dtype=torch.long)

        flava = flava_model_for_classification(NUM_CLASSES, pretrained_model_key=None)
        flava.eval()

        # Test multimodal scenario
        output = flava(image, text, "mm", labels)
        self.assertAlmostEqual(output.loss.item(), 0.7180, places=4)

        # Test unimodal image scenario
        output = flava(image, text, "image", labels)
        self.assertAlmostEqual(output.loss.item(), 0.7020, places=4)

        # Test unimodal text scenario
        output = flava(image, text, "text", labels)
        self.assertAlmostEqual(output.loss.item(), 0.6663, places=4)

    @torch.no_grad()
    def test_forward_pretraining(self):
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
        self.assertIsNone(output.mlm_output)
        self.assertIsNone(output.mim_output)
        self.assertIsNotNone(output.global_contrastive_output)
        self.assertIsNotNone(output.mmm_text_output)
        self.assertIsNotNone(output.mmm_image_output)
        self.assertIsNotNone(output.itm_output)
        self.assertAlmostEqual(
            sum(
                value if value is not None else 0 for value in output.losses.values()
            ).item(),
            21.4029,
            places=4,
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
        self.assertIsNone(output.mlm_output)
        self.assertIsNotNone(output.mim_output)
        self.assertIsNone(output.global_contrastive_output)
        self.assertIsNone(output.mmm_text_output)
        self.assertIsNone(output.mmm_image_output)
        self.assertIsNone(output.itm_output)
        self.assertAlmostEqual(
            sum(
                value if value is not None else 0 for value in output.losses.values()
            ).item(),
            8.6285,
            places=4,
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
        self.assertIsNotNone(output.mlm_output)
        self.assertIsNone(output.mim_output)
        self.assertIsNone(output.global_contrastive_output)
        self.assertIsNone(output.mmm_text_output)
        self.assertIsNone(output.mmm_image_output)
        self.assertIsNone(output.itm_output)

        self.assertAlmostEqual(
            sum(
                value if value is not None else 0 for value in output.losses.values()
            ).item(),
            11.0002,
            places=4,
        )


class TestFLAVAModel(unittest.TestCase):
    def setUp(self):
        self.text_encoder = flava_text_encoder(
            hidden_size=2,
            num_attention_heads=1,
            num_hidden_layers=1,
            intermediate_size=2,
        )
        self.image_encoder = flava_image_encoder(
            hidden_size=2,
            num_attention_heads=1,
            num_hidden_layers=1,
            intermediate_size=2,
            image_size=2,
            patch_size=1,
            num_channels=3,
            use_image_masking=True,
        )

        mm_encoder = nn.Identity()
        image_to_mm_projection = nn.Identity()
        text_to_mm_projection = nn.Identity()
        self.flava = FLAVAModel(
            image_encoder=self.image_encoder,
            text_encoder=self.text_encoder,
            mm_encoder=mm_encoder,
            image_to_mm_projection=image_to_mm_projection,
            text_to_mm_projection=text_to_mm_projection,
            text_projection=nn.Identity(),
            image_projection=nn.Identity(),
        )

    def test_forward_image_text(self):
        image = torch.ones(2, 3, 2, 2)
        text = torch.ones(2, 3, dtype=torch.int32)
        actual = self.flava(image, text)
        expected = FLAVAOutput(
            text_masked=TransformerOutput(),
            multimodal_masked=TransformerOutput(),
            multimodal=TransformerOutput(),
            text=self.text_encoder(
                text, return_attn_weights=True, return_hidden_states=True
            ),
            image=self.image_encoder(image),
            image_masked=self.image_encoder(image),
            projected_image_embeddings=actual.image.last_hidden_state[:, 0, :],
            projected_text_embeddings=actual.text.last_hidden_state[:, 0, :],
        )
        assert_expected_namedtuple(actual, expected, rtol=0, atol=1e-4)

    def test_forward_masked_image_and_text(self):
        image = torch.zeros(2, 3, 2, 2)
        masked_image = torch.ones(2, 1)
        text = torch.ones(2, 3, dtype=torch.int32)
        masked_text = torch.ones(2, 3, dtype=torch.int32)
        actual = self.flava(
            text=text,
            image=image,
            image_patches_mask=masked_image,
            text_masked=masked_text,
        )
        expected = FLAVAOutput(
            text_masked=self.text_encoder(
                masked_text, return_attn_weights=True, return_hidden_states=True
            ),
            multimodal_masked=torch.cat(
                [
                    actual.image_masked.hidden_states[-1],
                    actual.text_masked.hidden_states[-1],
                ],
                1,
            ),
            multimodal=TransformerOutput(),
            text=self.text_encoder(
                text, return_attn_weights=True, return_hidden_states=True
            ),
            image=self.image_encoder(image),
            image_masked=self.image_encoder(image, masked_image),
            projected_image_embeddings=actual.image.last_hidden_state[:, 0, :],
            projected_text_embeddings=actual.text.last_hidden_state[:, 0, :],
        )
        assert_expected_namedtuple(actual, expected, rtol=0, atol=1e-4)

    def test_forward_masked_text(self):
        text = torch.ones(2, 3, dtype=torch.int32)
        masked_text = torch.ones(2, 3, dtype=torch.int32)
        actual = self.flava(text=text, text_masked=masked_text)
        expected = FLAVAOutput(
            multimodal_masked=TransformerOutput(),
            multimodal=TransformerOutput(),
            text=self.text_encoder(
                text, return_attn_weights=True, return_hidden_states=True
            ),
            image=TransformerOutput(),
            image_masked=TransformerOutput(),
            text_masked=self.text_encoder(
                masked_text, return_attn_weights=True, return_hidden_states=True
            ),
            projected_image_embeddings=None,
            projected_text_embeddings=actual.text.last_hidden_state[:, 0, :],
        )
        assert_expected_namedtuple(actual, expected, rtol=0, atol=1e-4)

    def test_forward_text(self):
        text = torch.ones(2, 3, dtype=torch.int32)
        actual = self.flava(text=text)
        expected = FLAVAOutput(
            multimodal_masked=TransformerOutput(),
            multimodal=TransformerOutput(),
            text=self.text_encoder(
                text, return_attn_weights=True, return_hidden_states=True
            ),
            image=TransformerOutput(),
            image_masked=TransformerOutput(),
            text_masked=TransformerOutput(),
            projected_image_embeddings=None,
            projected_text_embeddings=actual.text.last_hidden_state[:, 0, :],
        )
        assert_expected_namedtuple(actual, expected, rtol=0, atol=1e-4)

    def test_forward_masked_image(self):
        image = torch.zeros(2, 3, 2, 2)
        masked_image = torch.ones(2, 1)
        actual = self.flava(image=image, image_patches_mask=masked_image)
        expected = FLAVAOutput(
            multimodal_masked=TransformerOutput(),
            multimodal=TransformerOutput(),
            text=TransformerOutput(),
            image=self.image_encoder(image),
            image_masked=self.image_encoder(image, masked_image),
            text_masked=TransformerOutput(),
            projected_image_embeddings=actual.image.last_hidden_state[:, 0, :],
            projected_text_embeddings=None,
        )
        assert_expected_namedtuple(actual, expected, rtol=0, atol=1e-4)

    def test_forward_image(self):
        image = torch.zeros(2, 3, 2, 2)
        actual = self.flava(image=image)
        expected = FLAVAOutput(
            multimodal_masked=TransformerOutput(),
            multimodal=TransformerOutput(),
            text=TransformerOutput(),
            image=self.image_encoder(image),
            image_masked=self.image_encoder(image),
            text_masked=TransformerOutput(),
            projected_image_embeddings=actual.image.last_hidden_state[:, 0, :],
            projected_text_embeddings=None,
        )
        assert_expected_namedtuple(actual, expected, rtol=0, atol=1e-4)
