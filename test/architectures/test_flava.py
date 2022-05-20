# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
from test.test_utils import assert_expected
from torch import nn
from torchmultimodal.architectures.flava import FLAVAArchitecture
from torchmultimodal.models.flava import (
    flava_text_encoder,
    flava_image_encoder,
)
from torchmultimodal.modules.layers.transformer import TransformerOutput


class TestFLAVA(unittest.TestCase):
    def setUp(self):
        self.text_encoder = flava_text_encoder(
            hidden_size=2,
            num_attention_heads=1,
            num_hidden_layers=1,
            hidden_dropout_prob=0.0,
            intermediate_size=2,
        )
        self.image_encoder = flava_image_encoder(
            hidden_size=2,
            num_attention_heads=1,
            num_hidden_layers=1,
            hidden_dropout_prob=0.0,
            intermediate_size=2,
            image_size=2,
            patch_size=1,
            num_channels=3,
            use_image_masking=True,
        )

        mm_encoder = nn.Identity()
        image_to_mm_projection = nn.Identity()
        text_to_mm_projection = nn.Identity()
        self.flava = FLAVAArchitecture(
            image_encoder=self.image_encoder,
            text_encoder=self.text_encoder,
            mm_encoder=mm_encoder,
            image_to_mm_projection=image_to_mm_projection,
            text_to_mm_projection=text_to_mm_projection,
        )

    def _assert_empty(self, field):
        self.assertEqual(
            field,
            TransformerOutput(
                last_hidden_state=None,
                pooler_output=None,
                hidden_states=None,
                attentions=None,
            ),
        )

    def test_forward_image_text(self):
        image = torch.ones(2, 3, 2, 2)
        text = torch.ones(2, 3, dtype=torch.int32)
        out = self.flava(image, text)
        self._assert_empty(out.text_masked)
        self._assert_empty(out.multimodal_masked)
        self._assert_empty(out.multimodal)
        assert_expected(out.text, self.text_encoder(text))
        assert_expected(out.image, self.image_encoder(image))
        assert_expected(out.image_masked, self.image_encoder(image))

    def test_forward_masked_image_and_text(self):
        image = torch.zeros(2, 3, 2, 2)
        masked_image = torch.ones(2, 1)
        text = torch.ones(2, 3, dtype=torch.int32)
        masked_text = torch.ones(2, 3, dtype=torch.int32)
        out = self.flava(
            text=text,
            image=image,
            image_patches_mask=masked_image,
            text_masked=masked_text,
        )
        self._assert_empty(out.multimodal)
        assert_expected(out.text, self.text_encoder(text))
        assert_expected(out.text_masked, self.text_encoder(masked_text))
        assert_expected(out.image, self.image_encoder(image))
        assert_expected(out.image_masked, self.image_encoder(image, masked_image))
        assert_expected(
            out.multimodal_masked,
            torch.cat(
                [out.image_masked.hidden_states[-1], out.text_masked.hidden_states[-1]],
                1,
            ),
        )

    def test_forward_masked_text(self):
        text = torch.ones(2, 3, dtype=torch.int32)
        masked_text = torch.ones(2, 3, dtype=torch.int32)
        out = self.flava(text=text, text_masked=masked_text)
        self._assert_empty(out.image)
        self._assert_empty(out.image_masked)
        self._assert_empty(out.multimodal)
        self._assert_empty(out.multimodal_masked)
        assert_expected(out.text, self.text_encoder(text))
        assert_expected(out.text_masked, self.text_encoder(masked_text))

    def test_forward_text(self):
        text = torch.ones(2, 3, dtype=torch.int32)
        out = self.flava(text=text)
        self._assert_empty(out.image)
        self._assert_empty(out.image_masked)
        self._assert_empty(out.multimodal)
        self._assert_empty(out.multimodal_masked)
        self._assert_empty(out.text_masked)
        assert_expected(out.text, self.text_encoder(text))

    def test_forward_masked_image(self):
        image = torch.zeros(2, 3, 2, 2)
        masked_image = torch.ones(2, 1)
        out = self.flava(image=image, image_patches_mask=masked_image)
        self._assert_empty(out.text)
        self._assert_empty(out.text_masked)
        self._assert_empty(out.multimodal)
        self._assert_empty(out.multimodal_masked)
        assert_expected(out.image, self.image_encoder(image))
        assert_expected(out.image_masked, self.image_encoder(image, masked_image))

    def test_forward_image(self):
        image = torch.zeros(2, 3, 2, 2)
        out = self.flava(image=image)
        self._assert_empty(out.text)
        self._assert_empty(out.text_masked)
        self._assert_empty(out.multimodal)
        self._assert_empty(out.multimodal_masked)
        assert_expected(out.image, self.image_encoder(image))
        assert_expected(out.image_masked, self.image_encoder(image))
