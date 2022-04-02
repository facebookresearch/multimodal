# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from dataclasses import asdict

import torch
from torchmultimodal.models.flava import (
    flava_model_for_pretraining,
    flava_model_for_classification,
    EMBEDDING_OPTIONS,
)


NUM_CLASSES = 2


class TestFLAVA(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(1234)

    @torch.no_grad()
    def test_forward_classification(self):
        flava = flava_model_for_classification(NUM_CLASSES)
        text = torch.randint(0, 30500, (2, 77), dtype=torch.long)
        image = torch.rand((2, 3, 224, 224))

        labels = torch.randint(0, 2, (2,), dtype=torch.long)

        # Test multimodal scenario
        output = flava(image, text, "mm", labels)
        print(output)
        print(output.logits.sum())
        self.assertTrue(
            torch.allclose(
                output.loss, torch.tensor(0.9303, dtype=torch.float), atol=1e-4
            )
        )
        self.assertTrue(
            torch.allclose(
                output.logits.sum(), torch.tensor(0.7676, dtype=torch.float), atol=1e-4
            )
        )

        # Test unimodal image scenario
        output = flava(image, text, "image", labels)
        print(output)
        print(output.logits.sum())
        self.assertTrue(
            torch.allclose(
                output.loss, torch.tensor(0.5453, dtype=torch.float), atol=1e-4
            )
        )
        self.assertTrue(
            torch.allclose(
                output.logits.sum(), torch.tensor(-0.2235, dtype=torch.float), atol=1e-4
            )
        )

        # Test unimodal text scenario
        output = flava(image, text, "text", labels)
        print(output)
        print(output.logits.sum())
        self.assertTrue(
            torch.allclose(
                output.loss, torch.tensor(0.7074, dtype=torch.float), atol=1e-4
            )
        )
        self.assertTrue(
            torch.allclose(
                output.logits.sum(), torch.tensor(-1.0528, dtype=torch.float), atol=1e-4
            )
        )

    @torch.no_grad()
    def test_forward_pretraining(self):
        flava = flava_model_for_pretraining()
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
        print(output)
        self.assertIsNone(output.mlm_output)
        self.assertIsNone(output.mim_output)
        self.assertIsNotNone(output.global_contrastive_output)
        self.assertIsNotNone(output.mmm_text_output)
        self.assertIsNotNone(output.mmm_image_output)
        self.assertIsNotNone(output.itm_output)
        self.assertTrue(
            torch.allclose(
                sum(
                    value if value is not None else 0
                    for _, value in asdict(output.losses).items()
                ),
                torch.tensor(20.6433, dtype=torch.float),
                atol=1e-4,
            )
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
        self.assertTrue(
            torch.allclose(
                sum(
                    value if value is not None else 0
                    for _, value in asdict(output.losses).items()
                ),
                torch.tensor(9.3403, dtype=torch.float),
                atol=1e-4,
            )
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
        print(
            sum(
                value if value is not None else 0
                for key, value in asdict(output.losses).items()
            )
        )
        self.assertTrue(
            torch.allclose(
                sum(
                    value if value is not None else 0
                    for _, value in asdict(output.losses).items()
                ),
                torch.tensor(10.8777, dtype=torch.float),
                atol=1e-4,
            )
        )
