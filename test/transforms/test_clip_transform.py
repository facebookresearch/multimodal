# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
from pytorch.text.fb.utils.manifold import register_manifold_handler
from torchmultimodal.transforms.clip_transform import CLIPTransform
from torchvision.transforms import ToPILImage


class TestCLIPTransform(unittest.TestCase):
    def setUp(self):
        register_manifold_handler()
        torch.manual_seed(1234)
        self.context_length = 77
        self.image1 = ToPILImage()(torch.ones(3, 300, 500))
        self.image2 = ToPILImage()(torch.ones(3, 50, 100))
        self.text1 = "Taken with my analogue EOS 500N with black & white film."
        self.text2 = "This is a shorter sentence."
        self.long_text = (self.text1 + " ") * 20
        self.text1_tokens = [
            49406,
            2807,
            593,
            607,
            46031,
            17805,
            276,
            271,
            271,
            333,
            593,
            1449,
            261,
            1579,
            1860,
            269,
            49407,
        ]
        self.bos_token = self.text1_tokens[0]
        self.eos_token = self.text1_tokens[-1]
        self.text1_token_len = len(self.text1_tokens)
        self.clip_transform = CLIPTransform()

    def test_clip_single_transform(self):
        expected_tensor = torch.tensor(
            self.text1_tokens + [0] * (self.context_length - self.text1_token_len),
            dtype=torch.long,
        )

        outputs = self.clip_transform(image=self.image1, text=self.text1)
        self.assertEqual(outputs["image"].size(), torch.Size([1, 3, 224, 224]))
        self.assertTrue(torch.equal(outputs["text"][0], expected_tensor))

    def test_clip_multi_transform(self):
        images = [self.image1] * 5 + [self.image2] * 2
        texts = [self.text1] * 5 + [self.text2] + [self.long_text]
        outputs = self.clip_transform(image=images, text=texts)
        self.assertEqual(outputs["image"].size(), torch.Size([7, 3, 224, 224]))
        self.assertEqual(outputs["text"].size(), torch.Size([7, self.context_length]))

        expected_long_encoding = torch.tensor(
            [self.bos_token] + self.text1_tokens[1:-1] * 20 + [self.eos_token],
            dtype=torch.long,
        )[: self.context_length]

        # Check encoding of long text
        self.assertTrue(torch.equal(outputs["text"][-1], expected_long_encoding))

        # Check zero padding for short texts
        self.assertEqual(outputs["text"][:-1, self.text1_token_len :].max(), 0.0)
