# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
from test.test_utils import get_asset_path, set_rng_seed, assert_tensors_equal
from torchmultimodal.transforms.clip_transform import CLIPTransform
from torchvision.transforms import ToPILImage


class TestCLIPTransform(unittest.TestCase):
    def setUp(self):
        set_rng_seed(1234)
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
        bpe_file = "clip_vocab.bpe"
        bpe_merges_file = get_asset_path(bpe_file)
        self.clip_transform = CLIPTransform(text_bpe_merges_path=bpe_merges_file)

    def test_clip_single_transform(self):
        transformed_image, transformed_text = self.clip_transform(
            image=self.image1, text=self.text1
        )

        actual_image_size = transformed_image.size()
        expected_image_size = torch.Size([1, 3, 224, 224])
        assert_tensors_equal(actual_image_size, expected_image_size)

        actual_text = transformed_text[0]
        expected_text = torch.tensor(
            self.text1_tokens + [0] * (self.context_length - self.text1_token_len),
            dtype=torch.long,
        )
        assert_tensors_equal(actual_text, expected_text)

    def test_clip_multi_transform(self):
        images = [self.image1] * 5 + [self.image2] * 2
        texts = [self.text1] * 5 + [self.text2] + [self.long_text]
        transformed_images, transformed_texts = self.clip_transform(
            image=images, text=texts
        )

        actual_images_size = transformed_images.size()
        expected_images_size = torch.Size([7, 3, 224, 224])
        assert_tensors_equal(actual_images_size, expected_images_size)

        actual_texts_size = transformed_texts.size()
        expected_texts_size = torch.Size([7, self.context_length])
        assert_tensors_equal(actual_texts_size, expected_texts_size)

        # Check encoding of long text
        actual_long_text = transformed_texts[-1]
        expected_long_text = torch.tensor(
            [self.bos_token] + self.text1_tokens[1:-1] * 20 + [self.eos_token],
            dtype=torch.long,
        )[: self.context_length]
        assert_tensors_equal(actual_long_text, expected_long_text)

        # Check zero padding for short texts
        actual_zero_pad_val = transformed_texts[:-1, self.text1_token_len :].max()
        expected_zero_pad_val = torch.tensor(0)
        assert_tensors_equal(actual_zero_pad_val, expected_zero_pad_val)
