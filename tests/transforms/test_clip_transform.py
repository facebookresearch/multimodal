# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest

import torch
from tests.test_utils import assert_expected, get_asset_path, set_rng_seed
from torchmultimodal.transforms.clip_transform import (
    CLIPImageTransform,
    CLIPTextTransform,
    CLIPTransform,
)
from torchvision.transforms import ToPILImage


class TestCLIPTransform:
    @pytest.fixture()
    def context_length(self):
        return 77

    @pytest.fixture()
    def image1(self):
        return ToPILImage()(torch.ones(3, 300, 500))

    @pytest.fixture()
    def image2(self):
        return ToPILImage()(torch.ones(3, 50, 100))

    @pytest.fixture()
    def text1(self):
        return "Taken with my analogue EOS 500N with black & white film."

    @pytest.fixture()
    def text2(self):
        return "This is a shorter sentence."

    @pytest.fixture()
    def long_text(self, text1):
        return (text1 + " ") * 20

    @pytest.fixture()
    def text1_tokens(self):
        return [
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

    @pytest.fixture()
    def bpe_merges_file(self):
        return get_asset_path("clip_vocab.bpe")

    @pytest.fixture()
    def clip_transform(self, bpe_merges_file):
        return CLIPTransform(text_bpe_merges_path=bpe_merges_file)

    def setUp(self):
        set_rng_seed(1234)

    def test_clip_single_transform(
        self,
        context_length,
        image1,
        text1,
        text1_tokens,
        clip_transform,
    ):
        transformed_image, transformed_text = clip_transform(image=image1, text=text1)

        actual_image_size = transformed_image.size()
        expected_image_size = torch.Size([3, 224, 224])
        assert_expected(actual_image_size, expected_image_size)

        actual_text = transformed_text
        text1_token_len = len(text1_tokens)
        expected_text = torch.tensor(
            text1_tokens + [0] * (context_length - text1_token_len),
            dtype=torch.long,
        )
        assert_expected(actual_text, expected_text)

    def test_clip_multi_transform(
        self,
        context_length,
        image1,
        image2,
        text1,
        text2,
        long_text,
        text1_tokens,
        clip_transform,
    ):
        images = [image1] * 5 + [image2] * 2
        texts = [text1] * 5 + [text2] + [long_text]
        transformed_images, transformed_texts = clip_transform(image=images, text=texts)

        actual_images_size = transformed_images.size()
        expected_images_size = torch.Size([7, 3, 224, 224])
        assert_expected(actual_images_size, expected_images_size)

        actual_texts_size = transformed_texts.size()
        expected_texts_size = torch.Size([7, context_length])
        assert_expected(actual_texts_size, expected_texts_size)

        # Check encoding of long text
        actual_long_text = transformed_texts[-1]
        bos_token = text1_tokens[0]
        eos_token = text1_tokens[-1]
        expected_long_text = torch.tensor(
            [bos_token] + (text1_tokens[1:-1] * 20)[: context_length - 2] + [eos_token],
            dtype=torch.long,
        )
        assert_expected(actual_long_text, expected_long_text)

        # Check zero padding for short texts
        text1_token_len = len(text1_tokens)
        actual_zero_pad_val = transformed_texts[:-1, text1_token_len:].max()
        expected_zero_pad_val = torch.tensor(0)
        assert_expected(actual_zero_pad_val, expected_zero_pad_val)

    def test_clip_image_transform_int_resize(self, image1):
        image_transform = CLIPImageTransform(is_train=False)
        # check the first transform which corresponds to the resize
        transformed_image = image_transform.image_transform.transforms[0](image1)

        actual_image_size = transformed_image.size
        expected_image_size = (373, 224)
        assert_expected(actual_image_size, expected_image_size)

    def test_clip_image_transform_tuple_resize(self, image1):
        image_transform = CLIPImageTransform(image_size=(224, 224), is_train=False)
        # check the first transform which corresponds to the resize
        transformed_image = image_transform.image_transform.transforms[0](image1)

        actual_image_size = transformed_image.size
        expected_image_size = (224, 224)
        assert_expected(actual_image_size, expected_image_size)

    # Only text transforms require torchscripting for now based on user needs
    def test_scripting_text_transform(self, text1, bpe_merges_file):
        text_transform = CLIPTextTransform(text_bpe_merges_path=bpe_merges_file)
        scripted_text_transform = torch.jit.script(text_transform)
        assert_expected(text_transform(text1), scripted_text_transform(text1))
