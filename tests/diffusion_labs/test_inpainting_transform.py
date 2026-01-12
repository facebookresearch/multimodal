#!/usr/bin/env fbpython
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import numpy as np
import torch
from PIL import Image
from tests.test_utils import assert_expected, set_rng_seed
from torchmultimodal.diffusion_labs.transforms.inpainting_transform import (
    brush_stroke_mask_image,
    draw_strokes,
    generate_vertexes,
    mask_full_image,
    random_inpaint_mask_image,
    random_outpaint_mask_image,
    RandomInpaintingMask,
)

BATCH = 4
CHANNELS = 3
IMAGE_HEIGHT = 256
IMAGE_WIDTH = 256


def set_seed(seed: int):
    set_rng_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class TestImageMasks(unittest.TestCase):
    def setUp(self):
        set_seed(1)
        self.batch_images = torch.randn(BATCH, CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH)
        self.image = self.batch_images[0, :, :, :]

    def test_random_inpaint_mask_image(self):
        set_seed(1)
        mask = random_inpaint_mask_image(self.image)
        self.assertIsInstance(mask, torch.Tensor)
        self.assertEqual(mask.shape, (1, self.image.shape[-2], self.image.shape[-1]))
        assert_expected(mask.sum(), torch.tensor(11524.0), rtol=0, atol=1e-4)

    def test_random_outpaint_mask_image(self):
        set_seed(1)
        mask = random_outpaint_mask_image(self.image)
        self.assertIsInstance(mask, torch.Tensor)
        self.assertEqual(mask.shape, (1, self.image.shape[-2], self.image.shape[-1]))
        assert_expected(mask.sum(), torch.tensor(27392.0), rtol=0, atol=1e-4)

    def test_brush_stroke_mask_image(self):
        set_seed(1)
        mask = brush_stroke_mask_image(self.image)
        self.assertIsInstance(mask, torch.Tensor)
        self.assertEqual(mask.shape, (1, self.image.shape[-2], self.image.shape[-1]))
        print(f"test_brush_stroke_mask_image: {mask.sum().item()}")
        assert_expected(mask.sum(), torch.tensor(26860.0), rtol=0, atol=1e-4)

    def test_mask_full_image(self):
        set_seed(1)
        mask = mask_full_image(self.image)
        self.assertIsInstance(mask, torch.Tensor)
        self.assertEqual(mask.shape, (1, self.image.shape[-2], self.image.shape[-1]))
        self.assertTrue(torch.allclose(mask, torch.ones_like(mask)))
        assert_expected(mask.sum(), torch.tensor(65536.0), rtol=0, atol=1e-4)

    def test_generate_vertexes(self):
        mask = Image.new("1", (IMAGE_WIDTH, IMAGE_HEIGHT), 0)
        vertexes = generate_vertexes(
            mask, num_vertexes=3, img_width=IMAGE_WIDTH, img_height=IMAGE_HEIGHT
        )
        self.assertIsInstance(vertexes, list)
        self.assertEqual(len(vertexes), 4)
        for vertex in vertexes:
            self.assertIsInstance(vertex, tuple)
            self.assertEqual(len(vertex), 2)
            self.assertTrue(0 <= vertex[0] < IMAGE_WIDTH)
            self.assertTrue(0 <= vertex[1] < IMAGE_HEIGHT)

    def test_draw_strokes(self):
        mask = Image.new("1", (IMAGE_WIDTH, IMAGE_HEIGHT), 0)
        vertexes = [(10, 10), (20, 20), (30, 30)]
        draw_strokes(mask, vertexes, width=2)
        self.assertIsInstance(mask, Image.Image)

    def test_generate_vertexes_and_draw_strokes(self):
        mask = Image.new("1", (IMAGE_WIDTH, IMAGE_HEIGHT), 0)

        vertexes = generate_vertexes(
            mask, num_vertexes=3, img_width=IMAGE_WIDTH, img_height=IMAGE_HEIGHT
        )
        draw_strokes(mask, vertexes, width=2)
        self.assertIsInstance(mask, Image.Image)

    def test_random_mask(self):
        random_mask = RandomInpaintingMask()
        inpainting_mask = random_mask({"x": self.batch_images})["mask"]
        assert inpainting_mask.shape == (BATCH, 1, IMAGE_HEIGHT, IMAGE_WIDTH)
        assert torch.all(
            torch.logical_or(inpainting_mask == 0.0, inpainting_mask == 1.0)
        )
