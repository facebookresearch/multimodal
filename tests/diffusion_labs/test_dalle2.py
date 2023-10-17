#!/usr/bin/env fbpython
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
from PIL import Image
from tests.test_utils import assert_expected, set_rng_seed
from torchmultimodal.diffusion_labs.models.dalle2.dalle2_decoder import dalle2_decoder
from torchmultimodal.diffusion_labs.models.dalle2.transforms import Dalle2ImageTransform


def test_dalle2_model():
    set_rng_seed(4)
    model = dalle2_decoder(
        timesteps=1,
        time_embed_dim=1,
        cond_embed_dim=1,
        clip_embed_dim=1,
        clip_embed_name="clip_image",
        predict_variance_value=True,
        image_channels=1,
        depth=32,
        num_resize=1,
        num_res_per_layer=1,
        use_cf_guidance=True,
        clip_image_guidance_dropout=0.1,
        guidance_strength=7.0,
        learn_null_emb=True,
    )
    model.eval()
    x = torch.randn(1, 1, 4, 4)
    c = torch.ones((1, 1))
    with torch.no_grad():
        actual = model(x, conditional_inputs={"clip_image": c}).mean()
    expected = torch.as_tensor(0.12768)
    assert_expected(actual, expected, rtol=0, atol=1e-4)


def test_dalle2_image_transform():
    img_size = 5
    transform = Dalle2ImageTransform(image_size=img_size, image_min=-1, image_max=1)
    image = Image.new("RGB", size=(20, 20), color=(128, 0, 0))
    actual = transform({"x": image})["x"].sum()
    normalized128 = 128 / 255 * 2 - 1
    normalized0 = -1
    expected = torch.tensor(
        normalized128 * img_size**2 + 2 * normalized0 * img_size**2
    )
    assert_expected(actual, expected, rtol=0, atol=1e-4)
