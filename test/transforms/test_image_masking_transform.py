# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch
from test.test_utils import assert_expected, set_rng_seed
from torchmultimodal.transforms.image_masking_transform import (
    MaskedImageModelingTransform,
)
from torchvision import transforms


class TestImageMaskingTransforms:
    @pytest.fixture
    def set_seed(self):
        set_rng_seed(1234)

    def test_image_masking_train(self, set_seed):
        transform = MaskedImageModelingTransform(
            encoder_input_size=3,
            codebook_input_size=3,
            mask_max_patches=1,
            mask_min_patches=1,
            mask_num_patches=1,
        )
        input = transforms.ToPILImage()(torch.ones(2, 2))
        out = transform(input)
        expected_image = torch.Tensor(
            [
                [
                    [1.9303, 1.9303, 1.9303],
                    [1.9303, 1.9303, 1.9303],
                    [1.9303, 1.9303, 1.9303],
                ],
                [
                    [2.0749, 2.0749, 2.0749],
                    [2.0749, 2.0749, 2.0749],
                    [2.0749, 2.0749, 2.0749],
                ],
                [
                    [2.1459, 2.1459, 2.1459],
                    [2.1459, 2.1459, 2.1459],
                    [2.1459, 2.1459, 2.1459],
                ],
            ]
        )

        assert_expected(out["image"], expected_image, atol=1e-4, rtol=1e-4)
        assert_expected(out["image_for_codebook"], torch.full((3, 3, 3), 0.9))
        assert out["image_patches_mask"].sum() == 1
