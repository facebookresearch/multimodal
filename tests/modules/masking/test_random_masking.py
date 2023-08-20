# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

import pytest

import torch
from tests.test_utils import assert_expected, set_rng_seed
from torchmultimodal.modules.masking.random_masking import (
    random_masking,
    random_masking_2d,
)


@pytest.fixture(autouse=True)
def random():
    set_rng_seed(0)


class TestRandomMasking:
    @pytest.fixture
    def inputs(self):
        return torch.ones(1, 8, 2)

    @pytest.fixture
    def mask_ratio_valid(self):
        return 0.75

    @pytest.fixture
    def mask_ratio_invalid(self):
        return 1.2

    @pytest.fixture
    def mask_ratio_zero(self):
        return 0

    @pytest.mark.parametrize(
        """inputs_fixture,
        mask_ratio_fixture,
        expected_x_masked,
        expected_mask,
        expected_ids_restore,
        expected_ids_keep""",
        [
            (
                "inputs",
                "mask_ratio_valid",
                torch.tensor([[[1.0, 1.0], [1.0, 1.0]]]),
                torch.tensor([[1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]]),
                torch.tensor([[4, 6, 0, 1, 2, 5, 3, 7]]),
                torch.tensor([[2, 3]]),
            ),
            (
                "inputs",
                "mask_ratio_zero",
                torch.tensor(
                    [
                        [
                            [1.0, 1.0],
                            [1.0, 1.0],
                            [1.0, 1.0],
                            [1.0, 1.0],
                            [1.0, 1.0],
                            [1.0, 1.0],
                            [1.0, 1.0],
                            [1.0, 1.0],
                        ]
                    ]
                ),
                torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]),
                torch.tensor([[4, 6, 0, 1, 2, 5, 3, 7]]),
                torch.tensor([[2, 3, 4, 6, 0, 5, 1, 7]]),
            ),
        ],
    )
    def test_masking(
        self,
        request,
        inputs_fixture,
        mask_ratio_fixture,
        expected_x_masked,
        expected_mask,
        expected_ids_restore,
        expected_ids_keep,
    ):
        inputs = request.getfixturevalue(inputs_fixture)
        mask_ratio = request.getfixturevalue(mask_ratio_fixture)

        random_masking_output = random_masking(x=inputs, mask_ratio=mask_ratio)
        actual_x_masked = random_masking_output.x_masked
        actual_mask = random_masking_output.mask
        actual_ids_restore = random_masking_output.ids_restore
        actual_ids_keep = random_masking_output.ids_keep

        assert_expected(actual_x_masked, expected_x_masked)
        assert_expected(actual_mask, expected_mask)
        assert_expected(actual_ids_restore, expected_ids_restore)
        assert_expected(actual_ids_keep, expected_ids_keep)

    def test_masking_invalid(self, inputs, mask_ratio_invalid):
        with pytest.raises(AssertionError):
            random_masking(x=inputs, mask_ratio=mask_ratio_invalid)


class TestRandomMasking2D:
    @pytest.fixture
    def inputs(self):
        return torch.tensor(
            [[[1, 2], [3, 4], [5, 6], [7, 8], [2, 1], [4, 3], [6, 5], [8, 7]]]
        )

    def test_masking(self, inputs):
        out = random_masking_2d(
            x=inputs, mask_ratio_h=0, mask_ratio_w=0, num_patches_h=4, num_patches_w=2
        )
        assert_expected(
            out,
            torch.tensor(
                [[[2, 1], [4, 3], [6, 5], [8, 7], [1, 2], [3, 4], [5, 6], [7, 8]]]
            ),
        )

    def test_masking_all(self, inputs):
        out = random_masking_2d(
            x=inputs, mask_ratio_h=1, mask_ratio_w=1, num_patches_h=4, num_patches_w=2
        )
        assert_expected(out, torch.empty(size=(1, 0, 2), dtype=torch.int64))

    def test_masking_partial(self, inputs):
        out = random_masking_2d(
            x=inputs,
            mask_ratio_h=0.5,
            mask_ratio_w=0.5,
            num_patches_h=4,
            num_patches_w=2,
        )
        assert_expected(out, torch.tensor([[[2, 1], [6, 5]]]))
