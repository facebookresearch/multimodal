# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from copy import deepcopy

import pytest
import torch
import torch.nn as nn
import torch.optim as optim

from tests.test_utils import assert_expected, set_rng_seed
from torchmultimodal.modules.optimizers.anyprecision import AnyPrecisionAdamW


@pytest.fixture(autouse=True)
def random():
    set_rng_seed(2020)


class TestAnyPrecisionOptimizer:
    def _test_adam_equivalence(self, model, model_clone):
        # Test non-default options
        betas = (0.8, 0.88)
        weight_decay = 0.03

        adam_opt = optim.AdamW(
            model_clone.parameters(), betas=betas, weight_decay=weight_decay
        )
        anyprecision_adam = AnyPrecisionAdamW(
            model.parameters(),
            variance_dtype=torch.float32,
            betas=betas,
            weight_decay=weight_decay,
        )

        # Verify params are equal initially
        model_orig_params = [p.clone() for p in model.parameters()]
        for p1, p2 in zip(model_clone.parameters(), model_orig_params):
            assert_expected(p1, p2)

        for i in range(6):
            adam_opt.zero_grad()
            anyprecision_adam.zero_grad()
            inp = torch.randn(5, 5, device=next(model.parameters()).device)
            model(inp).sum().backward()
            model_clone(inp).sum().backward()
            adam_opt.step()
            anyprecision_adam.step()

            # Ensure params are modified from original
            if i == 0:
                for p1, p2 in zip(model.parameters(), model_orig_params):
                    assert not torch.equal(p1, p2)

            for p1, p2 in zip(model.parameters(), model_clone.parameters()):
                assert_expected(p1, p2)

    def test_adam_equivalence(self, device="cpu"):
        """
        Tests that AnyPrecisionAdamW is equivalent to AdamW when
        kahan summation and different dtypes for momentum, variance,
        and compensation buffer are turned off (i.e. all float32).
        """
        if device == "cuda" and not torch.cuda.is_available():
            # raise unittest.SkipTest("CUDA not available")
            pytest.skip(reason="CUDA not available")

        model = nn.Sequential(nn.Linear(5, 5), nn.Linear(5, 5), nn.Linear(5, 5))
        if device == "cuda":
            model.cuda()

        model_clone = deepcopy(model)

        self._test_adam_equivalence(model, model_clone)

    def test_bfloat16_states(self, device="cpu"):
        """verify that AnyPrecision is running using bfloat16 for states when specified (momentum, variance)"""
        simple_model = nn.Sequential(
            nn.Linear(5, 10, dtype=torch.bfloat16),
            nn.GELU(),
            nn.Linear(10, 2, dtype=torch.bfloat16),
        )
        simple_model.to(torch.bfloat16)
        simple_model.to(device)

        anyprecision_adam = AnyPrecisionAdamW(
            simple_model.parameters(),
            variance_dtype=torch.bfloat16,
            momentum_dtype=torch.bfloat16,
        )

        for i in range(6):
            anyprecision_adam.zero_grad()
            inp = torch.randn(
                5,
                5,
                dtype=torch.bfloat16,
                device=next(simple_model.parameters()).device,
            )
            simple_model(inp).sum().backward()
            anyprecision_adam.step()

        for group in anyprecision_adam.param_groups:
            for p in group["params"]:
                state = anyprecision_adam.state[p]
                assert state["exp_avg"].dtype == torch.bfloat16
                assert state["exp_avg_sq"].dtype == torch.bfloat16

    def test_kahan_summation(self, device="cpu"):
        """verify that AnyPrecision is properly using Kahan summation when specified (momentum, variance).
        uses precomputed result tensors as comparison for the compensation buffers."""
        simple_model = nn.Sequential(nn.Linear(5, 10), nn.GELU(), nn.Linear(10, 2))
        simple_model.to(torch.bfloat16)
        simple_model.to(device)

        anyprecision_adam = AnyPrecisionAdamW(
            simple_model.parameters(),
            variance_dtype=torch.bfloat16,
            momentum_dtype=torch.bfloat16,
            use_kahan_summation=True,
            compensation_buffer_dtype=torch.bfloat16,
        )

        # pre-computed kahan buffer tensors for comparing results.
        # values determined by comparison to reference implementation.
        expected_kahan_buffer_param0 = torch.tensor(
            [
                [-1.5259e-05, 8.2397e-04, 1.7166e-04, 5.1880e-04, 1.1444e-05],
                [-4.5776e-05, -1.1826e-04, 9.1553e-04, 3.0518e-04, -4.3869e-05],
                [4.5776e-05, -2.8229e-04, 5.6028e-05, 2.8610e-06, -8.0872e-04],
                [2.6703e-05, -2.6703e-04, 5.0068e-05, -2.9755e-04, 1.0014e-04],
                [2.8610e-05, -4.1962e-05, -6.7139e-04, -9.9659e-05, 3.8147e-06],
                [-3.0518e-05, -2.1839e-04, 6.0320e-05, 2.8992e-04, -7.6294e-06],
                [0.0000e00, -3.4332e-04, 1.7166e-04, 4.8828e-04, 3.3951e-04],
                [-2.8849e-05, -1.2457e-05, -1.1444e-05, -6.3324e-04, -4.9591e-05],
                [-3.5286e-05, 3.4332e-04, -4.9353e-05, 9.4223e-04, -3.7956e-04],
                [5.4932e-04, -7.6294e-06, 3.4523e-04, 3.3760e-04, 4.5586e-04],
            ],
            dtype=torch.bfloat16,
        )

        expected_kahan_buffer_param1 = torch.tensor(
            [
                4.4823e-05,
                -4.7445e-05,
                4.5776e-05,
                4.5538e-05,
                4.5776e-05,
                3.8624e-05,
                2.8849e-05,
                1.5259e-05,
                -5.5313e-05,
                1.4114e-04,
            ],
            dtype=torch.bfloat16,
        )

        expected_kahan_buffer_param2 = torch.tensor(
            [
                [
                    -5.3406e-05,
                    -1.2815e-05,
                    0.0000e00,
                    -4.2725e-04,
                    -3.4332e-05,
                    2.2888e-05,
                    1.2589e-04,
                    -3.1281e-04,
                    0.0000e00,
                    -4.4632e-04,
                ],
                [
                    -5.3406e-05,
                    -1.5259e-05,
                    0.0000e00,
                    -4.2725e-04,
                    4.5586e-04,
                    2.2888e-05,
                    3.8147e-06,
                    -7.6294e-06,
                    0.0000e00,
                    4.1962e-05,
                ],
            ],
            dtype=torch.bfloat16,
        )

        expected_kahan_buffer_param3 = torch.tensor(
            [-4.5776e-05, -1.5259e-05], dtype=torch.bfloat16
        )

        expected_kahan_buffers = [
            expected_kahan_buffer_param0,
            expected_kahan_buffer_param1,
            expected_kahan_buffer_param2,
            expected_kahan_buffer_param3,
        ]

        for i in range(2):
            anyprecision_adam.zero_grad()
            inp = torch.randn(
                5,
                5,
                dtype=torch.bfloat16,
                device=next(simple_model.parameters()).device,
            )
            simple_model(inp).sum().backward()
            anyprecision_adam.step()

        for group in anyprecision_adam.param_groups:
            for index, p in enumerate(group["params"]):
                state = anyprecision_adam.state[p]
                pcomp = state["compensation"]
                assert pcomp.dtype == torch.bfloat16

                assert_expected(
                    pcomp,
                    expected_kahan_buffers[index],
                    atol=1e-4,
                    rtol=1e-4,
                )
