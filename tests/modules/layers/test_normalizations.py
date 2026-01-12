# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
from tests.test_utils import assert_expected
from torchmultimodal.modules.layers.normalizations import (
    Fp32GroupNorm,
    Fp32LayerNorm,
    RMSNorm,
    SimpleRMSNorm,
)


def test_fp32layernorm():
    x = torch.ones(1, 1, dtype=torch.float16)
    norm = Fp32LayerNorm(1)
    output = norm(x)
    assert output.dtype == torch.float16


def test_fp32groupnorm():
    x = torch.ones(2, 4, dtype=torch.float16)
    norm = Fp32GroupNorm(2, 4)
    output = norm(x)
    assert output.dtype == torch.float16


def test_rms_norm_core_algo():
    """compare RMSNorm with RMSNorm using F.norm version"""
    dims = 10
    rms_norm = RMSNorm(dims)

    input_ones = torch.ones(dims, dtype=torch.float)

    input_fixed = torch.tensor(
        [0.999, 1.1111, 2.222, 3.333, 4.444, 5.555, 6.678, 7.987, 8.123, 9.101010],
        dtype=torch.float16,
    )
    fixed_expected = torch.tensor(
        [
            0.1749,
            0.1946,
            0.3892,
            0.5835,
            0.7783,
            0.9727,
            1.1699,
            1.3984,
            1.4229,
            1.5938,
        ],
        dtype=torch.float,
    )

    output_fixed = rms_norm(input_fixed)
    output_ones = rms_norm(input_ones)

    assert_expected(output_ones, input_ones)
    assert_expected(output_fixed, fixed_expected, atol=1e-04, rtol=1e-05)
    assert output_fixed.dtype == torch.float32


def test_simple_rmsnorm():
    dims = 12
    srms_norm = SimpleRMSNorm(dims)

    input_bf16_ones = torch.ones(dims, dtype=torch.bfloat16)

    input_fixed_fp32 = torch.tensor(
        [
            0.999,
            1.1111,
            2.222,
            3.333,
            4.444,
            5.555,
            6.678,
            7.987,
            8.123,
            9.101010,
            110.00,
            120.2589,
        ],
        dtype=torch.float32,
    )

    expected_output_bf16_ones = torch.tensor(
        [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        dtype=torch.bfloat16,
    )
    expected_output_fixed = torch.tensor(
        [
            0.0211,
            0.0235,
            0.0469,
            0.0704,
            0.0939,
            0.1174,
            0.1411,
            0.1687,
            0.1716,
            0.1923,
            2.3238,
            2.5405,
        ],
        dtype=torch.float32,
    )

    actual_output_bf16_ones = srms_norm(input_bf16_ones)
    actual_output_fixed = srms_norm(input_fixed_fp32)

    # verify ones output and dtype
    assert_expected(
        actual_output_bf16_ones, expected_output_bf16_ones, atol=1e-04, rtol=1e-05
    )
    assert actual_output_bf16_ones.dtype == torch.bfloat16

    # verify fixed output and dtype
    assert_expected(actual_output_fixed, expected_output_fixed, atol=1e-04, rtol=1e-05)
    assert actual_output_fixed.dtype == torch.float32
