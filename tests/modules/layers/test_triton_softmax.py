# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

import pytest
import torch

from tests.test_utils import assert_expected, gpu_test, set_rng_seed
from torchmultimodal.triton.layers.softmax import fused_softmax


@pytest.fixture(autouse=True)
def set_seed():
    set_rng_seed(2020)


@gpu_test()
class TestForwardSoftMax:
    def test_forward_2d_float32(
        self,
    ):
        # float32
        seq_len = 768

        sample_constant_float32 = torch.ones(
            (seq_len, seq_len), dtype=torch.float32, device="cuda"
        )
        sample_random_float32 = torch.randn_like(sample_constant_float32)

        expected_out_constant32 = torch.softmax(sample_constant_float32, dim=1)
        expected_out_random32 = torch.softmax(sample_random_float32, dim=1)

        triton_out_c32 = fused_softmax(sample_constant_float32)
        triton_out_random32 = fused_softmax(sample_random_float32)

        assert_expected(triton_out_c32, expected_out_constant32)
        assert_expected(triton_out_random32, expected_out_random32)

    def test_forward_2d_bfloat16(
        self,
    ):
        # bfloat16
        seq_len = 2048
        sample_constant_bf16 = torch.ones(
            (seq_len, seq_len), dtype=torch.bfloat16, device="cuda"
        )
        sample_random_bf16 = torch.randn_like(sample_constant_bf16)

        expected_out_c_bf16 = torch.softmax(sample_constant_bf16, dim=1)
        expected_out_rand_bf16 = torch.softmax(sample_random_bf16, dim=1)

        triton_out_c_bf16 = fused_softmax(sample_constant_bf16)
        triton_out_rand_bf16 = fused_softmax(sample_random_bf16)

        assert_expected(triton_out_c_bf16, expected_out_c_bf16)
        assert_expected(triton_out_rand_bf16, expected_out_rand_bf16)

    def test_forward_3d_bfloat16(
        self,
    ):
        # bfloat16
        seq_len = 2048
        batch = 12

        sample_constant_bf16 = torch.ones(
            (batch, seq_len, seq_len), dtype=torch.bfloat16, device="cuda"
        )
        sample_random_bf16 = torch.randn_like(sample_constant_bf16)

        expected_out_c_bf16 = torch.softmax(sample_constant_bf16, dim=1)
        expected_out_rand_bf16 = torch.softmax(sample_random_bf16, dim=1)

        triton_out_c_bf16 = fused_softmax(sample_constant_bf16)
        triton_out_rand_bf16 = fused_softmax(sample_random_bf16)

        assert_expected(triton_out_c_bf16, expected_out_c_bf16, rtol=1e-4, atol=1e-2)
        assert_expected(
            triton_out_rand_bf16, expected_out_rand_bf16, rtol=1e-4, atol=1e-2
        )


@gpu_test()
class TestBackwardSoftMax:
    def test_backward_2d(
        self,
    ):
        seq_len = 1024

        sample_constant_float32 = torch.ones(
            (seq_len, seq_len), dtype=torch.float32, device="cuda", requires_grad=True
        )
        sample_random_float32 = torch.randn_like(
            sample_constant_float32, requires_grad=True
        )

        expected_fwd_constant32 = torch.softmax(sample_constant_float32, dim=1)
        expected_fwd_random32 = torch.softmax(sample_random_float32, dim=1)

        triton_fwd_c32 = fused_softmax(sample_constant_float32)
        triton_fwd_random32 = fused_softmax(sample_random_float32)

        dout = torch.randn_like(sample_constant_float32)

        expected_bwd_c32 = expected_fwd_constant32.backward(dout)
        expected_bwd_r32 = expected_fwd_random32.backward(dout)

        triton_bwd_c32 = triton_fwd_c32.backward(dout)
        triton_bwd_r32 = triton_fwd_random32.backward(dout)

        assert_expected(triton_bwd_c32, expected_bwd_c32)
        assert_expected(triton_bwd_r32, expected_bwd_r32)

    def test_bwd_3d(
        self,
    ):
        seq_len = 2048
        batch = 4

        sample_constant_float32 = torch.ones(
            (batch, seq_len, seq_len),
            dtype=torch.float32,
            device="cuda",
            requires_grad=True,
        )
        sample_random_float32 = torch.randn_like(
            sample_constant_float32, requires_grad=True
        )

        expected_fwd_constant32 = torch.softmax(sample_constant_float32, dim=1)
        expected_fwd_random32 = torch.softmax(sample_random_float32, dim=1)

        triton_fwd_c32 = fused_softmax(sample_constant_float32)
        triton_fwd_random32 = fused_softmax(sample_random_float32)

        dout = torch.randn_like(sample_constant_float32)

        expected_bwd_c32 = expected_fwd_constant32.backward(dout)
        expected_bwd_r32 = expected_fwd_random32.backward(dout)

        triton_bwd_c32 = triton_fwd_c32.backward(dout)
        triton_bwd_r32 = triton_fwd_random32.backward(dout)

        assert_expected(triton_bwd_c32, expected_bwd_c32)
        assert_expected(triton_bwd_r32, expected_bwd_r32)
