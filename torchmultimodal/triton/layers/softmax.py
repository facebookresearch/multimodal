# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# ---- Fused Softmax written in Triton ------
# Extra Credits:
# Triton Softmax Tutorial
# LucidRains Triton_Transformers

from typing import Tuple

import torch
import triton
import triton.language as tl

from torch import autograd


def _get_num_warps(block_size: int) -> int:
    num_warps = 4
    if block_size > 2047:
        num_warps = 8
    if block_size > 4095:
        num_warps = 16
    return num_warps


@triton.jit
def _softmax_kernel_fwd(
    output_ptr,
    output_row_stride,
    input_ptr,
    input_row_stride,
    n_cols,
    block_size: tl.constexpr,
) -> None:

    # setup input location
    row_index = tl.program_id(0)
    input_row_ptr = input_ptr + (row_index * input_row_stride)
    col_offsets = tl.arange(0, block_size)
    input_ptrs = input_row_ptr + col_offsets
    rw_mask = col_offsets < n_cols
    row = tl.load(input_ptrs, mask=rw_mask, other=float("-inf"))

    # safe softmax proper
    safe_row = row - tl.max(row, axis=0)
    numerator = tl.exp(safe_row)
    denom = tl.sum(numerator, axis=0)
    sm_out = numerator / denom

    # write results to HBM
    out_row_ptr = output_ptr + (row_index * output_row_stride)
    out_row_ptrs = out_row_ptr + col_offsets
    tl.store(out_row_ptrs, sm_out, mask=rw_mask)


@triton.jit
def _softmax_kernel_bwd(
    output_ptr,
    stride_output_row,
    grad_ptr,
    stride_grad_row,
    input_ptr,
    stride_input_row,
    n_cols,
    block_size: tl.constexpr,
) -> None:
    # setup input locations - need both grad and input access
    row_index = tl.program_id(0)

    input_row_ptr = input_ptr + (row_index * stride_input_row)
    grad_row_ptr = grad_ptr + (row_index * stride_grad_row)

    col_offsets = tl.arange(0, block_size)
    rw_mask = col_offsets < n_cols

    input_row_ptrs = input_row_ptr + col_offsets
    grad_row_ptrs = grad_row_ptr + col_offsets

    probs_row = tl.load(input_row_ptrs, mask=rw_mask, other=0)
    grads_row = tl.load(grad_row_ptrs, mask=rw_mask, other=0)

    # compute derivatives
    dx = probs_row * grads_row
    dsm_out = dx - probs_row * (tl.sum(dx, axis=0))

    # write to HBM
    output_row_ptr = output_ptr + (row_index * stride_output_row)
    output_ptrs = output_row_ptr + col_offsets
    tl.store(output_ptrs, dsm_out, mask=rw_mask)


class TritonSoftmax(autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:
        orig_shape = x.shape
        x = x.view(-1, orig_shape[-1])
        nrows, ncols = x.shape

        block_size = triton.next_power_of_2(ncols)
        num_warps = _get_num_warps(block_size)

        res = torch.empty_like(x)
        grid = (nrows,)

        _softmax_kernel_fwd[grid](
            res,
            res.stride(0),
            x,
            x.stride(0),
            ncols,
            block_size=block_size,
            num_warps=num_warps,
        )

        if x.requires_grad:
            ctx.save_for_backward(res)

        return res.view(*orig_shape)

    @staticmethod
    def backward(ctx, grad_probs: torch.Tensor) -> Tuple[torch.Tensor, None]:
        orig_shape = grad_probs.shape
        (probs,) = ctx.saved_tensors

        grad_probs = grad_probs.view(-1, orig_shape[-1])
        nrows, ncols = grad_probs.shape

        block_size = triton.next_power_of_2(ncols)
        num_warps = _get_num_warps(block_size)

        dx = torch.empty_like(probs)
        grid = (nrows,)

        _softmax_kernel_bwd[grid](
            dx,
            dx.stride(0),
            probs,
            probs.stride(0),
            grad_probs,
            grad_probs.stride(0),
            ncols,
            block_size=block_size,
            num_warps=num_warps,
        )
        return dx.view(*orig_shape), None


fused_softmax = TritonSoftmax.apply
