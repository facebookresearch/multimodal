# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest

import torch

from test.test_utils import assert_expected
from torch import nn
from torchmultimodal.modules.layers.transformer_decoder import checkpoint_wrapper


class TestCheckpointWrapper:
    @pytest.fixture
    def model(self):
        class DummyModule(nn.Module):
            def __init__(self):
                super().__init__()
                self.param = nn.Parameter(torch.ones(3))

            @checkpoint_wrapper
            def forward(self, x, y, attn_mask=None, use_cache=False):
                u = x if use_cache else y
                u = u * self.param
                if attn_mask is not None:
                    u = u * attn_mask
                return u

        return DummyModule()

    @pytest.fixture
    def inputs(self):
        # Set inputs to be requires_grad = True or else it will break autograd
        # during the backward pass,
        # "RuntimeError: element 0 of tensors does not require grad and does not have a grad_fn"
        x = torch.ones(3, requires_grad=True)
        y = torch.ones(3, requires_grad=True) * 2
        attn_mask = torch.tensor([1, 1, 0])

        return x, y, attn_mask

    @pytest.fixture
    def compute_grad(self, model):
        model.zero_grad()

        def _compute_grad(output):
            output.sum().backward()
            grad_checkpointed = {}
            for name, param in model.named_parameters():
                grad_checkpointed[name] = param.grad.data.clone()
            return grad_checkpointed

        return _compute_grad

    def test_training_mode(self, model, inputs, compute_grad):
        """Test training mode that checkpoint is on"""
        model.train()
        x, y, attn_mask = inputs
        actual = model(x, y, attn_mask=attn_mask, use_cache=True)
        expected = torch.tensor([2.0, 2.0, 0.0])
        assert_expected(actual, expected)
        actual_grad = compute_grad(actual)
        assert_expected(actual_grad["param"], torch.tensor([2.0, 2.0, 0.0]))

    def test_eval_model(self, model, inputs, compute_grad):
        """Test eval mode that checkpoint is off"""
        model.eval()
        x, y, attn_mask = inputs
        actual = model(x, y, attn_mask=attn_mask, use_cache=True)
        expected = torch.tensor([1.0, 1.0, 0.0])
        assert_expected(actual, expected)
        actual_grad = compute_grad(actual)
        assert_expected(actual_grad["param"], torch.tensor([1.0, 1.0, 0.0]))
