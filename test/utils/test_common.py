# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest

import torch

from test.test_utils import assert_expected
from torch import nn
from torchmultimodal.utils.common import (
    checkpoint_wrapper,
    shift_dim,
    tensor_slice,
    to_tuple_tuple,
)


def test_shift_dim():
    test_random_tensor = torch.randn(2, 2, 2, 2, 2)
    actual = shift_dim(test_random_tensor, 1, -1)
    expected = test_random_tensor.permute(0, 2, 3, 4, 1).contiguous()
    assert_expected(actual, expected)

    actual = shift_dim(test_random_tensor, -3, 3)
    expected = test_random_tensor.permute(0, 1, 3, 2, 4).contiguous()
    assert_expected(actual, expected)


class TestTensorSlice:
    @pytest.fixture(scope="class")
    def test_input(self):
        return torch.tensor([[[0, 1], [2, 3], [5, 6]]])

    def test_default(self, test_input):
        actual = tensor_slice(test_input, [0, 1, 0], [1, 1, 2])
        expected = torch.tensor([[[2, 3]]])
        assert_expected(actual, expected)

    def test_size_minus_one(self, test_input):
        """Test size -1"""
        actual = tensor_slice(test_input, [0, 1, 0], [1, -1, 2])
        expected = torch.tensor([[[2, 3], [5, 6]]])
        assert_expected(actual, expected)

    def test_uneven_begin_size(self, test_input):
        """Test uneven begin and size vectors"""
        actual = tensor_slice(test_input, [0, 1, 0], [1, 1])
        expected = torch.tensor([[[2, 3]]])
        assert_expected(actual, expected)

        actual = tensor_slice(test_input, [0, 1], [1, 1, 2])
        expected = torch.tensor([[[2, 3]]])
        assert_expected(actual, expected)

    @pytest.mark.xfail(raises=ValueError, reason="Invalid begin")
    def test_invalid_begin(self, test_input):
        tensor_slice(test_input, [-1, 1, 0], [1, 1, 2])

    @pytest.mark.xfail(raises=ValueError, reason="Invalid size")
    def test_invalid_size(self, test_input):
        tensor_slice(test_input, [0, 1, 0], [-2, 1, 2])


class TestToTupleTuple:
    @pytest.fixture(scope="class")
    def expected(self):
        return ((2, 2, 2), (2, 2, 2), (2, 2, 2))

    def test_int(self, expected):
        actual = to_tuple_tuple(2, 3, 3)
        assert actual == expected, "int -> tuple[tuple] incorrect"

    def test_tuple(self, expected):
        actual = to_tuple_tuple((2, 2, 2), 3, 3)
        assert actual == expected, "tuple -> tuple[tuple] incorrect"


class TestCheckpointWrapper:
    @pytest.fixture
    def device(self):
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    @pytest.fixture
    def model(self, device):
        class DummyAttention(nn.Module):
            def __init__(self):
                super().__init__()
                self.param = nn.Parameter(torch.ones(3))

            def forward(self, x, y, attn_mask=None, use_cache=None):
                u = x if use_cache else y
                u = u * self.param
                if attn_mask is not None:
                    u = u * attn_mask
                return u

        class DummyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.attention = DummyAttention()
                self.identity = nn.Identity()

            @checkpoint_wrapper
            def _layer_to_wrap(self, x, y, attn_mask, use_cache):
                return self.attention(x, y, attn_mask, use_cache)

            def forward(self, x, y, attn_mask=None, use_cache=False):
                out = self._layer_to_wrap(x, y, attn_mask, use_cache=use_cache)
                out = self.identity(out)
                return out

        return DummyModel().to(device)

    @pytest.fixture
    def inputs(self, device):
        # Set inputs to be requires_grad = True or else it will break autograd
        # during the backward pass,
        # "RuntimeError: element 0 of tensors does not require grad and does not have a grad_fn"
        x = torch.ones(3, requires_grad=True)
        y = torch.ones(3, requires_grad=True) * 2
        attn_mask = torch.tensor([1, 1, 0])
        x = x.to(device)
        y = y.to(device)
        attn_mask = attn_mask.to(device)

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
        with pytest.warns(UserWarning):
            model.train()
            x, y, attn_mask = inputs
            actual = model(x, y, attn_mask=attn_mask, use_cache=True)
            expected = torch.tensor([2.0, 2.0, 0.0])
            assert_expected(actual, expected)
            actual_grad = compute_grad(actual)
            assert_expected(
                actual_grad["attention.param"], torch.tensor([2.0, 2.0, 0.0])
            )

    def test_eval_model(self, model, inputs, compute_grad):
        """Test eval mode that checkpoint is off"""
        model.eval()
        x, y, attn_mask = inputs
        actual = model(x, y, attn_mask=attn_mask, use_cache=True)
        expected = torch.tensor([1.0, 1.0, 0.0])
        assert_expected(actual, expected)
        actual_grad = compute_grad(actual)
        assert_expected(actual_grad["attention.param"], torch.tensor([1.0, 1.0, 0.0]))
