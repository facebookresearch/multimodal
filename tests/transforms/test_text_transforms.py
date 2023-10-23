# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch
from tests.test_utils import assert_expected
from torchmultimodal.transforms.text_transforms import (
    add_token,
    AddToken,
    PadTransform,
    to_tensor,
    ToTensor,
    truncate,
    Truncate,
)


class TestTransforms:
    def _totensor(self, test_scripting):
        padding_value = 0
        transform = ToTensor(padding_value=padding_value)
        if test_scripting:
            transform = torch.jit.script(transform)
        inputs = [[1, 2], [1, 2, 3]]

        actual = transform(inputs)
        expected = torch.tensor([[1, 2, 0], [1, 2, 3]], dtype=torch.long)
        assert_expected(actual, expected)

        inputs = [1, 2]
        actual = transform(inputs)
        expected = torch.tensor([1, 2], dtype=torch.long)
        assert_expected(actual, expected)

    def test_totensor(self) -> None:
        """test tensorization on both single sequence and batch of sequence"""
        self._totensor(test_scripting=False)

    def test_totensor_jit(self) -> None:
        """test tensorization with scripting on both single sequence and batch of sequence"""
        self._totensor(test_scripting=True)

    def _truncate(self, test_scripting):
        max_seq_len = 2
        transform = Truncate(max_seq_len=max_seq_len)
        if test_scripting:
            transform = torch.jit.script(transform)

        inputs = [[1, 2], [1, 2, 3]]
        actual = transform(inputs)
        expected = [[1, 2], [1, 2]]
        assert_expected(actual, expected)

        inputs = [1, 2, 3]
        actual = transform(inputs)
        expected = [1, 2]
        assert_expected(actual, expected)

        inputs = [["a", "b"], ["a", "b", "c"]]
        actual = transform(inputs)
        expected = [["a", "b"], ["a", "b"]]
        assert actual == expected

        inputs = ["a", "b", "c"]
        actual = transform(inputs)
        expected = ["a", "b"]
        assert actual == expected

    def test_truncate(self) -> None:
        """test truncation on both sequence and batch of sequence with both str and int types"""
        self._truncate(test_scripting=False)

    def test_truncate_jit(self) -> None:
        """test truncation with scripting on both sequence and batch of sequence with both str and int types"""
        self._truncate(test_scripting=True)

    def _add_token(self, test_scripting):
        token_id = 0
        transform = AddToken(token_id, begin=True)
        if test_scripting:
            transform = torch.jit.script(transform)
        inputs = [[1, 2], [1, 2, 3]]

        actual = transform(inputs)
        expected = [[0, 1, 2], [0, 1, 2, 3]]
        assert_expected(actual, expected)

        transform = AddToken(token_id, begin=False)
        if test_scripting:
            transform = torch.jit.script(transform)

        actual = transform(inputs)
        expected = [[1, 2, 0], [1, 2, 3, 0]]
        assert_expected(actual, expected)

        inputs = [1, 2]
        actual = transform(inputs)
        expected = [1, 2, 0]
        assert_expected(actual, expected)

        token_id = "0"
        transform = AddToken(token_id, begin=True)
        if test_scripting:
            transform = torch.jit.script(transform)
        inputs = [["1", "2"], ["1", "2", "3"]]

        actual = transform(inputs)
        expected = [["0", "1", "2"], ["0", "1", "2", "3"]]
        assert actual == expected

        transform = AddToken(token_id, begin=False)
        if test_scripting:
            transform = torch.jit.script(transform)

        actual = transform(inputs)
        expected = [["1", "2", "0"], ["1", "2", "3", "0"]]
        assert actual == expected

        inputs = ["1", "2"]
        actual = transform(inputs)
        expected = ["1", "2", "0"]
        assert actual == expected

    def test_add_token(self) -> None:
        self._add_token(test_scripting=False)

    def test_add_token_jit(self) -> None:
        self._add_token(test_scripting=True)

    def _pad_transform(self, test_scripting):
        """
        Test padding transform on 1D and 2D tensors.
        When max_length < tensor length at dim -1, this should be a no-op.
        Otherwise the tensor should be padded to max_length in dim -1.
        """

        inputs_1d_tensor = torch.ones(5)
        inputs_2d_tensor = torch.ones((8, 5))
        pad_long = PadTransform(max_length=7, pad_value=0)
        if test_scripting:
            pad_long = torch.jit.script(pad_long)
        padded_1d_tensor_actual = pad_long(inputs_1d_tensor)
        padded_1d_tensor_expected = torch.cat([torch.ones(5), torch.zeros(2)])
        assert_expected(
            padded_1d_tensor_actual,
            padded_1d_tensor_expected,
        )

        padded_2d_tensor_actual = pad_long(inputs_2d_tensor)
        padded_2d_tensor_expected = torch.cat(
            [torch.ones(8, 5), torch.zeros(8, 2)], axis=-1
        )
        assert_expected(
            padded_2d_tensor_actual,
            padded_2d_tensor_expected,
        )

        pad_short = PadTransform(max_length=3, pad_value=0)
        if test_scripting:
            pad_short = torch.jit.script(pad_short)
        padded_1d_tensor_actual = pad_short(inputs_1d_tensor)
        padded_1d_tensor_expected = inputs_1d_tensor
        assert_expected(
            padded_1d_tensor_actual,
            padded_1d_tensor_expected,
        )

        padded_2d_tensor_actual = pad_short(inputs_2d_tensor)
        padded_2d_tensor_expected = inputs_2d_tensor
        assert_expected(
            padded_2d_tensor_actual,
            padded_2d_tensor_expected,
        )

    def test_pad_transform(self) -> None:
        self._pad_transform(test_scripting=False)

    def test_pad_transform_jit(self) -> None:
        self._pad_transform(test_scripting=True)


class TestFunctional:
    @pytest.mark.parametrize("test_scripting", [True, False])
    @pytest.mark.parametrize(
        "configs",
        [
            [[[1, 2], [1, 2, 3]], 0, [[1, 2, 0], [1, 2, 3]]],
            [[[1, 2], [1, 2, 3]], 1, [[1, 2, 1], [1, 2, 3]]],
            [[1, 2], 0, [1, 2]],
        ],
    )
    def test_to_tensor(self, test_scripting, configs):
        """test tensorization on both single sequence and batch of sequence"""
        inputss, padding_value, expected_list = configs
        func = to_tensor
        if test_scripting:
            func = torch.jit.script(func)

        actual = func(inputss, padding_value=padding_value)
        expected = torch.tensor(expected_list, dtype=torch.long)
        assert_expected(actual, expected)

    def test_to_tensor_assert_raises(self) -> None:
        """test raise type error if inputs provided is not in Union[List[int],List[List[int]]]"""
        with pytest.raises(TypeError):
            to_tensor("test")

    @pytest.mark.parametrize("test_scripting", [True, False])
    @pytest.mark.parametrize(
        "configs",
        [
            [[[1, 2], [1, 2, 3]], [[1, 2], [1, 2]]],
            [[1, 2, 3], [1, 2]],
            [[["a", "b"], ["a", "b", "c"]], [["a", "b"], ["a", "b"]]],
            [["a", "b", "c"], ["a", "b"]],
        ],
    )
    def test_truncate(self, test_scripting, configs):
        """test truncation to max_seq_len length on both sequence and batch of sequence with both str/int types"""
        inputss, expected = configs
        max_seq_len = 2
        func = truncate
        if test_scripting:
            func = torch.jit.script(func)

        actual = func(inputss, max_seq_len=max_seq_len)
        assert actual == expected

    def test_truncate_assert_raises(self) -> None:
        """test raise type error if inputs provided is not in Union[List[Union[str, int]], List[List[Union[str, int]]]]"""
        with pytest.raises(TypeError):
            truncate("test", max_seq_len=2)

    @pytest.mark.parametrize("test_scripting", [True, False])
    @pytest.mark.parametrize(
        "configs",
        [
            # case: List[List[int]]
            [[[1, 2], [1, 2, 3]], 0, [[0, 1, 2], [0, 1, 2, 3]], True],
            [[[1, 2], [1, 2, 3]], 0, [[1, 2, 0], [1, 2, 3, 0]], False],
            # case: List[int]
            [[1, 2], 0, [0, 1, 2], True],
            [[1, 2], 0, [1, 2, 0], False],
            # case: List[List[str]]
            [[["a", "b"], ["c", "d"]], "x", [["x", "a", "b"], ["x", "c", "d"]], True],
            [[["a", "b"], ["c", "d"]], "x", [["a", "b", "x"], ["c", "d", "x"]], False],
            # case: List[str]
            [["a", "b"], "x", ["x", "a", "b"], True],
            [["a", "b"], "x", ["a", "b", "x"], False],
        ],
    )
    def test_add_token(self, test_scripting, configs):
        inputss, token_id, expected, begin = configs
        func = add_token
        if test_scripting:
            func = torch.jit.script(func)

        actual = func(inputss, token_id=token_id, begin=begin)
        assert actual == expected
