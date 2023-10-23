# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, List, Optional, Union

import torch
from torch import nn, Tensor
from torch.nn.utils.rnn import pad_sequence


class Truncate(nn.Module):
    r"""Truncate input sequence

    :param max_seq_len: The maximum allowable length for input sequence
    :type max_seq_len: int
    """

    def __init__(self, max_seq_len: int) -> None:
        super().__init__()
        self.max_seq_len = max_seq_len

    def forward(self, x: Any) -> Any:
        """
        :param input: Input sequence or batch of sequence to be truncated
        :type input: Union[List[Union[str, int]], List[List[Union[str, int]]]]
        :return: Truncated sequence
        :rtype: Union[List[Union[str, int]], List[List[Union[str, int]]]]
        """
        return truncate(x, self.max_seq_len)


class AddToken(nn.Module):
    """Add token to beginning or end of sequence

    :param token: The token to be added
    :type token: Union[int, str]
    :param begin: Whether to insert token at start or end or sequence, defaults to True
    :type begin: bool, optional
    """

    def __init__(self, token: Union[int, str], begin: bool = True) -> None:
        super().__init__()
        self.token = token
        self.begin = begin

    def forward(self, input: Any) -> Any:
        """
        :param input: Input sequence or batch
        :type input: Union[List[Union[str, int]], List[List[Union[str, int]]]]
        """

        return add_token(input, self.token, self.begin)


class PadTransform(nn.Module):
    """Pad tensor to a fixed length with given padding value.

    :param max_length: Maximum length to pad to
    :type max_length: int
    :param pad_value: Value to pad the tensor with
    :type pad_value: bool
    """

    def __init__(self, max_length: int, pad_value: int) -> None:
        super().__init__()
        self.max_length = max_length
        self.pad_value = float(pad_value)

    def forward(self, x: Tensor) -> Tensor:
        """
        :param x: The tensor to pad
        :type x: Tensor
        :return: Tensor padded up to max_length with pad_value
        :rtype: Tensor
        """
        max_encoded_length = x.size(-1)
        if max_encoded_length < self.max_length:
            pad_amount = self.max_length - max_encoded_length
            x = torch.nn.functional.pad(x, (0, pad_amount), value=self.pad_value)
        return x


class ToTensor(nn.Module):
    r"""Convert input to torch tensor

    :param padding_value: Pad value to make each input in the batch of length equal to the longest sequence in the batch.
    :type padding_value: Optional[int]
    :param dtype: :class:`torch.dtype` of output tensor
    :type dtype: :class:`torch.dtype`
    """

    def __init__(
        self, padding_value: Optional[int] = None, dtype: torch.dtype = torch.long
    ) -> None:
        super().__init__()
        self.padding_value = padding_value
        self.dtype = dtype

    def forward(self, input: Any) -> Tensor:
        """
        :param input: Sequence or batch of token ids
        :type input: Union[List[int], List[List[int]]]
        :rtype: Tensor
        """
        return to_tensor(input, padding_value=self.padding_value, dtype=self.dtype)


def to_tensor(
    input: Any, padding_value: Optional[int] = None, dtype: torch.dtype = torch.long
) -> Tensor:
    r"""Convert input to torch tensor

    :param padding_value: Pad value to make each input in the batch of length equal to the longest sequence in the batch.
    :type padding_value: Optional[int]
    :param dtype: :class:`torch.dtype` of output tensor
    :type dtype: :class:`torch.dtype`
    :param input: Sequence or batch of token ids
    :type input: Union[List[int], List[List[int]]]
    :rtype: Tensor
    """
    if torch.jit.isinstance(input, List[int]):
        return torch.tensor(input, dtype=torch.long)
    elif torch.jit.isinstance(input, List[List[int]]):
        if padding_value is None:
            output = torch.tensor(input, dtype=dtype)
            return output
        else:
            output = pad_sequence(
                [torch.tensor(ids, dtype=dtype) for ids in input],
                batch_first=True,
                padding_value=float(padding_value),
            )
            return output
    else:
        raise TypeError("Input type not supported")


def truncate(input: Any, max_seq_len: int) -> Any:
    """Truncate input sequence or batch

    :param input: Input sequence or batch to be truncated
    :type input: Union[List[Union[str, int]], List[List[Union[str, int]]]]
    :param max_seq_len: Maximum length beyond which input is discarded
    :type max_seq_len: int
    :return: Truncated sequence
    :rtype: Union[List[Union[str, int]], List[List[Union[str, int]]]]
    """
    if torch.jit.isinstance(input, List[int]):
        return input[:max_seq_len]
    elif torch.jit.isinstance(input, List[str]):
        return input[:max_seq_len]
    elif torch.jit.isinstance(input, List[List[int]]):
        output_int: List[List[int]] = []
        for ids in input:
            output_int.append(ids[:max_seq_len])
        return output_int
    elif torch.jit.isinstance(input, List[List[str]]):
        output_str: List[List[str]] = []
        for ids in input:
            output_str.append(ids[:max_seq_len])
        return output_str
    else:
        raise TypeError("Input type not supported")


def add_token(input: Any, token_id: Any, begin: bool = True) -> Any:
    """Add token to start or end of sequence

    :param input: Input sequence or batch
    :type input: Union[List[Union[str, int]], List[List[Union[str, int]]]]
    :param token_id: token to be added
    :type token_id: Union[str, int]
    :param begin: Whether to insert token at start or end or sequence, defaults to True
    :type begin: bool, optional
    :return: sequence or batch with token_id added to begin or end or input
    :rtype: Union[List[Union[str, int]], List[List[Union[str, int]]]]
    """
    if torch.jit.isinstance(input, List[int]) and torch.jit.isinstance(token_id, int):
        if begin:
            return [token_id] + input
        else:
            return input + [token_id]
    elif torch.jit.isinstance(input, List[str]) and torch.jit.isinstance(token_id, str):
        if begin:
            return [token_id] + input
        else:
            return input + [token_id]
    elif torch.jit.isinstance(input, List[List[int]]) and torch.jit.isinstance(
        token_id, int
    ):
        output_int: List[List[int]] = []

        if begin:
            for ids in input:
                output_int.append([token_id] + ids)
        else:
            for ids in input:
                output_int.append(ids + [token_id])

        return output_int
    elif torch.jit.isinstance(input, List[List[str]]) and torch.jit.isinstance(
        token_id, str
    ):
        output_str: List[List[str]] = []
        if begin:
            for ids in input:
                output_str.append([token_id] + ids)
        else:
            for ids in input:
                output_str.append(ids + [token_id])

        return output_str
    else:
        raise TypeError("Input type not supported")
