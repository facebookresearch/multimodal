# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
import random
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, NamedTuple, Optional, Tuple, Union

import pytest
import torch
import torch.distributed as dist
from torch import nn, Tensor


def gpu_test(gpu_count: int = 1):
    """
    Annotation for GPU tests, skipping the test if the
    required amount of GPU is not available
    """
    message = f"Not enough GPUs to run the test: required {gpu_count}"
    return pytest.mark.skipif(torch.cuda.device_count() < gpu_count, reason=message)


def init_distributed_on_file(world_size: int, gpu_id: int, sync_file: str):
    """
    Init the process group need to do distributed training, by syncing
    the different workers on a file.
    """
    torch.cuda.set_device(gpu_id)
    dist.init_process_group(
        backend="nccl",
        init_method="file://" + sync_file,
        world_size=world_size,
        rank=gpu_id,
    )


@contextmanager
def with_temp_files(count: int):
    """
    Context manager to create temporary files and remove them
    after at the end of the context
    """
    if count == 1:
        fd, file_name = tempfile.mkstemp()
        yield file_name
        os.close(fd)
    else:
        temp_files = [tempfile.mkstemp() for _ in range(count)]
        yield [t[1] for t in temp_files]
        for t in temp_files:
            os.close(t[0])


def set_rng_seed(seed):
    """Sets the seed for pytorch and numpy random number generators"""
    torch.manual_seed(seed)
    random.seed(seed)


_ASSET_DIR = (Path(__file__).parent / "assets").resolve()


def get_asset_path(file_name: str) -> str:
    """Get the path to the file under assets directory."""
    return str(_ASSET_DIR.joinpath(file_name))


def assert_expected(
    actual: Any,
    expected: Any,
    rtol: Optional[float] = None,
    atol: Optional[float] = None,
    check_device=True,
):
    torch.testing.assert_close(
        actual,
        expected,
        rtol=rtol,
        atol=atol,
        check_device=check_device,
        msg=f"actual: {actual}, expected: {expected}",
    )


def tuple_to_dict(t: Tuple) -> Dict:
    if not isinstance(t, tuple):
        raise TypeError(f"Input must be of type tuple but got {type(t)}")

    return {k: v for k, v in enumerate(t)}


def is_namedtuple(nt: Any) -> bool:
    # namedtuple is a subclass of tuple with additional attributes
    # we verify specifically here the attribute `_fields` which should be a tuple of field names
    # from the namedtuple instance
    if not isinstance(nt, tuple):
        return False
    f = getattr(nt, "_fields", None)
    if not isinstance(f, tuple):
        return False
    return all(type(name) == str for name in f)


def namedtuple_to_dict(nt: NamedTuple) -> Dict:
    # Do this for safety.  _asdict is a public method as of python 3.8:
    # https://docs.python.org/3/library/collections.html#collections.somenamedtuple._asdict
    if not hasattr(nt, "_asdict"):
        raise AttributeError(f"{nt} must have the attribute `_asdict`.")

    return nt._asdict()


def assert_expected_namedtuple(
    actual: Union[Dict, NamedTuple],
    expected: Union[Dict, NamedTuple],
    rtol: Optional[float] = None,
    atol: Optional[float] = None,
):
    """Helper function that calls assert_expected recursively on nested Dict/NamedTuple

    Example::
        >>>from collections import namedtuple
        >>>Out = namedtuple("Out", "x y")
        >>>InnerOut = namedtuple("InnerOut", "z")
        >>>actual = Out(x=InnerOut(z=tensor([1])), y=tensor([2]))
        >>>expected = Out(x=InnerOut(z=tensor([1])), y=tensor([2]))
        >>>assert_expected_namedtuple(actual, expected)
    """
    # convert NamedTuple to dictionary
    if is_namedtuple(actual):
        actual = namedtuple_to_dict(actual)

    if is_namedtuple(expected):
        expected = namedtuple_to_dict(expected)

    if not isinstance(actual, Dict):
        raise TypeError(
            f"'actual' needs to be either of type 'NamedTuple' or 'Dict' but got {type(actual)}"
        )

    if not isinstance(expected, Dict):
        raise TypeError(
            f"'expected' needs to be either of type 'NamedTuple' or 'Dict' but got {type(expected)}"
        )

    for attr, _expected in expected.items():
        _actual = actual[attr]

        if _expected is None:
            # optional output
            assert _actual is None
        elif isinstance(_actual, Dict):
            # dictionary output, e.g., cache of k/v
            assert_expected_namedtuple(_actual, _expected, rtol=rtol, atol=atol)
        elif isinstance(_actual, tuple) and (not is_namedtuple(_actual)):
            # outputs are from multiple layers: (Tensor, Tensor, ...)
            assert_expected_namedtuple(
                tuple_to_dict(_actual), tuple_to_dict(_expected), rtol=rtol, atol=atol
            )
        elif is_namedtuple(_actual):
            # output is another named tuple instance
            assert_expected_namedtuple(_actual, _expected, rtol=rtol, atol=atol)
        elif isinstance(_actual, Tensor):
            # single tensor output
            if isinstance(_expected, tuple) and len(_expected) == 2:
                # test shape and sum
                _expected_shape, _expected_sum = _expected
                assert_expected(_actual.shape, _expected_shape)
                assert_expected(
                    _actual.sum().item(), _expected_sum, rtol=rtol, atol=atol
                )
            elif isinstance(_expected, Tensor):
                # test value
                assert_expected(_actual, _expected, rtol=rtol, atol=atol)
            else:
                raise TypeError(
                    f"Unsupported type for expected when actual is a tensor: {type(_expected)}"
                )
        else:
            raise TypeError(
                f"Unsupported types for test assertion: actual {type(_actual)}, expected {type(_expected)}"
            )


def init_weights_with_constant(model: nn.Module, constant: float = 1.0) -> None:
    for p in model.parameters():
        nn.init.constant_(p, constant)


def tensor_hash(x: torch.tensor, scaling=0.05, buckets=1000) -> torch.tensor:
    """hashes a multi-dim tensor for unit test verification

    usage: hash a forward tensor to serve as correct answer
    compare unit test result hashes with hashed answer using allclose

    example:
    t1 = torch.randn(1,16,32)  # answer tensor
    t2 = t1.clone()
    t2[0][0][0] += .01  # clone but modified with a single .01 change

    t1_hash = tensor_hash(t1)
    t2_hash = tensor_hash(t2)

    torch.allclose(t1_hash, t2_hash) # << -- False... + .01 difference detected

    # t1_hash ...
     tensor([[**175**,  46, 151, 958,  38, 905, 187,  93,  76, 966, 950, 966, 965, 985,
            12, 133]])
    # t2_hash ...
      tensor([[**176**,  46, 151, 958,  38, 905, 187,  93,  76, 966, 950, 966, 965, 985,
            12, 133]])

    """

    quant_tensor = torch.quantize_per_tensor(x, scaling, 0, torch.qint32)
    hashed_tensor = quant_tensor.int_repr().sum(-1) % buckets

    return hashed_tensor


def split_tensor_for_distributed_test(
    x: Tensor,
    local_batch_size: int,
    device_id: int,
    dim: int = 0,
    move_to_device: bool = True,
) -> Tensor:
    """
    Utility for distributed testing. Splits a tensor into chunks along a given dim,
    takes the kth chunk, and optionally moves to the specified device.
    """
    x = torch.split(x, local_batch_size, dim)[device_id]
    if move_to_device:
        x = x.to(device=device_id)
    return x
