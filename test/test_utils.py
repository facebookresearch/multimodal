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

import torch
import torch.distributed as dist
from torch import Tensor


def gpu_test(gpu_count: int = 1):
    """
    Annotation for GPU tests, skipping the test if the
    required amount of GPU is not available
    """
    import unittest

    message = f"Not enough GPUs to run the test: required {gpu_count}"
    return unittest.skipIf(torch.cuda.device_count() < gpu_count, message)


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
):
    torch.testing.assert_close(
        actual,
        expected,
        rtol=rtol,
        atol=atol,
        msg=f"actual: {actual}, expected: {expected}",
    )


def tuple_to_dict(t: Tuple) -> Dict:
    if not isinstance(t, tuple):
        raise TypeError(f"Input must be of type tuple but got {type(t)}")

    return {k: v for k, v in enumerate(t)}


def is_named_tuple(nt: Any) -> bool:
    # namedtuple is a subclass of tuple with additional attributes
    # we verify specifically here the attribute `_fields` which should be a tuple of field names
    # from the namedtuple instance
    if not isinstance(nt, tuple):
        return False
    f = getattr(nt, "_fields", None)
    if not isinstance(f, tuple):
        return False
    return all(type(name) == str for name in f)


def named_tuple_to_dict(nt: NamedTuple) -> Dict:
    # Do this for safety.  _asdict is a public method as of python 3.8:
    # https://docs.python.org/3/library/collections.html#collections.somenamedtuple._asdict
    if not hasattr(nt, "_asdict"):
        raise AttributeError(f"{actual} must have the attribute `_asdict`.")

    return nt._asdict()


def assert_expected_wrapper(
    actual: Union[Dict, NamedTuple],
    expected: Union[Dict, NamedTuple],
    rtol: Optional[float] = None,
    atol: Optional[float] = None,
):
    """Helper function that calls assert_expected recursively on nested Dict/NamedTuple"""
    # convert NamedTuple to dictionary
    if is_named_tuple(actual):
        actual = named_tuple_to_dict(actual)

    if is_named_tuple(expected):
        expected = named_tuple_to_dict(expected)

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

        if _actual is None:
            # optional output
            assert _expected is None
        elif isinstance(_actual, Dict):
            # dictionary output, e.g., cache of k/v
            assert_expected_wrapper(_actual, _expected, rtol=rtol, atol=atol)
        elif isinstance(_actual, tuple) and (not is_named_tuple(_actual)):
            # outputs are from multiple layers: (Tensor, Tensor, ...)
            assert_expected_wrapper(
                tuple_to_dict(_actual), tuple_to_dict(_expected), rtol=rtol, atol=atol
            )
        elif is_named_tuple(_actual):
            # output is another named tuple instance
            assert_expected_wrapper(_actual, _expected, rtol=rtol, atol=atol)
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
                    f"Unsupported type for expected when actual is a tensor: {_expected}"
                )
        else:
            raise TypeError(
                f"Unsupported types for test assertion: {_actual}, {_expected}"
            )
