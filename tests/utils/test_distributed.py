# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch
import torch.multiprocessing as mp
from tests.test_utils import (
    assert_expected,
    gpu_test,
    init_distributed_on_file,
    set_rng_seed,
    with_temp_files,
)
from torch import Tensor
from torchmultimodal.utils.distributed import gather_tensor

BATCH_SIZE = 4
EMBEDDING_DIM = 8


class TestGatherTensor:
    """
    Test gather_tensor method with backprop_in_gather param
    """

    @pytest.fixture(autouse=True)
    def setup(self):
        set_rng_seed(1234)

    @pytest.fixture
    def input_tensor(self):
        return torch.randn(BATCH_SIZE, EMBEDDING_DIM)

    @staticmethod
    def _worker(
        gpu_id: int,
        world_size: int,
        sync_file: str,
        input_tensor: Tensor,
        backprop_in_gather: bool,
    ):
        init_distributed_on_file(
            world_size=world_size, gpu_id=gpu_id, sync_file=sync_file
        )
        gpu_tensor = input_tensor.clone().requires_grad_().to(gpu_id) + gpu_id
        expected_output = [input_tensor + i for i in range(world_size)]
        gathered_output = gather_tensor(gpu_tensor, backprop_in_gather)
        assert_expected(len(expected_output), len(gathered_output))
        for i, tensor in enumerate(gathered_output):
            assert_expected(tensor, expected_output[i], check_device=False)
            if i == gpu_id or backprop_in_gather:
                assert tensor.grad_fn is not None
            else:
                assert tensor.grad_fn is None

    @gpu_test(gpu_count=1)
    @pytest.mark.parametrize("backprop_in_gather", [True, False])
    def test_single_gpu_gather(self, input_tensor: Tensor, backprop_in_gather: bool):
        world_size = 1
        with with_temp_files(count=1) as sync_file:
            mp.spawn(
                TestGatherTensor._worker,
                (
                    world_size,
                    sync_file,
                    input_tensor,
                    backprop_in_gather,
                ),
                nprocs=world_size,
            )

    @gpu_test(gpu_count=2)
    @pytest.mark.parametrize("backprop_in_gather", [True, False])
    def test_multi_gpu_gather(self, input_tensor: Tensor, backprop_in_gather: bool):
        world_size = 2
        with with_temp_files(count=1) as sync_file:
            mp.spawn(
                TestGatherTensor._worker,
                (
                    world_size,
                    sync_file,
                    input_tensor,
                    backprop_in_gather,
                ),
                nprocs=world_size,
            )
