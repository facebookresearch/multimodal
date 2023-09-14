# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from itertools import chain
from typing import List

import pytest

import torch
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim

from tests.test_utils import (
    assert_expected,
    gpu_test,
    init_distributed_on_file,
    set_rng_seed,
    with_temp_files,
)
from torch import distributed as dist, Tensor
from torchmultimodal.modules.losses.contrastive_loss_with_temperature import (
    ContrastiveLossWithTemperature,
)
from torchmultimodal.utils.common import get_current_device
from torchmultimodal.utils.distributed import BackpropType


class TestContrastiveLossWithTemperature:
    """
    Test the contrastive loss with temperature param
    """

    @pytest.fixture(autouse=True)
    def set_seed(self):
        set_rng_seed(0)
        torch.backends.cudnn.deterministic = True

    @pytest.fixture()
    def text_dim(self):
        return 5

    @pytest.fixture()
    def image_dim(self):
        return 8

    @pytest.fixture()
    def embedding_dim(self):
        return 3

    @pytest.fixture()
    def global_batch_size(self):
        return 4

    @pytest.fixture()
    def text_tensor(self, global_batch_size, text_dim):
        return torch.randn(global_batch_size, text_dim)

    @pytest.fixture()
    def image_tensor(self, global_batch_size, image_dim):
        return torch.randn(global_batch_size, image_dim)

    @pytest.fixture()
    def text_encoder(self, text_dim, embedding_dim):
        return nn.Linear(text_dim, embedding_dim)

    @pytest.fixture()
    def image_encoder(self, image_dim, embedding_dim):
        return nn.Linear(image_dim, embedding_dim)

    def test_local_loss(self):
        torch.manual_seed(1234)
        clip_loss = ContrastiveLossWithTemperature()
        clip_loss = clip_loss.to(get_current_device())
        embeddings_a = torch.randn(3, 5)
        embeddings_b = torch.randn(3, 5)
        loss = clip_loss(embeddings_a=embeddings_a, embeddings_b=embeddings_b)

        assert_expected(loss.item(), 9.8753, rtol=0, atol=1e-3)

    def test_temperature_clamp_max(self):
        torch.manual_seed(1234)
        clip_loss_at_max = ContrastiveLossWithTemperature(
            logit_scale=2, logit_scale_max=2
        ).to(get_current_device())
        clip_loss_above_max = ContrastiveLossWithTemperature(
            logit_scale=3, logit_scale_max=2
        ).to(get_current_device())
        embeddings_a = torch.randn(3, 5)
        embeddings_b = torch.randn(3, 5)
        loss_at_max = clip_loss_at_max(embeddings_a, embeddings_b).item()
        loss_above_max = clip_loss_above_max(embeddings_a, embeddings_b).item()
        assert_expected(loss_above_max, loss_at_max, rtol=0, atol=1e-3)

    def test_temperature_clamp_min(self):
        torch.manual_seed(1234)
        clip_loss_at_min = ContrastiveLossWithTemperature(
            logit_scale=2, logit_scale_min=2
        ).to(get_current_device())
        clip_loss_below_min = ContrastiveLossWithTemperature(
            logit_scale=1, logit_scale_min=2
        ).to(get_current_device())
        embeddings_a = torch.randn(3, 5)
        embeddings_b = torch.randn(3, 5)
        loss_at_min = clip_loss_at_min(embeddings_a, embeddings_b).item()
        loss_below_min = clip_loss_below_min(embeddings_a, embeddings_b).item()
        assert_expected(loss_below_min, loss_at_min, rtol=0, atol=1e-3)

    def test_loss_with_ce_kwargs(self):
        torch.manual_seed(1234)
        clip_loss = ContrastiveLossWithTemperature()
        clip_loss = clip_loss.to(get_current_device())
        embeddings_a = torch.randn(3, 5)
        embeddings_b = torch.randn(3, 5)
        loss = clip_loss(
            embeddings_a=embeddings_a,
            embeddings_b=embeddings_b,
            cross_entropy_kwargs={"label_smoothing": 0.1},
        )

        assert_expected(loss.item(), 10.2524, rtol=0, atol=1e-3)

    def test_temperature_clamp_invalid(self):
        with pytest.raises(ValueError):
            ContrastiveLossWithTemperature(logit_scale_max=None, logit_scale_min=None)

    @staticmethod
    def _model_worker(
        gpu_id: int,
        sync_file: str,
        world_size: int,
        global_batch_size: int,
        all_images: Tensor,
        all_texts: Tensor,
        image_encoder: nn.Module,
        text_encoder: nn.Module,
    ):
        init_distributed_on_file(
            world_size=world_size, gpu_id=gpu_id, sync_file=sync_file
        )
        assert global_batch_size % world_size == 0
        local_batch_size = global_batch_size // world_size

        # Split images and text across GPUs
        local_images = torch.split(all_images, local_batch_size)[gpu_id].cuda(gpu_id)
        local_texts = torch.split(all_texts, local_batch_size)[gpu_id].cuda(gpu_id)

        image_encoder = image_encoder.cuda(gpu_id)
        text_encoder = text_encoder.cuda(gpu_id)
        loss_fn = ContrastiveLossWithTemperature()
        loss_fn = loss_fn.cuda(gpu_id)

        all_params = chain(
            image_encoder.parameters(), text_encoder.parameters(), loss_fn.parameters()
        )

        optimizer = optim.SGD(all_params, lr=1e-4)

        # Forward pass
        local_image_embeddings = image_encoder(local_images)
        local_text_embeddings = text_encoder(local_texts)
        loss = loss_fn(
            local_image_embeddings,
            local_text_embeddings,
            backprop_type=BackpropType.GLOBAL,
        )

        # Compute gradients
        optimizer.zero_grad()
        loss.backward()

        # Gather gradients from all devices
        def gather_grads(x: torch.Tensor) -> List[torch.Tensor]:
            grads = [torch.zeros_like(x).cuda(gpu_id) for i in range(world_size)]
            dist.all_gather(grads, x)
            grad = torch.stack(grads).mean()
            return grad

        # Gather losses from all devices
        gathered_loss = gather_grads(torch.Tensor([loss]).cuda(gpu_id))
        assert_expected(gathered_loss.item(), 3.8848, rtol=0, atol=1e-3)

        # Gradients for image encoder weights
        img_encoder_weight_grad = gather_grads(image_encoder.weight.grad)
        assert_expected(
            img_encoder_weight_grad.mean().item(), 0.0979, rtol=0, atol=1e-3
        )

        # Gradients for text encoder bias
        text_encoder_bias_grad = gather_grads(text_encoder.bias.grad)
        assert_expected(
            text_encoder_bias_grad.mean().item(), -1.8151, rtol=0, atol=1e-3
        )

        # Logit scale gradient
        logit_scale_grad = gather_grads(loss_fn.logit_scale.grad)
        assert_expected(logit_scale_grad.mean().item(), 3.6792, rtol=0, atol=1e-3)

    @gpu_test(gpu_count=1)
    def test_single_gpu_loss(
        self, global_batch_size, image_tensor, text_tensor, image_encoder, text_encoder
    ):
        with with_temp_files(count=1) as sync_file:
            world_size = 1
            mp.spawn(
                TestContrastiveLossWithTemperature._model_worker,
                (
                    sync_file,
                    world_size,
                    global_batch_size,
                    image_tensor,
                    text_tensor,
                    image_encoder,
                    text_encoder,
                ),
                nprocs=world_size,
            )

    @gpu_test(gpu_count=2)
    def test_multi_gpu_loss(
        self, global_batch_size, image_tensor, text_tensor, image_encoder, text_encoder
    ):
        with with_temp_files(count=1) as sync_file:
            world_size = 2
            mp.spawn(
                TestContrastiveLossWithTemperature._model_worker,
                (
                    sync_file,
                    world_size,
                    global_batch_size,
                    image_tensor,
                    text_tensor,
                    image_encoder,
                    text_encoder,
                ),
                nprocs=world_size,
            )
