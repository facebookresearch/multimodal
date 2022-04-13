# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from itertools import chain
from typing import List

import torch
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
from test.test_utils import (
    gpu_test,
    init_distributed_on_file,
    with_temp_files,
)
from torch import distributed as dist
from torchmultimodal.modules.losses.contrastive_loss_with_temperature import (
    ContrastiveLossWithTemperature,
)
from torchmultimodal.utils.common import get_current_device


class TestContrastiveLossWithTemperature(unittest.TestCase):
    """
    Test the contrastive loss with temperature param
    """

    def setUp(self):
        torch.manual_seed(0)
        torch.cuda.manual_seed(0)
        torch.backends.cudnn.deterministic = True
        self.num_iterations = 1
        self.global_batch_size = 4
        self.embedding_dim = 3
        self.text_dim = 5
        self.image_dim = 8
        self.all_images = torch.randn(size=(self.global_batch_size, self.image_dim))
        self.all_texts = torch.randn(size=(self.global_batch_size, self.text_dim))
        # Create a simple model
        self.image_encoder = nn.Linear(self.image_dim, self.embedding_dim)
        self.text_encoder = nn.Linear(self.text_dim, self.embedding_dim)

    def test_local_loss(self):
        torch.manual_seed(1234)
        clip_loss = ContrastiveLossWithTemperature()
        clip_loss = clip_loss.to(get_current_device())
        image_embeddings = torch.randn(3, 5)
        text_embeddings = torch.randn(3, 5)
        loss = clip_loss(
            image_embeddings=image_embeddings, text_embeddings=text_embeddings
        )

        self.assertEqual(loss.size(), torch.Size([]))
        self.assertAlmostEqual(loss.item(), 9.8753, 3)

    @staticmethod
    def _model_worker(
        gpu_id: int,
        self,
        sync_file: str,
        world_size: int,
    ):
        init_distributed_on_file(
            world_size=world_size, gpu_id=gpu_id, sync_file=sync_file
        )
        assert self.global_batch_size % world_size == 0
        local_batch_size = self.global_batch_size // world_size

        # Split images and text across GPUs
        local_images = torch.split(self.all_images, local_batch_size)[gpu_id].cuda(
            gpu_id
        )
        local_texts = torch.split(self.all_texts, local_batch_size)[gpu_id].cuda(gpu_id)

        image_encoder = self.image_encoder.cuda(gpu_id)
        text_encoder = self.text_encoder.cuda(gpu_id)
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
            local_image_embeddings, local_text_embeddings, backprop_in_gather=True
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
        self.assertAlmostEqual(gathered_loss.item(), 3.8848, 3)

        # Gradients for image encoder weights
        img_encoder_weight_grad = gather_grads(image_encoder.weight.grad)
        self.assertAlmostEqual(img_encoder_weight_grad.mean().item(), 0.0979, 2)

        # Gradients for text encoder bias
        text_encoder_bias_grad = gather_grads(text_encoder.bias.grad)
        self.assertAlmostEqual(text_encoder_bias_grad.mean().item(), -1.8151, 2)

        # Logit scale gradient
        logit_scale_grad = gather_grads(loss_fn.logit_scale.grad)
        self.assertAlmostEqual(logit_scale_grad.mean().item(), 3.6781, 2)

    @gpu_test(gpu_count=1)
    def test_single_gpu_loss(self):
        with with_temp_files(count=1) as sync_file:
            world_size = 1
            mp.spawn(
                self._model_worker,
                (self, sync_file, world_size),
                nprocs=world_size,
            )

    @gpu_test(gpu_count=2)
    def test_multi_gpu_loss(self):
        with with_temp_files(count=1) as sync_file:
            world_size = 2
            mp.spawn(
                self._model_worker,
                (self, sync_file, world_size),
                nprocs=world_size,
            )
