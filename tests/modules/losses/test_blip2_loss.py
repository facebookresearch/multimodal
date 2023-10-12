# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from itertools import chain

import pytest
import torch
from tests.test_utils import (
    assert_expected,
    gpu_test,
    init_distributed_on_file,
    init_weights_with_constant,
    with_temp_files,
)
from torch import distributed as dist, multiprocessing as mp, nn, optim
from torchmultimodal.models.blip2.blip2 import BLIP2, Blip2Output
from torchmultimodal.models.blip2.qformer_model import QformerForCLM
from torchmultimodal.modules.encoders.vision_transformer import VisionTransformer
from torchmultimodal.modules.layers.patch_embedding import PatchEmbeddings
from torchmultimodal.modules.layers.transformer import TransformerEncoder
from torchmultimodal.modules.losses.blip2_losses import Blip2Phase1Loss


@pytest.fixture
def dim_q():
    return 4


@pytest.fixture
def dim_kv():
    return 2


@pytest.fixture
def dim_feedforward():
    return 6


@pytest.fixture
def num_hidden_layers():
    return 2


@pytest.fixture
def num_heads():
    return 2


@pytest.fixture
def vocab_size():
    return 20


@pytest.fixture
def vit():
    embedding = PatchEmbeddings(image_size=2, patch_size=1, hidden_size=2)
    encoder = TransformerEncoder(
        n_layer=1,
        d_model=2,
        n_head=1,
        dim_feedforward=1,
        activation=nn.GELU,
        norm_first=True,
        final_layer_norm_eps=1e-5,
    )
    image_encoder = VisionTransformer(
        embeddings=embedding,
        encoder=encoder,
    )
    init_weights_with_constant(image_encoder)
    image_encoder.eval()
    return image_encoder


class TestBLIP2Stage1Loss:
    @pytest.fixture
    def images(self):
        return torch.ones(4, 3, 2, 2)

    @pytest.fixture
    def input_ids(self):
        return torch.ones(4, 4).long()

    @pytest.fixture
    def all_attn_mask(self):
        return torch.ones([4, 4])

    @pytest.fixture
    def global_batch_size(self):
        return 4

    @pytest.fixture
    def qformer_model_for_clm(
        self,
        dim_q,
        dim_kv,
        dim_feedforward,
        num_hidden_layers,
        num_heads,
        vocab_size,
    ):
        qformer_for_clm = QformerForCLM(
            dim_q=dim_q,
            dim_kv=dim_kv,
            dim_feedforward=dim_feedforward,
            num_heads=num_heads,
            attn_dropout=0.0,
            dropout=0.0,
            num_hidden_layers=num_hidden_layers,
            max_position_embeddings=512,
            vocab_size=vocab_size,
        )
        return qformer_for_clm

    @pytest.fixture
    def blip2_output(self):
        return Blip2Output(
            image_embeddings=torch.ones([4, 5, 2]),
            image_features=torch.ones([4, 32, 4]) * 0.5,
            image_qformer_output=torch.ones([4, 32, 4]) * 0.5,
            text_features=torch.ones([4, 4]) * 0.5,
            prediction_scores=torch.ones([4, 4, 20]) * 5,
        )

    @pytest.fixture
    def blip2(self, dim_q, dim_kv, qformer_model_for_clm, vit):
        blip2 = BLIP2(
            dim_q=dim_q,
            image_encoder_embedding_dim=dim_kv,
            qformer=qformer_model_for_clm,
            vision_encoder=vit,
            embedding_dim=4,
            decoder_bos_token_id=19,
        )
        init_weights_with_constant(blip2)
        blip2.eval()
        return blip2

    def test_local_loss(self, all_attn_mask, blip2_output, blip2, dim_q, input_ids):
        blip2_loss = Blip2Phase1Loss(dim_q=dim_q)
        init_weights_with_constant(blip2_loss)
        local_loss = blip2_loss(
            model_output=blip2_output,
            blip2=blip2,
            input_ids=input_ids,
            attention_mask=all_attn_mask,
        )
        assert_expected(local_loss.total_loss.item(), 5.07517, rtol=0, atol=1e-4)

    def test_local_itc_only_loss(
        self, all_attn_mask, blip2_output, blip2, dim_q, input_ids
    ):
        blip2_loss = Blip2Phase1Loss(dim_q=dim_q, enable_itm=False, enable_itg=False)
        init_weights_with_constant(blip2_loss)
        local_loss = blip2_loss(
            model_output=blip2_output,
            blip2=blip2,
            input_ids=input_ids,
            attention_mask=all_attn_mask,
        )
        assert_expected(local_loss.total_loss.item(), 1.38629, rtol=0, atol=1e-4)

    def test_local_itm_only_loss(
        self, all_attn_mask, blip2_output, blip2, dim_q, input_ids
    ):
        blip2_loss = Blip2Phase1Loss(dim_q=dim_q, enable_itc=False, enable_itg=False)
        init_weights_with_constant(blip2_loss)
        local_loss = blip2_loss(
            model_output=blip2_output,
            blip2=blip2,
            input_ids=input_ids,
            attention_mask=all_attn_mask,
        )
        assert_expected(local_loss.total_loss.item(), 0.69315, rtol=0, atol=1e-4)

    def test_local_itg_only_loss(
        self, all_attn_mask, blip2_output, blip2, dim_q, input_ids
    ):
        blip2_loss = Blip2Phase1Loss(dim_q=dim_q, enable_itc=False, enable_itm=False)
        init_weights_with_constant(blip2_loss)
        local_loss = blip2_loss(
            model_output=blip2_output,
            blip2=blip2,
            input_ids=input_ids,
            attention_mask=all_attn_mask,
        )
        assert_expected(local_loss.total_loss.item(), 2.9957, rtol=0, atol=1e-4)

    def test_invalid_loss_input(self):
        with pytest.raises(ValueError):
            Blip2Phase1Loss(
                dim_q=dim_q, enable_itc=False, enable_itm=False, enable_itg=False
            )

    @staticmethod
    def _model_worker(
        gpu_id: int,
        sync_file: str,
        world_size: int,
        global_batch_size: int,
        all_images: torch.Tensor,
        all_input_ids: torch.Tensor,
        all_attn_mask: torch.Tensor,
        blip2_output: Blip2Output,
        blip2: nn.Module,
        dim_q=dim_q,
    ):
        init_distributed_on_file(
            world_size=world_size, gpu_id=gpu_id, sync_file=sync_file
        )
        assert global_batch_size % world_size == 0
        local_batch_size = global_batch_size // world_size
        all_attn_mask = torch.ones([4, 4])

        # Split inputs across GPUs
        local_images = torch.split(all_images, local_batch_size)[gpu_id].cuda(gpu_id)
        local_input_ids = torch.split(all_input_ids, local_batch_size)[gpu_id].cuda(
            gpu_id
        )
        local_attn_mask = torch.split(all_attn_mask, local_batch_size)[gpu_id].cuda(
            gpu_id
        )
        assert blip2_output.text_features is not None
        assert blip2_output.prediction_scores is not None
        local_blip2_output = Blip2Output(
            image_embeddings=torch.split(
                blip2_output.image_embeddings, local_batch_size
            )[gpu_id].cuda(gpu_id),
            image_features=torch.split(blip2_output.image_features, local_batch_size)[
                gpu_id
            ].cuda(gpu_id),
            image_qformer_output=torch.split(
                blip2_output.image_qformer_output, local_batch_size
            )[gpu_id].cuda(gpu_id),
            text_features=torch.split(blip2_output.text_features, local_batch_size)[
                gpu_id
            ].cuda(gpu_id),
            prediction_scores=torch.split(
                blip2_output.prediction_scores, local_batch_size
            )[gpu_id].cuda(gpu_id),
        )

        blip2 = blip2.cuda(gpu_id)
        loss_fn = Blip2Phase1Loss(dim_q=dim_q)
        init_weights_with_constant(loss_fn)
        loss_fn = loss_fn.cuda(gpu_id)

        all_params = chain(blip2.parameters(), loss_fn.parameters())

        optimizer = optim.SGD(all_params, lr=1e-4)

        # Forward pass
        loss = loss_fn(
            model_output=local_blip2_output,
            blip2=blip2,
            images=local_images,
            input_ids=local_input_ids,
            attention_mask=local_attn_mask,
        ).total_loss

        # Compute gradients
        optimizer.zero_grad()
        loss.backward()

        # Gather gradients from all devices
        def gather_grads(x: torch.Tensor) -> torch.Tensor:
            grads = [torch.zeros_like(x).cuda(gpu_id) for i in range(world_size)]
            dist.all_gather(grads, x)
            grad = torch.stack(grads).mean()
            return grad

        # Gather losses from all devices
        gathered_loss = gather_grads(torch.Tensor([loss]).cuda(gpu_id))
        assert_expected(gathered_loss.item(), 5.07517, rtol=0, atol=1e-4)

    @gpu_test(gpu_count=1)
    def test_single_gpu_loss(
        self,
        global_batch_size,
        input_ids,
        blip2_output,
        blip2,
        attn_mask,
        dim_q,
    ):
        with with_temp_files(count=1) as sync_file:
            world_size = 1
            mp.spawn(
                TestBLIP2Stage1Loss._model_worker,
                (
                    sync_file,
                    world_size,
                    global_batch_size,
                    input_ids,
                    attn_mask,
                    blip2_output,
                    blip2,
                    dim_q,
                ),
                nprocs=world_size,
            )

    @gpu_test(gpu_count=2)
    def test_multi_gpu_loss(
        self,
        global_batch_size,
        input_ids,
        blip2_output,
        blip2,
        attn_mask,
        dim_q,
    ):
        with with_temp_files(count=1) as sync_file:
            world_size = 2
            mp.spawn(
                TestBLIP2Stage1Loss._model_worker,
                (
                    sync_file,
                    world_size,
                    global_batch_size,
                    input_ids,
                    attn_mask,
                    blip2_output,
                    blip2,
                    dim_q,
                ),
                nprocs=world_size,
            )
