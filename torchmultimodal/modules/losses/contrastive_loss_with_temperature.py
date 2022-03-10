# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import math
from typing import Tuple

import torch
import torch.nn.functional as F
from torch import nn
from torch.distributed import all_gather as all_gather_no_backprop
from torch.distributed.nn.functional import all_gather as all_gather_with_backprop


class ContrastiveLossWithTemperature(nn.Module):
    """Contrastive loss with a temperature parameter, as used in CLIP and FLAVA.
    CLIP: https://arxiv.org/pdf/2103.00020.pdf
    FLAVA: https://arxiv.org/pdf/2112.04482.pdf


    A contrastive loss over pairs of image and text embeddings. For each image
    embedding, we compute a weighted cosine similarity with all text embeddings,
    then calculate the cross entropy loss against the true (image, text) pairing.
    Each text embedding is evaluated against all image embeddings similarly.
    The batch's loss is the average cross entropy over all image and text embeddings
    in the batch.

    Temperature is a learned parameter clamped to ``[1, 100]`` and
    initialized to 1 / 0.07 as in the CLIP paper.


    Args:
        logit_scale (float): Log of the learnable temperature parameter value

    Inputs: img_embeddings (Tensor): Tensor containing image features.
                (In the CLIP model, these are the outputs of the image encoder.)
            text_embeddings (Tensor): Tensor containing text features.
                (In the CLIP model, these are the outputs of the text encoder.)
            backprop_in_gather (bool): Whether to backpropagate the gradients from
                all_gather to all workers (versus just the local worker).
    """

    def __init__(self, logit_scale: float = None):
        super().__init__()
        if logit_scale is None:
            logit_scale = math.log(1 / 0.07)
        self.logit_scale = nn.Parameter(logit_scale * torch.ones([]))

    def _gather_embeddings_and_labels(
        self,
        img_embeddings: torch.Tensor,
        text_embeddings: torch.Tensor,
        backprop_in_gather: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        if not torch.distributed.is_initialized():
            labels = torch.arange(img_embeddings.size(0))
            return img_embeddings, text_embeddings, labels

        # img_embeddings has shape [local_batch_size, embedding_dim]
        local_batch_size = img_embeddings.size(0)

        world_size = torch.distributed.get_world_size()

        # Backpropagate gradients to all workers using the all_gather from
        # torch.distributed.nn.functional. This is based on FLAVA's global
        # contrastive loss: https://arxiv.org/pdf/2112.04482.pdf
        if backprop_in_gather:
            img_embeddings_all_gpus = all_gather_with_backprop(img_embeddings)
            text_embeddings_all_gpus = all_gather_with_backprop(text_embeddings)

        # Otherwise just backprop to the current worker
        # This means that the image gradients on a given worker will only
        # consider the text samples from the same worker
        else:
            text_embeddings_all_gpus = [
                torch.zeros_like(text_embeddings) for _ in range(world_size)
            ]
            img_embeddings_all_gpus = [
                torch.zeros_like(img_embeddings) for _ in range(world_size)
            ]
            all_gather_no_backprop(img_embeddings_all_gpus, img_embeddings)
            all_gather_no_backprop(text_embeddings_all_gpus, text_embeddings)

        labels = local_batch_size * torch.distributed.get_rank() + torch.arange(
            local_batch_size, device=self.logit_scale.device
        )

        return (
            torch.cat(img_embeddings_all_gpus),
            torch.cat(text_embeddings_all_gpus),
            labels,
        )

    def forward(
        self,
        img_embeddings: torch.Tensor,
        text_embeddings: torch.Tensor,
        backprop_in_gather: bool = True,
    ):

        # Note: we clamp to 4.6052 = ln(100), as in the original paper.
        self.logit_scale.data.clamp_(0, 4.6052)
        # this temperature implementation follows CLIP Figure 3
        temperature = torch.exp(self.logit_scale)

        (
            img_embeddings_all_gpus,
            text_embeddings_all_gpus,
            labels,
        ) = self._gather_embeddings_and_labels(
            img_embeddings, text_embeddings, backprop_in_gather
        )

        # logits_per_image has shape [local_batch_size, global_batch_size]
        logits_per_image = (
            torch.matmul(img_embeddings, text_embeddings_all_gpus.transpose(0, 1))
            * temperature
        )
        logits_per_text = (
            torch.matmul(text_embeddings, img_embeddings_all_gpus.transpose(0, 1))
            * temperature
        )

        loss_i = F.cross_entropy(logits_per_image, labels)
        loss_t = F.cross_entropy(logits_per_text, labels)
        loss = (loss_i + loss_t) / 2

        return loss
