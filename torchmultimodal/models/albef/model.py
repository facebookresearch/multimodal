# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import copy
from collections import namedtuple
from typing import Tuple

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torchmultimodal.utils.common import momentum_update, remove_grad


ALBEFOutput = namedtuple(
    "ALBEFOutput",
    [
        "image_embeddings",
        "image_embeddings_m",
        "text_embeddings",
        "text_embeddings_m",
        "multimodal_embeddings",
        "multimodal_embeddings_m",
    ],
    defaults=(None, None, None, None, None, None),
)

ALBEFSimilarity = namedtuple(
    "ALBEFSimilarity",
    [
        "sim_i2t",  # image to text similarity
        "sim_t2i",  # text to image similarity
        "sim_i2t_m",  # image to text similarity for momentum embeddings
        "sim_t2i_m",  # text to image similarity for momentum embeddings
    ],
    defaults=(None, None, None, None),
)

ALBEFWithSimilarityOutput = namedtuple(
    "ALBEFWithSimilarityOutput",
    [
        "image_embeddings",
        "text_embeddings",
        "multimodal_embeddings",
        "multimodal_embeddings_neg",
        "similarity",
        "sim_targets",
    ],
    defaults=(None, None, None, None, None, None),
)


class ALBEFModel(nn.Module):
    """
    ALBEF is a model to ALign the image and text representations BEfore Fusing
    (ALBEF) them through cross-modal attention, which enables more grounded vision
    and language representation learning. (https://arxiv.org/pdf/2107.07651.pdf)

    Args:   vision_encoder (nn.Module): Instantiated vision encoder.
            text_encoder (nn.Module): Instantiated text encoder.
            multimodal_encoder (nn.Module): Instantiated multimodal encoder.
            momentum (float): Momentum parameter. Default is 0.995.

    Inputs: image (Tensor): Tensor of shape (B, C, H, W) containing image features.
            text (Tensor): Tensor of shape (B, L) containing text features.
            text_atts (Tensor): Tensor of shape (B, L) containing text attention mask.
    """

    def __init__(
        self,
        vision_encoder: nn.Module,
        text_encoder: nn.Module,
        multimodal_encoder: nn.Module,
        momentum: float = 0.995,
    ) -> None:
        super().__init__()
        self.vision_encoder = vision_encoder
        self.text_encoder = text_encoder
        self.multimodal_encoder = multimodal_encoder
        self.vision_encoder_m = copy.deepcopy(vision_encoder)
        self.text_encoder_m = copy.deepcopy(text_encoder)
        self.multimodal_encoder_m = copy.deepcopy(multimodal_encoder)

        remove_grad(self.vision_encoder_m)
        remove_grad(self.text_encoder_m)
        remove_grad(self.multimodal_encoder_m)
        self.momentum = momentum

    def forward(
        self,
        image: Tensor,
        text: Tensor,
        text_atts: Tensor,
    ) -> ALBEFOutput:
        image_embeds = self.vision_encoder(image)
        text_embeds = self.text_encoder(text, text_atts)
        multimodal_embeddings = self.multimodal_encoder(
            hidden_states=text_embeds.last_hidden_state,
            attention_mask=text_atts,
            encoder_hidden_states=image_embeds,
        )

        with torch.no_grad():
            momentum_update(self.vision_encoder, self.vision_encoder_m, self.momentum)
            momentum_update(self.text_encoder, self.text_encoder_m, self.momentum)
            momentum_update(
                self.multimodal_encoder, self.multimodal_encoder_m, self.momentum
            )
            image_embeds_m = self.vision_encoder_m(image)
            text_embeds_m = self.text_encoder_m(text, text_atts)
            multimodal_embeddings_m = self.multimodal_encoder_m(
                hidden_states=text_embeds_m.last_hidden_state,
                attention_mask=text_atts,
                encoder_hidden_states=image_embeds_m,
            )

        return ALBEFOutput(
            image_embeddings=image_embeds,
            image_embeddings_m=image_embeds_m,
            text_embeddings=text_embeds.last_hidden_state,
            text_embeddings_m=text_embeds_m.last_hidden_state,
            multimodal_embeddings=multimodal_embeddings,
            multimodal_embeddings_m=multimodal_embeddings_m,
        )


class ALBEFModelWithSimilarity(nn.Module):
    """
    ALBEFModelWithSimilarity outputs image embeddings, text embeddings, multimodal embeddings,
    negative image-text pairs multimodal embeddings, and image-text similarity, as used in ITC
    and ITM losses.

    Args:   albef_model (ALBEFModel): Instantiated ALBEF model.
            vision_proj (nn.Module): Instantiated vision projection layer.
            text_proj (nn.Module): Instantiated text projection layer.
            embed_size (int): Embedding size of the vision and text projection layers. Default is 256.
            queue_size (int): Size of image and text queues for momentum distillation. Default is 65536.
            masked_token_id (int): The token id indicating a masked token. Default is -100.
            temp (float): Temperature parameter. Default is 0.07.

    Inputs: image (Tensor): Tensor of shape (B, C, H, W) containing image features.
            text (Tensor): Tensor of shape (B, L) containing text features.
            text_atts (Tensor): Tensor of shape (B, L) containing text attention mask.
            idx (Tensor): Tensor of shape (B) containing unique identifiers for each sample.
    """

    def __init__(
        self,
        albef_model: ALBEFModel,
        vision_proj: nn.Module,
        text_proj: nn.Module,
        embed_size: int = 256,
        queue_size: int = 65536,
        mask_token_id: int = -100,
        temp: float = 0.07,
    ) -> None:
        super().__init__()
        self.albef_model = albef_model
        self.vision_proj = vision_proj
        self.text_proj = text_proj
        self.vision_proj_m = copy.deepcopy(vision_proj)
        self.text_proj_m = copy.deepcopy(text_proj)

        remove_grad(self.vision_proj_m)
        remove_grad(self.text_proj_m)

        self.queue_size = queue_size
        self.temp = nn.Parameter(torch.ones([]) * temp)

        # queues keep track of the most recent M image and text representations for momentum distillation
        # queues decouple M from the batch size, allowing it to be big
        self.register_buffer("image_queue", torch.randn(embed_size, queue_size))
        self.register_buffer("text_queue", torch.randn(embed_size, queue_size))
        self.register_buffer(
            "idx_queue", torch.full((1, self.queue_size), mask_token_id)
        )
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

        self.image_queue: Tensor
        self.text_queue: Tensor
        self.idx_queue: Tensor
        self.queue_ptr: Tensor
        self.image_queue = nn.functional.normalize(self.image_queue, dim=0)
        self.text_queue = nn.functional.normalize(self.text_queue, dim=0)

    def forward(
        self,
        image: Tensor,
        text: Tensor,
        text_atts: Tensor,
        idx: Tensor,
    ) -> ALBEFWithSimilarityOutput:
        outputs = self.albef_model(image, text, text_atts)

        # reshape idx to (B, 1)
        idx = idx.view(-1, 1)
        # get identifiers for the most recent M samples
        idx_all = torch.cat([idx.t(), self.idx_queue.detach().clone()], dim=1)
        # check for seen identifiers in the most recent M samples
        pos_idx = torch.eq(idx, idx_all).float()
        sim_targets = pos_idx / pos_idx.sum(1, keepdim=True)
        similarity = self._similarity(
            outputs.image_embeddings,
            outputs.image_embeddings_m,
            outputs.text_embeddings,
            outputs.text_embeddings_m,
            idx,
        )
        image_embeds_neg, text_embeds_neg, text_atts_neg = self._neg_embeddings(
            outputs.image_embeddings, outputs.text_embeddings, text_atts, similarity
        )
        multimodal_embeddings_neg = self.albef_model.multimodal_encoder(
            torch.cat([outputs.text_embeddings, text_embeds_neg], dim=0),
            torch.cat([text_atts, text_atts_neg], dim=0),
            torch.cat([image_embeds_neg, outputs.image_embeddings], dim=0),
        )

        return ALBEFWithSimilarityOutput(
            image_embeddings=outputs.image_embeddings,
            text_embeddings=outputs.text_embeddings,
            multimodal_embeddings=outputs.multimodal_embeddings,
            multimodal_embeddings_neg=multimodal_embeddings_neg,
            similarity=similarity,
            sim_targets=sim_targets,
        )

    @torch.no_grad()
    def _dequeue_and_enqueue(
        self, image_feat_m: Tensor, text_feat_m: Tensor, idx: Tensor
    ) -> None:
        # gather keys before updating queue
        image_feats = _gather_embeddings(image_feat_m)
        text_feats = _gather_embeddings(text_feat_m)
        idxs = _gather_embeddings(idx)
        batch_size = image_feats.shape[0]
        ptr = int(self.queue_ptr)

        assert (
            self.queue_size % batch_size == 0
        ), "queue_size should be divisible by batch_size"

        # replace the keys at ptr (dequeue and enqueue)
        self.image_queue[:, ptr : ptr + batch_size] = image_feats.T
        self.text_queue[:, ptr : ptr + batch_size] = text_feats.T
        self.idx_queue[:, ptr : ptr + batch_size] = idxs.T
        ptr = (ptr + batch_size) % self.queue_size
        self.queue_ptr[0] = ptr

    def _similarity(
        self,
        image_embeds: Tensor,
        image_embeds_m: Tensor,
        text_embeds: Tensor,
        text_embeds_m: Tensor,
        idx: Tensor,
    ) -> ALBEFSimilarity:
        # transform the [CLS] embeddings to normalized lower-dimensional representations
        image_feat = F.normalize(self.vision_proj(image_embeds[:, 0, :]), dim=-1)
        text_feat = F.normalize(self.text_proj(text_embeds[:, 0, :]), dim=-1)

        with torch.no_grad():
            momentum_update(
                self.vision_proj, self.vision_proj_m, self.albef_model.momentum
            )
            momentum_update(self.text_proj, self.text_proj_m, self.albef_model.momentum)
            image_feat_m = F.normalize(
                self.vision_proj_m(image_embeds_m[:, 0, :]), dim=-1
            )
            text_feat_m = F.normalize(self.text_proj_m(text_embeds_m[:, 0, :]), dim=-1)
            image_feat_all = torch.cat(
                [image_feat_m.t(), self.image_queue.detach().clone()], dim=1
            )
            text_feat_all = torch.cat(
                [text_feat_m.t(), self.text_queue.detach().clone()], dim=1
            )
            sim_i2t_m = image_feat_m @ text_feat_all / self.temp
            sim_t2i_m = text_feat_m @ image_feat_all / self.temp

        sim_i2t = image_feat @ text_feat_all / self.temp
        sim_t2i = text_feat @ image_feat_all / self.temp

        self._dequeue_and_enqueue(image_feat_m, text_feat_m, idx)

        return ALBEFSimilarity(
            sim_i2t=sim_i2t,
            sim_t2i=sim_t2i,
            sim_i2t_m=sim_i2t_m,
            sim_t2i_m=sim_t2i_m,
        )

    def _neg_embeddings(
        self,
        image_embeds: Tensor,
        text_embeds: Tensor,
        text_atts: Tensor,
        similarity: ALBEFSimilarity,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        with torch.no_grad():
            bs = image_embeds.size(0)
            weights_i2t = F.softmax(similarity.sim_i2t[:, :bs], dim=1)
            weights_t2i = F.softmax(similarity.sim_t2i[:, :bs], dim=1)
            weights_i2t.fill_diagonal_(0)
            weights_t2i.fill_diagonal_(0)

        image_embeds_neg, text_embeds_neg, text_atts_neg = [], [], []
        for b in range(bs):
            neg_idx = int(torch.multinomial(weights_t2i[b], 1).item())
            image_embeds_neg.append(image_embeds[neg_idx])
        image_embeds_neg = torch.stack(image_embeds_neg, dim=0)

        for b in range(bs):
            neg_idx = int(torch.multinomial(weights_i2t[b], 1).item())
            text_embeds_neg.append(text_embeds[neg_idx])
            text_atts_neg.append(text_atts[neg_idx])
        text_embeds_neg = torch.stack(text_embeds_neg, dim=0)
        text_atts_neg = torch.stack(text_atts_neg, dim=0)
        return image_embeds_neg, text_embeds_neg, text_atts_neg


def _gather_embeddings(embeddings: torch.Tensor) -> torch.Tensor:
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        return embeddings

    embeddings_all_gpus = [
        torch.zeros_like(embeddings) for _ in range(torch.distributed.get_world_size())
    ]
    torch.distributed.all_gather(embeddings_all_gpus, embeddings)

    return torch.cat(embeddings_all_gpus)
