# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import copy
from collections import namedtuple

import torch
import torch.nn.functional as F
from torch import nn, Tensor

from torchmultimodal.modules.losses.contrastive_loss_with_temperature import (
    _gather_embeddings_and_labels,
)


ALBEFOutput = namedtuple(
    "ALBEFOutput",
    [
        "image_embeddings",
        "image_embeddings_m",
        "text_embeddings",
        "text_atts",
        "vl_embeddings",
        "similarity",
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


class ALBEFModel(nn.Module):
    """
    ALBEF is a model to ALign the image and text representations BEfore Fusing
    (ALBEF) them through cross-modal attention, which enables more grounded vision
    and language representation learning. (https://arxiv.org/pdf/2107.07651.pdf)

    Args:   vision_encoder (nn.Module): Instantiated vision encoder
            text_encoder (nn.Module): instantiated text encoder
            multimodal_encoder (nn.Module): Instantiated multimodal encoder
            vision_proj (nn.Module): Instantiated vision projection layer
            text_proj (nn.Module): Instantiated text projection layer
            embed_dim (int): embedding size of the vision and text projection layers
            queue_size (int): size of image and text queues for momentum distillation
            temp (float): temperature parameter
            momentum (float): momentum parameter

    Inputs: image (Tensor): Tensor containing image features
            text (Tensor): Tensor containing text features
            text_atts (Tensor): Tensor containing text attention mask
    """

    def __init__(
        self,
        vision_encoder: nn.Module,
        text_encoder: nn.Module,
        multimodal_encoder: nn.Module,
        vision_proj: nn.Module,
        text_proj: nn.Module,
        embed_dim: int = 256,
        queue_size: int = 65536,
        temp: float = 0.07,
        momentum: float = 0.995,
    ):
        super().__init__()
        self.vision_encoder = vision_encoder
        self.text_encoder = text_encoder
        self.multimodal_encoder = multimodal_encoder
        self.vision_proj = vision_proj
        self.text_proj = text_proj
        self.vision_encoder_m = copy.deepcopy(vision_encoder)
        self.text_encoder_m = copy.deepcopy(text_encoder)
        self.multimodal_encoder_m = copy.deepcopy(multimodal_encoder)
        self.vision_proj_m = copy.deepcopy(vision_proj)
        self.text_proj_m = copy.deepcopy(text_proj)

        self.models = [
            self.vision_encoder,
            self.text_encoder,
            self.multimodal_encoder,
            self.vision_proj,
            self.text_proj,
        ]
        self.models_m = [
            self.vision_encoder_m,
            self.text_encoder_m,
            self.multimodal_encoder_m,
            self.vision_proj_m,
            self.text_proj_m,
        ]

        self._copy_params_momentum_models()

        self.queue_size = queue_size
        self.temp = temp
        self.momentum = momentum

        # queues keep track of the most recent M image and text representations for momentum distillation
        # queues decouple M from the batch size, allowing it to be big
        self.register_buffer("image_queue", torch.randn(embed_dim, queue_size))
        self.register_buffer("text_queue", torch.randn(embed_dim, queue_size))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

        self.image_queue: Tensor
        self.text_queue: Tensor
        self.queue_ptr: Tensor
        self.image_queue = nn.functional.normalize(self.image_queue, dim=0)
        self.text_queue = nn.functional.normalize(self.text_queue, dim=0)

    def forward(
        self,
        image: Tensor,
        text: Tensor,
        text_atts: Tensor = None,
    ):
        image_embeds, text_embeds, image_feat, text_feat = self._unimodal_embeddings(
            image, text, text_atts
        )
        image_embeds_m, image_feat_m, text_feat_m = self._momentum_embeddings(
            image, text, text_atts
        )
        similarity = self._similarity(image_feat, text_feat, image_feat_m, text_feat_m)
        image_embeds_neg, text_embeds_neg, text_atts_neg = self._neg_embeddings(
            image_embeds, text_embeds, text_atts, similarity
        )
        vl_embeds = self._multimodal_embeddings(
            image_embeds,
            text_embeds,
            image_embeds_neg,
            text_embeds_neg,
            text_atts,
            text_atts_neg,
        )
        return ALBEFOutput(
            image_embeddings=image_embeds,
            image_embeddings_m=image_embeds_m,
            text_embeddings=text_embeds,
            text_atts=text_atts,
            vl_embeddings=vl_embeds,
            similarity=similarity,
        )

    @torch.no_grad()
    def _copy_params_momentum_models(self):
        for model, model_m in zip(self.models, self.models_m):
            for param, param_m in zip(model.parameters(), model_m.parameters()):
                param_m.data.copy_(param.data)
                param_m.requires_grad = False

    def _unimodal_embeddings(self, image, text, text_atts):
        image_embeds = self.vision_encoder(image)
        text_embeds = self.text_encoder(text, attention_mask=text_atts)
        image_feat = F.normalize(self.vision_proj(image_embeds[:, 0, :]), dim=-1)
        text_feat = F.normalize(self.text_proj(text_embeds[:, 0, :]), dim=-1)
        return image_embeds, text_embeds, image_feat, text_feat

    @torch.no_grad()
    def _momentum_embeddings(self, image, text, text_atts):
        self._momentum_update()
        image_embeds_m = self.vision_encoder_m(image)
        text_embeds_m = self.text_encoder_m(text, attention_mask=text_atts)
        image_feat_m = F.normalize(self.vision_proj_m(image_embeds_m[:, 0, :]), dim=-1)
        text_feat_m = F.normalize(self.text_proj_m(text_embeds_m[:, 0, :]), dim=-1)
        return image_embeds_m, image_feat_m, text_feat_m

    @torch.no_grad()
    def _momentum_update(self):
        for model, model_m in zip(self.models, self.models_m):
            for param, param_m in zip(model.parameters(), model_m.parameters()):
                param_m.data = param_m.data * self.momentum + param.data * (
                    1 - self.momentum
                )

    @torch.no_grad()
    def _update_queue(self, image_feat_m, text_feat_m):
        image_feats, text_feats, _ = _gather_embeddings_and_labels(
            image_feat_m, text_feat_m
        )
        batch_size = image_feats.shape[0]
        ptr = int(self.queue_ptr)

        assert (
            self.queue_size % batch_size == 0
        ), "queue_size should be divisible by batch_size"

        self.image_queue[:, ptr : ptr + batch_size] = image_feats.T
        self.text_queue[:, ptr : ptr + batch_size] = text_feats.T
        ptr = (ptr + batch_size) % self.queue_size
        self.queue_ptr[0] = ptr

    def _similarity(self, image_feat, text_feat, image_feat_m, text_feat_m):
        with torch.no_grad():
            image_feat_all = torch.cat(
                [image_feat_m.t(), self.image_queue.clone().detach()], dim=1
            )
            text_feat_all = torch.cat(
                [text_feat_m.t(), self.text_queue.clone().detach()], dim=1
            )
            sim_i2t_m = image_feat_m @ text_feat_all / self.temp
            sim_t2i_m = text_feat_m @ image_feat_all / self.temp

        sim_i2t = image_feat @ text_feat_all / self.temp
        sim_t2i = text_feat @ image_feat_all / self.temp
        self._update_queue(image_feat_m, text_feat_m)

        return ALBEFSimilarity(
            sim_i2t=sim_i2t,
            sim_t2i=sim_t2i,
            sim_i2t_m=sim_i2t_m,
            sim_t2i_m=sim_t2i_m,
        )

    def _neg_embeddings(self, image_embeds, text_embeds, text_atts, similarity):
        with torch.no_grad():
            bs = image_embeds.size(0)
            weights_i2t = F.softmax(similarity.sim_i2t[:, :bs], dim=1)
            weights_t2i = F.softmax(similarity.sim_t2i[:, :bs], dim=1)
            weights_i2t.fill_diagonal_(0)
            weights_t2i.fill_diagonal_(0)

        image_embeds_neg, text_embeds_neg, text_atts_neg = [], [], []
        for b in range(bs):
            neg_idx = torch.multinomial(weights_t2i[b], 1).item()
            image_embeds_neg.append(image_embeds[neg_idx])
        image_embeds_neg = torch.stack(image_embeds_neg, dim=0)

        for b in range(bs):
            neg_idx = torch.multinomial(weights_i2t[b], 1).item()
            text_embeds_neg.append(text_embeds[neg_idx])
            text_atts_neg.append(text_atts[neg_idx])
        text_embeds_neg = torch.stack(text_embeds_neg, dim=0)
        text_atts_neg = torch.stack(text_atts_neg, dim=0)
        return image_embeds_neg, text_embeds_neg, text_atts_neg

    def _multimodal_embeddings(
        self,
        image_embeds,
        text_embeds,
        image_embeds_neg,
        text_embeds_neg,
        text_atts,
        text_atts_neg,
    ):
        image_embeds_all = torch.cat([image_embeds_neg, image_embeds], dim=0)
        text_embeds_all = torch.cat([text_embeds, text_embeds_neg], dim=0)
        text_atts_all = torch.cat([text_atts, text_atts_neg], dim=0)
        output_pos = self.multimodal_encoder(
            image_embeds=image_embeds, text_embeds=text_embeds, text_atts=text_atts
        )
        output_neg = self.multimodal_encoder(
            image_embeds=image_embeds_all,
            text_embeds=text_embeds_all,
            text_atts=text_atts_all,
        )
        vl_embeddings = torch.cat([output_pos, output_neg], dim=0)
        return vl_embeddings
