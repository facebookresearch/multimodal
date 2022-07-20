# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from collections import OrderedDict
from copy import deepcopy
from functools import partial
from typing import Dict, List, NamedTuple, Optional

import torch
import torch.nn.functional as F
from examples.mdetr.loss import (
    contrastive_alignment_loss,
    masked_dict_accuracy,
    masked_dict_cross_entropy,
    MDETRLoss,
)
from examples.mdetr.matcher import HungarianMatcher
from torch import nn, Tensor
from torchmultimodal.models.mdetr import MDETR, mdetr_resnet101, MDETRModelOutput
from torchmultimodal.modules.losses.mdetr import box_losses, soft_token_prediction_loss

from transformers import BatchEncoding


class MultiHead(nn.Module):
    def __init__(self, heads: nn.ModuleDict):
        super().__init__()
        self.heads = heads

    def forward(
        self,
        embeddings: Tensor,
    ) -> Dict[str, Tensor]:
        if embeddings.size(0) != len(self.heads):
            raise ValueError("Number of embeddings must equal number of heads")
        out = OrderedDict()
        for (head_name, head), embedding in zip(self.heads.items(), embeddings):
            out[head_name] = head(embedding)

        return out


def mdetr_gqa_heads(hidden_dim: int = 256) -> MultiHead:
    answer_type_head = nn.Linear(hidden_dim, 5)  # Number of answer types
    answer_obj_head = nn.Linear(hidden_dim, 3)
    answer_attr_head = nn.Linear(hidden_dim, 403)
    answer_rel_head = nn.Linear(hidden_dim, 1594)
    answer_global_head = nn.Linear(hidden_dim, 111)
    answer_cat_head = nn.Linear(hidden_dim, 678)
    heads = nn.ModuleDict(
        {
            "answer_type": answer_type_head,
            "answer_obj": answer_obj_head,
            "answer_rel": answer_rel_head,
            "answer_attr": answer_attr_head,
            "answer_cat": answer_cat_head,
            "answer_global": answer_global_head,
        }
    )
    return MultiHead(heads)


class MDETRVQAOutput(NamedTuple):
    model_output: MDETRModelOutput
    vqa_preds: Dict[str, Tensor]
    loss: Dict[str, Tensor]


class MDETRForVQA(nn.Module):
    def __init__(
        self,
        model: MDETR,
        vqa_embed: nn.Embedding,
        vqa_heads: MultiHead,
        matcher: HungarianMatcher,
        loss: MDETRLoss,
    ):
        super().__init__()
        self.model = model
        self.vqa_embed = vqa_embed
        self.num_vqa_embeddings = self.vqa_embed.num_embeddings
        self.matcher = matcher
        self.vqa_heads = vqa_heads
        self.loss = loss
        if self.num_vqa_embeddings != len(self.vqa_heads.heads.keys()):
            raise ValueError("Number of heads must match number of QA embeddings")
        self.is_preprocessed = False

    def _pre_mdetr_forward(self) -> None:
        if not self.is_preprocessed:
            concat_embedding_weights = nn.Parameter(
                torch.cat(
                    [self.model.query_embed.weight, self.vqa_embed.weight], axis=0
                )
            )
            self.model.query_embed = nn.Embedding(
                self.model.query_embed.num_embeddings + self.vqa_embed.num_embeddings,
                self.vqa_embed.embedding_dim,
            )
            self.model.query_embed.weight = concat_embedding_weights
            self.bbox_embed = deepcopy(self.model.bbox_embed)
            self.class_embed = deepcopy(self.model.class_embed)
            self.model.bbox_embed = None
            self.model.class_embed = None
            self.is_preprocessed = True

    def _post_mdetr_forward(self, model_output: MDETRModelOutput) -> None:
        final_hidden_state = model_output.transformer_output.decoder_hidden_states[
            -1, :, : -self.num_vqa_embeddings
        ]
        pred_logits = self.class_embed(final_hidden_state)
        pred_boxes = self.bbox_embed(final_hidden_state).sigmoid()
        model_output = model_output._replace(
            pred_logits=pred_logits, pred_boxes=pred_boxes
        )
        return model_output

    def forward(
        self,
        images: List[Tensor],
        text: List[Tensor],
        targets,
        positive_map,
        answers,
        answer_types: Dict[str, Tensor],
        weight_dict: Optional[Dict[str, float]] = None,
    ) -> MDETRVQAOutput:
        self._pre_mdetr_forward()
        model_output = self.model(images, text)
        model_output = self._post_mdetr_forward(model_output)

        answer_embeddings = model_output.transformer_output.decoder_hidden_states[
            0, :, -self.num_vqa_embeddings :
        ].transpose(
            0, 1
        )  # (num_qa_embeddings, batch_size, embedding_dim)
        answer_preds = self.vqa_heads(answer_embeddings)
        target_boxes = [t["boxes"] for t in targets]
        indices = self.matcher(
            model_output.pred_logits,
            model_output.pred_boxes,
            target_boxes,
            positive_map,
        )
        loss = self.loss(
            model_output.pred_logits,
            model_output.pred_boxes,
            targets,
            positive_map,
            indices,
            vqa_preds=answer_preds,
            vqa_labels=answers,
            vqa_masks=answer_types,
            weight_dict=weight_dict,
        )

        return MDETRVQAOutput(model_output, answer_preds, loss)


def mdetr_for_vqa(
    num_queries: int = 100,
    num_classes: int = 255,
    embedding_dim: int = 768,
    transformer_d_model: int = 256,
    transformer_num_heads: int = 8,
    transformer_encoder_layers: int = 6,
    transformer_decoder_layers: int = 6,
    transformer_dim_feedforward: int = 2048,
    transformer_dropout: float = 0.1,
    return_intermediate_dec: bool = True,
    matcher_cost_class: int = 1,
    matcher_cost_bbox: int = 5,
    matcher_cost_giou: int = 2,
    no_object_weight: float = 0.1,
):
    model = mdetr_resnet101(
        num_queries,
        num_classes,
        embedding_dim,
        transformer_d_model,
        transformer_num_heads,
        transformer_encoder_layers,
        transformer_decoder_layers,
        transformer_dim_feedforward,
        transformer_dropout,
        return_intermediate_dec,
    )
    hidden_dim = transformer_d_model
    vqa_heads = mdetr_gqa_heads()
    num_heads = len(vqa_heads.heads.keys())
    vqa_embedding = nn.Embedding(num_heads, hidden_dim)

    matcher = HungarianMatcher(matcher_cost_class, matcher_cost_bbox, matcher_cost_giou)

    soft_token_loss = partial(
        soft_token_prediction_loss, no_object_weight=no_object_weight
    )
    loss = MDETRLoss(
        soft_token_loss=soft_token_loss,
        box_losses=box_losses,
        vqa_losses=[masked_dict_cross_entropy, masked_dict_accuracy],
    )

    return MDETRForVQA(model, vqa_embedding, vqa_heads, matcher, loss)


class MDETRForPhraseGrounding(nn.Module):
    def __init__(
        self,
        model: MDETR,
        contrastive_alignment_image_projection: nn.Module,
        contrastive_alignment_text_projection: nn.Module,
        matcher: HungarianMatcher,
        loss: MDETRLoss,
    ):
        super().__init__()
        self.model = model
        self.matcher = matcher
        self.contrastive_alignment_image_projection = (
            contrastive_alignment_image_projection
        )
        self.contrastive_alignment_text_projection = (
            contrastive_alignment_text_projection
        )
        self.loss = loss

    def forward(
        self,
        images: List[Tensor],
        text: List[Tensor],
        targets,
        positive_map,
        tokenized: BatchEncoding,
        weight_dict: Optional[Dict[str, float]] = None,
    ):
        model_output = self.model(images, text)
        final_hidden_state = model_output.transformer_output.decoder_hidden_states[-1]

        contrastive_query_embeddings = F.normalize(
            self.contrastive_alignment_image_projection(final_hidden_state),
            p=2,
            dim=-1,
        )
        contrastive_token_embeddings = F.normalize(
            self.contrastive_alignment_text_projection(
                model_output.transformer_output.text_memory
            ).transpose(0, 1),
            p=2,
            dim=-1,
        )

        target_boxes = [t["boxes"] for t in targets]
        indices = self.matcher(
            model_output.pred_logits,
            model_output.pred_boxes,
            target_boxes,
            positive_map,
        )

        loss = self.loss(
            model_output.pred_logits,
            model_output.pred_boxes,
            targets,
            positive_map,
            indices,
            contrastive_query_embeddings,
            contrastive_token_embeddings,
            tokenized,
            weight_dict=weight_dict,
        )
        return model_output, loss


def mdetr_for_phrase_grounding(
    num_queries: int = 100,
    num_classes: int = 255,
    embedding_dim: int = 768,
    transformer_d_model: int = 256,
    transformer_num_heads: int = 8,
    transformer_encoder_layers: int = 6,
    transformer_decoder_layers: int = 6,
    transformer_dim_feedforward: int = 2048,
    transformer_dropout: float = 0.1,
    return_intermediate_dec: bool = True,
    matcher_cost_class: int = 1,
    matcher_cost_bbox: int = 5,
    matcher_cost_giou: int = 2,
    contrastive_dim: int = 64,
    no_object_weight: float = 0.1,
    temperature: float = 0.07,
):
    model = mdetr_resnet101(
        num_queries,
        num_classes,
        embedding_dim,
        transformer_d_model,
        transformer_num_heads,
        transformer_encoder_layers,
        transformer_decoder_layers,
        transformer_dim_feedforward,
        transformer_dropout,
        return_intermediate_dec,
    )
    hidden_dim = transformer_d_model
    contrastive_alignment_image_projection = nn.Linear(hidden_dim, contrastive_dim)
    contrastive_alignment_text_projection = nn.Linear(hidden_dim, contrastive_dim)
    matcher = HungarianMatcher(matcher_cost_class, matcher_cost_bbox, matcher_cost_giou)

    soft_token_loss = partial(
        soft_token_prediction_loss, no_object_weight=no_object_weight
    )
    contrastive_loss = partial(contrastive_alignment_loss, temperature=temperature)
    loss = MDETRLoss(
        soft_token_loss=soft_token_loss,
        box_losses=box_losses,
        contrastive_alignment_loss=contrastive_loss,
    )

    return MDETRForPhraseGrounding(
        model,
        contrastive_alignment_image_projection,
        contrastive_alignment_text_projection,
        matcher,
        loss,
    )
