# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from collections import OrderedDict
from functools import partial
from typing import Any, Dict, List, NamedTuple, Optional

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
        vqa_heads: MultiHead,
        matcher: HungarianMatcher,
        loss: MDETRLoss,
        contrastive_alignment_image_projection: Optional[nn.Module] = None,
        contrastive_alignment_text_projection: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.model = model
        self.vqa_heads = vqa_heads
        if self.model.extra_query_embeddings is None:
            raise ValueError("MDETRForVQA requires extra query embeddings ")
        if self.model.extra_query_embeddings.num_embeddings != len(
            self.vqa_heads.heads.keys()
        ):
            raise ValueError("Number of heads must match number of QA embeddings")

        self.matcher = matcher
        self.loss = loss
        self.contrastive_alignment_image_projection = (
            contrastive_alignment_image_projection
        )
        self.contrastive_alignment_text_projection = (
            contrastive_alignment_text_projection
        )

    def _is_eligible_for_contrastive(self, tokenized: Optional[Any]):
        if (
            self.contrastive_alignment_image_projection is None
            or self.contrastive_alignment_text_projection is None
            or self.loss.contrastive_alignment_loss is None
            or tokenized is None
        ):
            return False
        return True

    def forward(
        self,
        images: List[Tensor],
        text: List[Tensor],
        targets,
        positive_map,
        answers,
        answer_types: Dict[str, Tensor],
        tokenized: Optional[Any],
        weight_dict: Optional[Dict[str, float]] = None,
        include_contrastive: bool = True,
    ) -> MDETRVQAOutput:
        # Calculate MDETR model outputs
        model_output = self.model(images, text)

        final_hidden_state = model_output.transformer_output.decoder_hidden_states[-1]

        # Contrastive loss is optional for VQA.
        # If it's being calculated we perform the projections here
        if include_contrastive:
            if not self._is_eligible_for_contrastive(tokenized):
                raise ValueError(
                    "Module is not eligible to calculate contrastive alignment loss"
                )

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
        else:
            contrastive_query_embeddings, contrastive_token_embeddings = None, None

        # Apply VQA heads to get answer predictions
        answer_preds = self.vqa_heads(model_output.extra_embeddings.transpose(0, 1))

        target_boxes = [t["boxes"] for t in targets]
        # Get the matching between predicted and target boxes
        indices = self.matcher(
            model_output.pred_logits,
            model_output.pred_boxes,
            target_boxes,
            positive_map,
        )

        # Calculate MDETR loss with VQA losses
        loss = self.loss(
            model_output.pred_logits,
            model_output.pred_boxes,
            targets,
            positive_map,
            indices,
            contrastive_query_embeddings,
            contrastive_token_embeddings,
            tokenized,
            answer_preds,
            answers,
            answer_types,
            weight_dict,
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
    contrastive_dim: Optional[int] = None,
    temperature: Optional[float] = None,
):
    hidden_dim = transformer_d_model
    vqa_heads = mdetr_gqa_heads()
    num_heads = len(vqa_heads.heads.keys())

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
        num_extra_query_embeddings=num_heads,
    )
    if contrastive_dim is not None:
        contrastive_alignment_image_projection = nn.Linear(hidden_dim, contrastive_dim)
        contrastive_alignment_text_projection = nn.Linear(hidden_dim, contrastive_dim)
        contrastive_loss = partial(contrastive_alignment_loss, temperature=temperature)
    else:
        (
            contrastive_alignment_image_projection,
            contrastive_alignment_text_projection,
            contrastive_loss,
        ) = (None, None, None)

    matcher = HungarianMatcher(matcher_cost_class, matcher_cost_bbox, matcher_cost_giou)

    soft_token_loss = partial(
        soft_token_prediction_loss, no_object_weight=no_object_weight
    )

    loss = MDETRLoss(
        soft_token_loss=soft_token_loss,
        box_losses=box_losses,
        contrastive_alignment_loss=contrastive_loss,
        vqa_losses=[masked_dict_cross_entropy, masked_dict_accuracy],
    )

    return MDETRForVQA(
        model,
        vqa_heads,
        matcher,
        loss,
        contrastive_alignment_image_projection,
        contrastive_alignment_text_projection,
    )


class MDETRPhraseGroundingOutput(NamedTuple):
    model_output: MDETRModelOutput
    loss: Dict[str, Tensor]


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
        tokenized: Any,
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
        return MDETRPhraseGroundingOutput(model_output, loss)


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
