# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import math
from collections import OrderedDict
from typing import Dict, List, NamedTuple, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torchmultimodal.models.mdetr.image_encoder import (
    mdetr_resnet101_backbone,
    PositionEmbedding2D,
)
from torchmultimodal.models.mdetr.text_encoder import (
    FeatureResizer,
    mdetr_roberta_text_encoder,
)
from torchmultimodal.models.mdetr.transformer import (
    mdetr_transformer,
    MDETRTransformerOutput,
)
from torchmultimodal.modules.layers.mlp import MLP


class MDETRModelOutput(NamedTuple):
    transformer_output: MDETRTransformerOutput
    pred_logits: torch.Tensor
    pred_boxes: torch.Tensor
    extra_embeddings: Optional[torch.Tensor]


class MDETR(nn.Module):
    """
    MDETR (https://arxiv.org/abs/2104.12763) is a modulated detection model
    used to detect objects in an image conditioned on text or captions.
    This class contains the entire MDETR architecture, including the
    image backbone, text encoder, and multimodal transformer. (Note that the
    matcher and losses are provided elsewhere.)

    Args:   image_backbone (nn.Module): Torch module of the backbone to be used.
                See image_encoder.py.
            text_encoder (nn.Module): Torch module of the text encoder to be used.
                See text_encoder.py.
            transformer (nn.Module): The multimodal transformer module. See the
                Transformer class in this file.
            pos_embed (nn.Module): Module for positional embedding of images.
            text_projection (nn.Module): Module to resize text encoder outputs before feeding
                them to the multimodal transformer.
            image_projection (nn.Module): Projection module applied to image embeddings
                prior to the multimodal transformer.
            query_embed (nn.Module): Learned object query embeddings (used in
                transformer decoder).
            bbox_embed (nn.Module): Embedding mapping transformer outputs to
                bounding boxes.
            class_embed (nn.Module): Embedding mapping transformer outputs to classes.
            extra_query_embeddings (Optional[nn.Embedding]): Additional query embeddings,
                as used in e.g. VQA. Default: None

    Inputs: images (List[Tensor]): A list of image Tensors (possibly of different sizes).
            text (List[Tensor]): A list of Tensors of tokenized texts (possibly of different lengths).

    Returns:
        A dict with the following elements:
           - "pred_logits": the classification logits (including no-object) for all queries.
                            Shape= [batch_size x num_queries x (num_classes + 1)]
           - "pred_boxes": The normalized boxes coordinates for all queries, represented as
                           (center_x, center_y, height, width). These values are normalized in [0, 1],
                           relative to the size of each individual image (disregarding possible padding).
                           See PostProcess for information on how to retrieve the unnormalized bounding box.
    """

    def __init__(
        self,
        image_backbone: nn.Module,
        text_encoder: nn.Module,
        transformer: nn.Module,
        pos_embed: nn.Module,
        text_projection: nn.Module,
        image_projection: nn.Module,
        query_embed: nn.Embedding,
        bbox_embed: nn.Module,
        class_embed: nn.Module,
        extra_query_embeddings: Optional[nn.Embedding] = None,
    ):
        super().__init__()
        self.image_backbone = image_backbone
        self.text_encoder = text_encoder
        self.text_projection = text_projection
        self.transformer = transformer
        self.pos_embed = pos_embed
        self.image_projection = image_projection
        self.query_embed = query_embed
        self.bbox_embed = bbox_embed
        self.class_embed = class_embed
        self.extra_query_embeddings = extra_query_embeddings

    def _pad_images(self, images: List[Tensor]) -> Tuple[Tensor, Tensor]:
        max_size = tuple(max(s) for s in zip(*[img.shape for img in images]))
        batch_shape = (len(images),) + max_size
        b, _, h, w = batch_shape

        dtype = images[0].dtype
        device = images[0].device
        padded_images = torch.zeros(batch_shape, dtype=dtype, device=device)
        mask = torch.ones((b, h, w), dtype=torch.bool, device=device)
        for img, pad_img, m in zip(images, padded_images, mask):
            pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)
            m[: img.shape[1], : img.shape[2]] = False
        return padded_images, mask

    def _pad_text(
        self, text: List[Tensor], padding_idx: int = 1
    ) -> Tuple[Tensor, Tensor]:
        padded_text = nn.utils.rnn.pad_sequence(
            text, batch_first=True, padding_value=padding_idx
        )
        mask = padded_text == padding_idx
        return padded_text, mask

    def forward(self, images: List[Tensor], text: List[Tensor]) -> MDETRModelOutput:

        images, image_mask = self._pad_images(images)
        text, text_attention_mask = self._pad_text(text)
        encoded_text = self.text_encoder(text, text_attention_mask)

        # Transpose memory because pytorch's attention expects sequence first
        text_memory = encoded_text.last_hidden_state.transpose(0, 1)

        image_embeddings, image_mask = self.image_backbone(images, image_mask)
        pos = self.pos_embed(image_mask).to(image_embeddings.dtype)
        query_embed = self.query_embed.weight

        # If extra embeddings are provided for VQA, we concatenate them with
        # the other query embeddings prior to the transformer
        if self.extra_query_embeddings is not None:
            n_extra_embeddings = self.extra_query_embeddings.num_embeddings
            query_embed = torch.cat([query_embed, self.extra_query_embeddings.weight])

        text_memory_resized = self.text_projection(text_memory)

        transformer_output = self.transformer(
            self.image_projection(image_embeddings),
            image_mask,
            query_embed,
            pos,
            text_memory=text_memory_resized,
            text_attention_mask=text_attention_mask,
        )

        # Detach the extra embeddings from the hidden states returned by the decoder
        if self.extra_query_embeddings is not None:
            extra_embeddings = transformer_output.decoder_hidden_states[
                0, :, -n_extra_embeddings:
            ]
            decoder_hidden_states_truncated = transformer_output.decoder_hidden_states[
                :, :, :-n_extra_embeddings
            ]
            transformer_output = transformer_output._replace(
                decoder_hidden_states=decoder_hidden_states_truncated
            )
        else:
            extra_embeddings = None
        final_hidden_state = transformer_output.decoder_hidden_states[-1]
        outputs_class = self.class_embed(final_hidden_state)
        outputs_coord = self.bbox_embed(final_hidden_state).sigmoid()

        return MDETRModelOutput(
            transformer_output, outputs_class, outputs_coord, extra_embeddings
        )


def mdetr_resnet101(
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
    num_extra_query_embeddings: Optional[int] = None,
) -> MDETR:
    image_backbone = mdetr_resnet101_backbone()
    pos_embed = PositionEmbedding2D(128, scale=2 * math.pi)
    # Size of layer4 output in ResNet101. See
    # https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py#L204
    image_backbone_num_channels = 2048
    text_encoder = mdetr_roberta_text_encoder()
    transformer = mdetr_transformer(
        transformer_d_model,
        transformer_num_heads,
        transformer_encoder_layers,
        transformer_decoder_layers,
        transformer_dim_feedforward,
        transformer_dropout,
        return_intermediate_dec,
    )
    hidden_dim = transformer_d_model
    text_projection = FeatureResizer(embedding_dim, hidden_dim)
    image_projection = nn.Conv2d(image_backbone_num_channels, hidden_dim, kernel_size=1)
    query_embed = nn.Embedding(num_queries, hidden_dim)
    # 4 gives the number of coordinates that represent the bounding box
    bbox_embed = MLP(hidden_dim, 4, [hidden_dim] * 2, dropout=0.0)
    # The + 1 here corresponds to the "no class" label
    class_embed = nn.Linear(hidden_dim, num_classes + 1)
    if num_extra_query_embeddings is not None:
        extra_query_embeddings = nn.Embedding(num_extra_query_embeddings, hidden_dim)
    else:
        extra_query_embeddings = None

    mdetr = MDETR(
        image_backbone,
        text_encoder,
        transformer,
        pos_embed,
        text_projection,
        image_projection,
        query_embed,
        bbox_embed,
        class_embed,
        extra_query_embeddings,
    )
    return mdetr


def mdetr_gqa_heads(hidden_dim: int = 256) -> nn.ModuleDict:
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
    return heads


class ContrastiveEmbeddingsOutput(NamedTuple):
    query_embeddings: Tensor
    token_embeddings: Tensor


class MDETRVQAOutput(NamedTuple):
    model_output: MDETRModelOutput
    vqa_preds: Dict[str, Tensor]
    contrastive_embeddings: ContrastiveEmbeddingsOutput


class MDETRForVQA(nn.Module):
    def __init__(
        self,
        model: MDETR,
        vqa_heads: nn.ModuleDict,
        contrastive_alignment_image_projection: nn.Module,
        contrastive_alignment_text_projection: nn.Module,
    ):
        super().__init__()
        self.model = model
        self.vqa_heads = vqa_heads
        if self.model.extra_query_embeddings is None:
            raise ValueError("MDETRForVQA requires extra query embeddings ")
        if self.model.extra_query_embeddings.num_embeddings != len(
            self.vqa_heads.keys()
        ):
            raise ValueError("Number of heads must match number of QA embeddings")

        self.contrastive_alignment_image_projection = (
            contrastive_alignment_image_projection
        )
        self.contrastive_alignment_text_projection = (
            contrastive_alignment_text_projection
        )

    def forward(
        self,
        images: List[Tensor],
        text: List[Tensor],
    ) -> MDETRVQAOutput:
        # Calculate MDETR model outputs
        model_output = self.model(images, text)

        final_hidden_state = model_output.transformer_output.decoder_hidden_states[-1]

        # Perform projections for contrastive alignment loss.
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
        contrastive_outputs = ContrastiveEmbeddingsOutput(
            contrastive_query_embeddings, contrastive_token_embeddings
        )

        # Apply VQA heads to get answer predictions
        answer_preds = OrderedDict()
        vqa_embeddings = model_output.extra_embeddings.transpose(0, 1)
        for (head_name, head), embedding in zip(self.vqa_heads.items(), vqa_embeddings):
            answer_preds[head_name] = head(embedding)

        return MDETRVQAOutput(model_output, answer_preds, contrastive_outputs)


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
    vqa_heads: Optional[nn.ModuleDict] = None,
    contrastive_dim: int = 64,
) -> MDETRForVQA:
    if vqa_heads is None:
        vqa_heads = mdetr_gqa_heads()

    hidden_dim = transformer_d_model
    num_heads = len(vqa_heads.keys())

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
    contrastive_alignment_image_projection = nn.Linear(hidden_dim, contrastive_dim)
    contrastive_alignment_text_projection = nn.Linear(hidden_dim, contrastive_dim)

    return MDETRForVQA(
        model,
        vqa_heads,
        contrastive_alignment_image_projection,
        contrastive_alignment_text_projection,
    )


class MDETRPhraseGroundingOutput(NamedTuple):
    model_output: MDETRModelOutput
    contrastive_embeddings: ContrastiveEmbeddingsOutput


class MDETRForPhraseGrounding(nn.Module):
    def __init__(
        self,
        model: MDETR,
        contrastive_alignment_image_projection: nn.Module,
        contrastive_alignment_text_projection: nn.Module,
    ):
        super().__init__()
        self.model = model
        self.contrastive_alignment_image_projection = (
            contrastive_alignment_image_projection
        )
        self.contrastive_alignment_text_projection = (
            contrastive_alignment_text_projection
        )

    def forward(
        self,
        images: List[Tensor],
        text: List[Tensor],
    ) -> MDETRPhraseGroundingOutput:

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
        contrastive_outputs = ContrastiveEmbeddingsOutput(
            contrastive_query_embeddings, contrastive_token_embeddings
        )

        return MDETRPhraseGroundingOutput(model_output, contrastive_outputs)


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
    contrastive_dim: int = 64,
) -> MDETRForPhraseGrounding:
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

    return MDETRForPhraseGrounding(
        model,
        contrastive_alignment_image_projection,
        contrastive_alignment_text_projection,
    )
