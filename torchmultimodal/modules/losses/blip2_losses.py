# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from typing import Optional, OrderedDict, Tuple

import torch
from torch import nn
from torch.nn import functional as F
from torchmultimodal.models.blip2.blip2 import Blip2Output
from torchmultimodal.utils.distributed import (
    BackpropType,
    concat_gather_all_gpu,
    get_rank,
)


@dataclass
class Blip2Stage1Losses(OrderedDict):
    "Blip-2 stage 1 losses"
    image_text_contrastive_loss: torch.Tensor
    image_text_matching_loss: torch.Tensor
    image_captioning_loss: torch.Tensor
    total_loss: torch.Tensor


def compute_image_text_similarity(
    image_features: torch.Tensor, text_features: torch.Tensor, temp: nn.Parameter
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute image-text similarity across all the devices for itc and itm usage.

    Inputs:
        image_features (torch.Tensor): Blip2 image output of shape [bsz, num_query_tokens, embed_dim]
        text_features (torch.Tensor): Blip2 text output of shape [bsz, embed_dim]
        temp (nn.Parameter): Temperature parameter

    Returns:
        a tuple of tensor contains image-to-text similarity and text-to-image similarity.
    """
    image_features_all = concat_gather_all_gpu(
        image_features, backprop_type=BackpropType.NONE
    )  # [bsz x num_gpu, num_query_tokens, embed_dim]
    text_features_all = concat_gather_all_gpu(
        text_features, backprop_type=BackpropType.NONE
    )  # [bsz x num_gpu, embed_dim]
    sim_q2t = torch.matmul(
        image_features.unsqueeze(1), text_features_all.unsqueeze(-1)
    ).squeeze()
    # [bsz, bsz x num_gpu, num_query_tokens]

    # image-text similarity: aggregate across all query tokens
    sim_i2t, _ = sim_q2t.max(-1)
    sim_i2t = sim_i2t / temp

    # text-query similarity: [bsz, bsz x num_gpu, num_query_tokens]
    sim_t2q = torch.matmul(
        text_features.unsqueeze(1).unsqueeze(1), image_features_all.permute(0, 2, 1)
    ).squeeze()

    # text-image similarity: aggregate across all query tokens
    sim_t2i, _ = sim_t2q.max(-1)
    sim_t2i = sim_t2i / temp  # [bsz, bsz x num_gpu]

    return sim_i2t, sim_t2i


def itc_loss(
    sim_i2t: torch.Tensor,
    sim_t2i: torch.Tensor,
    label_smoothing: float = 0.1,
) -> torch.Tensor:
    """Compute image-text contrastive loss by given similarity between image and text.

    Inputs:
        sim_i2t(torch.Tensor): image-to-text similarity, shape [bsz, bsz x num_gpu]
        sim_t2i (torch.Tensor): text-to-image similarity, shape [bsz, bsz x num_gpu]
        label_smoothing (Optional[float]): Label smoothing for cross-entropy. Default: 0.1.

    Returns:
        itc_loss (torch.Tensor)
    """
    rank = get_rank()

    local_batch_size = sim_i2t.size(0)
    targets = local_batch_size * rank + torch.arange(
        local_batch_size, device=sim_i2t.device
    )

    loss = (
        F.cross_entropy(sim_i2t, targets, label_smoothing=label_smoothing)
        + F.cross_entropy(sim_t2i, targets, label_smoothing=label_smoothing)
    ) / 2
    return loss


def itg_loss(
    input_ids: torch.Tensor,
    prediction_scores: torch.Tensor,
    decoder_bos_token_id: int,
    pad_token_id: int,
    vocab_size: int,
    reduction: str = "mean",
    label_smoothing: float = 0.1,
) -> torch.Tensor:
    """Compute image caption loss from BLIP2 predictions.

    Inputs:
        input_ids (torch.Tensor): text input ids of shape (bsz, seq_len).
        prediction_scores (torch.Tensor): BLIP2 prediction scores, shape of (bsz, seq_len, vocab_size)
        decoder_bos_token_id (int): bos_token_id for decoder, which is used to replace CLS token.
        pad_token_id (int): pad_token_id for decoder
        vocab_size (int): vocab size of BLIP2 model
        reduction (str): reduction for loss computation, default is "mean".
        label_smoothing (float): label smoothing value for cross-entropy loss, default is 0.1.

    Returns:
        itg_loss (torch.Tensor): image caption loss.
    """
    decoder_input_ids = input_ids.clone()
    # Replace CLS token to signal the decoding task as mentioned in paper https://arxiv.org/pdf/2301.12597.pdf
    decoder_input_ids[:, 0] = decoder_bos_token_id
    labels = decoder_input_ids.masked_fill(decoder_input_ids == pad_token_id, -100)
    shifted_prediction_scores = prediction_scores[:, :-1, :].contiguous()
    labels = labels[:, 1:].contiguous()
    itg_loss = F.cross_entropy(
        shifted_prediction_scores.view(-1, vocab_size),
        labels.view(-1),
        reduction=reduction,
        label_smoothing=label_smoothing,
    )

    return itg_loss


# TODO: upstream itm_loss for other model usage
def itm_loss(
    input_ids: torch.Tensor,
    image_embeds: torch.Tensor,
    sim_i2t: torch.Tensor,
    sim_t2i: torch.Tensor,
    model_query_tokens: nn.Parameter,
    qformer: nn.Module,
    itm_head: nn.Module,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Compute image-text matching loss
    ITM loss computation uses hard negative mining strategy. Negative text and image examples
    are selected based on their corresponding similarities.

    The concatenated image-text pairs are constructed as a size of 3 x bsz batch (pos, neg, neg)
    with text concatenated inputs (pos, pos, neg) and image inputs (pos, neg, pos).

    Query embedding output are fed into a two-class linear classifier to obtain a logit,
    and average the logits across all queries as the output matching score.

    Inputs:
        input_ids (torch.Tensor): text input ids of shape [bsz, seq_len].
        image_embeds (torch.Tensor): image embeddings returned by vision encoder
            with shape [bsz, image_embedding_dim]
        sim_i2t (torch.Tensor): image-to-text similarity, shape [bsz, bsz x num_gpu]
        sim_t2i (torch.Tensor): text-to-image similarity, shape [bsz, bsz x num_gpu]
        model_query_tokens(nn.Parameter): Blip2 query tokens
        qformer (nn.Module): Q-Former module
        itm_head (nn.Module): ITM head defined in blip2 stage1 loss
        attention_mask (torch.Tensor): attention mask for text input, shape [bsz, seq_len].

    Returns:
        itm_loss (torch.Tensor): image-text matching loss
    """
    local_batch_size = image_embeds.size(0)
    device = image_embeds.device
    text_input_ids_all_gpus = concat_gather_all_gpu(
        input_ids,
        backprop_type=BackpropType.NONE,
    )

    text_attention_mask_all_gpus = concat_gather_all_gpu(
        attention_mask,
        backprop_type=BackpropType.NONE,
    )
    image_embeds_all_gpus = concat_gather_all_gpu(
        image_embeds, backprop_type=BackpropType.GLOBAL
    )
    rank = get_rank()
    # compute weights for negative sample selection
    with torch.no_grad():
        weights_t2i_for_neg_sampling = F.softmax(sim_t2i, dim=1) + 1e-4
        weights_t2i_for_neg_sampling[
            :, rank * local_batch_size : rank * local_batch_size + local_batch_size
        ].fill_diagonal_(0)
        weights_i2t_for_neg_sampling = F.softmax(sim_i2t, dim=1) + 1e-4
        weights_i2t_for_neg_sampling[
            :, rank * local_batch_size : rank * local_batch_size + local_batch_size
        ].fill_diagonal_(0)

    # select a negative image for each text
    image_embeds_neg = []
    for b in range(local_batch_size):
        neg_idx = int(torch.multinomial(weights_t2i_for_neg_sampling[b], 1).item())
        image_embeds_neg.append(image_embeds_all_gpus[neg_idx])
    image_embeds_neg = torch.stack(image_embeds_neg, dim=0)

    # select a negative text for each image
    text_ids_neg = []
    text_atts_neg = []
    for b in range(local_batch_size):
        neg_idx = int(torch.multinomial(weights_i2t_for_neg_sampling[b], 1).item())
        text_ids_neg.append(text_input_ids_all_gpus[neg_idx])
        text_atts_neg.append(text_attention_mask_all_gpus[neg_idx])

    text_ids_neg = torch.stack(text_ids_neg, dim=0)
    text_atts_neg = torch.stack(text_atts_neg, dim=0)

    text_ids_all = torch.cat(
        [input_ids, input_ids, text_ids_neg], dim=0
    )  # pos, pos, neg
    text_atts_all = torch.cat(
        [attention_mask, attention_mask, text_atts_neg],
        dim=0,
    )

    query_tokens_itm = model_query_tokens.expand(text_ids_all.shape[0], -1, -1)
    query_atts_itm = torch.ones(query_tokens_itm.size()[:-1], dtype=torch.long).to(
        device
    )
    attention_mask_all = torch.cat([query_atts_itm, text_atts_all], dim=1)

    image_embeds_all = torch.cat(
        [image_embeds, image_embeds_neg, image_embeds], dim=0
    )  # pos, neg, pos
    output_itm = qformer(
        input_ids=text_ids_all,
        query_embeds=query_tokens_itm,
        attention_mask=attention_mask_all,
        encoder_hidden_states=image_embeds_all,
    )
    vl_embeddings = output_itm[0][
        :, : query_tokens_itm.size(1), :
    ]  # [bsz x 3, query_token_len, dim_q]
    vl_output = itm_head(vl_embeddings)  # [bsz x 3, query_token_len, 2]
    itm_logits = vl_output.mean(dim=1)

    itm_labels = torch.cat(
        [
            torch.ones(local_batch_size, dtype=torch.long),
            torch.zeros(2 * local_batch_size, dtype=torch.long),
        ],
        dim=0,
    ).to(device)

    return F.cross_entropy(itm_logits, itm_labels, reduction="mean")


class Blip2Phase1Loss(nn.Module):
    """
    Blip2 phase 1 loss module

    Args:
        dim_q (int): Dimension of query tensor, this value should be the same as dim_q in qformer.
            default value is 768 as in the paper.
        enable_itc (bool): enable image-text contrastive loss, default is True
        enable_itm (bool): enable image-text matching, default is True
        enable_itg (bool): enable image caption loss, default is True
        temp (float): temperature for image-text similarity computation, default is 0.07
        label_smoothing (float): label smoothing value, default is 0.1
    """

    def __init__(
        self,
        dim_q: int = 768,
        enable_itc: bool = True,
        enable_itm: bool = True,
        enable_itg: bool = True,
        temp: float = 0.07,
        label_smoothing: float = 0.1,
    ) -> None:
        super().__init__()
        if not enable_itc and not enable_itm and not enable_itg:
            raise ValueError(
                "All the loss tasks are disabled, please set at least one of them."
            )
        self.label_smoothing = label_smoothing
        self.enable_itc = enable_itc
        self.enable_itm = enable_itm
        self.enable_itg = enable_itg
        self.itm_head = nn.Linear(dim_q, 2)
        self.temp = nn.Parameter(temp * torch.ones([]))

    def forward(
        self,
        model_output: Blip2Output,
        blip2: nn.Module,
        input_ids: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor],
    ) -> Blip2Stage1Losses:
        """
        Inputs:
            model_output (Blip2Output): model output from BLIP2 (see blip2.py)
            blip2 (nn.Module): BLIP2 model with updated params
            input_ids (Optional[torch.Tensor]): text input ids of shape [bsz, seq_len].
            attention_mask (Optional[torch.Tensor]): text input attention mask of shape [bsz, seq_len].

        Returns:
            loss (Blip2Stage1Losses): computed loss for phase 1 tasks.
        """

        # calculate similarities
        assert model_output.text_features is not None
        (sim_i2t, sim_t2i,) = compute_image_text_similarity(
            model_output.image_features,
            model_output.text_features,
            temp=self.temp,
        )

        # calculate image-text matching loss
        loss_itm = torch.tensor(0.0)
        if self.enable_itm:
            assert input_ids is not None and attention_mask is not None
            loss_itm = itm_loss(
                input_ids=input_ids,
                attention_mask=attention_mask,
                image_embeds=model_output.image_embeddings,
                sim_i2t=sim_i2t,
                sim_t2i=sim_t2i,
                model_query_tokens=blip2.query_tokens,
                qformer=blip2.qformer.model,
                itm_head=self.itm_head,
            )

        # calculate image captioning loss (aka image-text generation)
        loss_itg = torch.tensor(0.0)
        if self.enable_itg:
            assert input_ids is not None and model_output.prediction_scores is not None
            loss_itg = itg_loss(
                input_ids=input_ids,
                prediction_scores=model_output.prediction_scores,
                decoder_bos_token_id=blip2.decoder_bos_token_id,
                pad_token_id=blip2.qformer.pad_token_id,
                vocab_size=blip2.qformer.vocab_size,
                label_smoothing=self.label_smoothing,
            )

        # calculate image-text contrastive loss
        loss_itc = torch.tensor(0.0)
        if self.enable_itc:
            loss_itc = itc_loss(
                sim_i2t=sim_i2t,
                sim_t2i=sim_t2i,
                label_smoothing=self.label_smoothing,
            )

        return Blip2Stage1Losses(
            image_text_contrastive_loss=loss_itc,
            image_captioning_loss=loss_itg,
            image_text_matching_loss=loss_itm,
            total_loss=loss_itc + loss_itm + loss_itg,
        )
