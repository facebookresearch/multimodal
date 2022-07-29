# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from collections import OrderedDict
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torchmultimodal.modules.losses.mdetr import BoxLosses


def contrastive_alignment_loss(
    projected_queries: Tensor,
    projected_tokens: Tensor,
    target_tokens: List[List[List[int]]],
    indices: List[Tuple[Tensor, Tensor]],
    num_boxes: int,
    tokenized: Any,
    temperature: float = 0.07,
) -> Tensor:
    """Contrastive alignment loss.
    Enforces alignment between the text representations after cross encoder and the
    object representations after the decoder.
                projected_queries (Tensor): Tensor containing object representations
                    projected to query dimension.
                    Size: (batch_size, num_queries, contrastive_dim)
                projected_tokens: Tensor containing text representations projected
                    to token dimension.
                    Size: (batch_size, num_tokens, contrastive_dim)
                target_tokens (List[List[List[int]]]): A very nested list of tokens
                    that correspond to each target. From outermost to innermost:
                    batch, object, list of disjoint (start, end) tokens
                indices (List[Tuple[Tensor, Tensor]]): A list of size batch_size,
                containing tuples of (index_i, index_j) where:
                    - index_i is the indices of the selected predictions (in order)
                    - index_j is the indices of the corresponding selected targets
                For each batch element, it holds:
                    len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
            num_boxes (int): Normalization factor. Should equal the average number of
                boxes per local batch.
            tokenized (Any): Tokenized output from a transformers fast tokenizer.
                Used for token lookup based on character positions.
            temperature (float): Scaling factor used in calculating the logits.
                Default: 0.07
    """
    logits = (
        torch.matmul(projected_queries, projected_tokens.transpose(-1, -2))
        / temperature
    )  # BS x (num_queries) x (num_tokens)
    positive_map = construct_positive_map(logits, target_tokens, indices, tokenized)

    positive_logits = -logits.masked_fill(~positive_map, 0)
    negative_logits = logits

    # Calculate the contrastive loss for all objects
    boxes_with_pos = positive_map.any(2)
    pos_term = positive_logits.sum(2)
    neg_term = negative_logits.logsumexp(2)
    nb_pos = positive_map.sum(2) + 1e-6
    box_to_token_loss = (
        ((pos_term / nb_pos + neg_term)).masked_fill(~boxes_with_pos, 0).sum()
    )

    # Calculate the contrastive loss for all tokens
    tokens_with_pos = positive_map.any(1)
    pos_term = positive_logits.sum(1)
    neg_term = negative_logits.logsumexp(1)
    nb_pos = positive_map.sum(1) + 1e-6
    tokens_to_boxes_loss = (
        ((pos_term / nb_pos + neg_term)).masked_fill(~tokens_with_pos, 0).sum()
    )

    tot_loss = (box_to_token_loss + tokens_to_boxes_loss) / 2

    return tot_loss / num_boxes


def char_to_token(
    encodings,
    batch_or_char_index: int,
    char_index: Optional[int] = None,
    sequence_index: int = 0,
):
    if char_index is not None:
        batch_index = batch_or_char_index
    else:
        batch_index = 0
        char_index = batch_or_char_index
    return encodings[batch_index].char_to_token(char_index, sequence_index)


def construct_positive_map(
    logits: Tensor,
    target_tokens: List[List[List[int]]],
    indices: List[Tuple[Tensor, Tensor]],
    tokenized: Any,
):
    # construct a map such that positive_map[k, i,j] = True iff query i is associated to token j in batch item k
    # For efficency, the construction happens on CPU, then the whole matrix is transferred to GPU in one go.
    positive_map = torch.zeros(logits.shape, dtype=torch.bool)
    for i, ((idx_src, idx_tgt), tgt) in enumerate(zip(indices, target_tokens)):
        cur_tokens = [tgt[j] for j in idx_tgt]
        for j, tok_list in enumerate(cur_tokens):
            for (beg, end) in tok_list:
                beg_pos = char_to_token(tokenized, i, beg)
                end_pos = char_to_token(tokenized, i, end - 1)

                if beg_pos is None and end_pos is None:
                    raise ValueError(
                        "At least one of beg_pos and end_pos must not be None"
                    )
                positive_map[i, idx_src[j], beg_pos : end_pos + 1].fill_(True)
    return positive_map.to(logits.device)


def masked_dict_accuracy(
    pred_dict: Optional[Dict[str, Tensor]] = None,
    label_dict: Optional[Dict[str, Tensor]] = None,
    mask_dict: Optional[Dict[str, Tensor]] = None,
    answer_type_key: Optional[str] = "answer_type",
):
    accuracies = OrderedDict()
    for k in pred_dict.keys():
        if mask_dict is None or mask_dict[k] is None:
            mask = torch.ones_like(pred_dict[k])
        else:
            mask = mask_dict[k]
        accuracies[f"{k}_accuracy"] = (
            (pred_dict[k][mask].argmax(-1) == label_dict[k][mask]).sum() / mask.sum()
            if mask.any()
            else torch.as_tensor(1.0, device=mask.device)
        )
    weighted = sum(
        [
            accuracies[f"{k}_accuracy"] * mask_dict[k].sum()
            for k in pred_dict.keys()
            if k != answer_type_key
        ]
    )
    accuracies["answer_total_accuracy"] = (
        accuracies[f"{answer_type_key}_accuracy"]
        * weighted
        / label_dict[answer_type_key].numel()
    )
    return accuracies


def masked_dict_cross_entropy(
    pred_dict: Optional[Dict[str, Tensor]] = None,
    label_dict: Optional[Dict[str, Tensor]] = None,
    mask_dict: Optional[Dict[str, Tensor]] = None,
):
    losses = OrderedDict()
    if pred_dict.keys() != label_dict.keys():
        raise ValueError("Keys of pred_dict and label_dict must match")
    for k in pred_dict.keys():
        if mask_dict is None or mask_dict[k] is None:
            mask = torch.ones_like(pred_dict[k])
        else:
            mask = mask_dict[k]
        norm_factor = mask.sum() if mask.any() else 1.0
        losses[f"{k}_loss"] = (
            F.cross_entropy(pred_dict[k], label_dict[k]).masked_fill(~mask, 0).sum()
            / norm_factor
        )

    return losses


class MDETRLoss(nn.Module):
    def __init__(
        self,
        soft_token_loss: Callable[..., Tensor],
        box_losses: Callable[..., BoxLosses],
        contrastive_alignment_loss: Optional[nn.Module] = None,
        vqa_losses: Optional[Iterable[nn.Module]] = None,
    ):
        super().__init__()
        self.soft_token_loss = soft_token_loss
        self.box_losses = box_losses
        self.contrastive_alignment_loss = contrastive_alignment_loss
        self.vqa_losses = vqa_losses

    def get_average_num_boxes_across_workers(self, num_boxes: Tensor):
        # Compute the average number of target boxes across all workers for normalization purposes
        if not (
            torch.distributed.is_available() and torch.distributed.is_initialized()
        ):
            return torch.clamp(num_boxes, min=1).item()
        torch.distributed.all_reduce(num_boxes)
        num_boxes_all_workers = torch.clamp(
            num_boxes / torch.distributed.get_world_size(), min=1
        ).item()
        return num_boxes_all_workers

    def total_losses_with_weights(
        self,
        loss_dict: Dict[str, Tensor],
        weight_dict: Optional[Dict[str, float]] = None,
    ) -> torch.Tensor:
        for k in weight_dict.keys():
            if k not in loss_dict.keys():
                raise ValueError(f"Weight dict contains invalid key {k}")
        return sum([weight_dict[k] * loss_dict[k] for k in weight_dict.keys()])

    def forward(
        self,
        pred_logits: Tensor,
        pred_boxes: Tensor,
        targets: List[Dict[str, Any]],
        positive_map,
        indices: List[Tuple[Tensor, Tensor]],
        contrastive_query_embeddings: Optional[Tensor] = None,
        contrastive_token_embeddings: Optional[Tensor] = None,
        tokenized: Optional[Any] = None,
        vqa_preds: Optional[Dict[str, Tensor]] = None,
        vqa_labels: Optional[Dict[str, Tensor]] = None,
        vqa_masks: Optional[Dict[str, Tensor]] = None,
        weight_dict: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Tensor]:

        target_boxes = [t["boxes"] for t in targets]
        target_tokens = [t["tokens_positive"] for t in targets]
        n_target_boxes = [len(t) for t in target_boxes]
        num_boxes = sum(n_target_boxes)
        num_boxes = torch.as_tensor(
            [num_boxes], dtype=torch.float, device=pred_logits.device
        )
        num_boxes_all_workers = self.get_average_num_boxes_across_workers(num_boxes)

        self.pred_logits = pred_logits
        self.n_target_boxes = n_target_boxes
        self.positive_map = positive_map
        self.indices = indices
        self.num_boxes_all_workers = num_boxes_all_workers
        soft_token_loss = self.soft_token_loss(
            pred_logits, n_target_boxes, positive_map, indices, num_boxes_all_workers
        )
        box_losses = self.box_losses(
            pred_boxes, target_boxes, indices, num_boxes_all_workers
        )

        loss_dict = {
            "soft_token_loss": soft_token_loss,
            "l1_loss": box_losses.l1_loss,
            "giou_loss": box_losses.giou_loss,
        }

        if self.contrastive_alignment_loss is not None:
            if (
                contrastive_query_embeddings is None
                or contrastive_token_embeddings is None
                or tokenized is None
            ):
                raise ValueError(
                    "For contrastive alignment loss must pass contrastive query/token embeddings and tokenized text"
                )
            contrastive_alignment_loss = self.contrastive_alignment_loss(
                contrastive_query_embeddings,
                contrastive_token_embeddings,
                target_tokens,
                indices,
                num_boxes_all_workers,
                tokenized,
            )
            loss_dict.update(contrastive_alignment_loss=contrastive_alignment_loss)

        if self.vqa_losses is not None:
            if vqa_preds is None or vqa_labels is None:
                raise ValueError("For QA loss qa_preds and qa_labels must not be None")
            for vqa_loss in self.vqa_losses:
                loss_dict.update(vqa_loss(vqa_preds, vqa_labels, vqa_masks))

        if weight_dict is not None:
            total_loss = self.total_losses_with_weights(loss_dict, weight_dict)
            loss_dict.update(total_loss=total_loss)

        return loss_dict


def build_weight_dict(
    args,
    vqa_keys: Optional[Iterable[str]] = None,
    include_contrastive_loss: bool = True,
):
    weight_dict = {
        "soft_token_loss": args.ce_loss_coef,
        "l1_loss": args.bbox_loss_coef,
        "giou_loss": args.giou_loss_coef,
    }
    if vqa_keys is not None:
        for k in vqa_keys:
            weight_dict.update({f"{k}_loss": args.qa_loss_coef})
    if include_contrastive_loss:
        weight_dict.update(contrastive_alignment_loss=args.contrastive_align_loss_coef)
    return weight_dict
