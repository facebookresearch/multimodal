# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, NamedTuple, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor
from torchvision.ops.boxes import box_convert, generalized_box_iou


def _get_src_permutation_idx(
    indices: List[Tuple[Tensor, Tensor]]
) -> Tuple[Tensor, Tensor]:
    """
    Given a list of matched (src, tgt) indices, concatenate the src indices and
    return along with a tensor identifying which sample they came from.

    Args:
        indices (List[Tuple[Tensor, Tensor]]): A list of size batch_size, containing
            tuples of ``(index_i, index_j)`` where:
            - ``index_i`` is the indices of the selected predictions (i.e. srcs)
            - ``index_j`` is the indices of the corresponding selected targets
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
    Returns:
        A Tuple[Tensor, Tensor] x, where
        x[0] gives the index of the sample in the batch
        x[1] gives the src value from indices
        Both x[0] and x[1] have size = (sum([len(index_i) for index_i in indices]))
    """
    batch_idx = torch.cat(
        [torch.full_like(src, i) for i, (src, _) in enumerate(indices)]
    )
    src_idx = torch.cat([src for (src, _) in indices])
    return batch_idx, src_idx


# TODO: we can calculate num_boxes in dataloader and add it to targets
# Note: num_tokens = num_classes + 1. We should make this clear in other docstrings
def soft_token_prediction_loss(
    pred_logits: Tensor,
    n_target_boxes: List[int],
    positive_map: Tensor,
    indices: List[Tuple[Tensor, Tensor]],
    num_boxes: int,
    no_object_weight: float = 0.1,
) -> Tensor:
    """Classification loss (NLL).

    Calculate the negative log-likelihood loss between the predicted logits and the
    uniform distribution over matched tokens from the ground truth, as in MDETR. The
    loss for unmatched boxes is downweighted by the value no_object_weight.
    Ref: https://github.com/ashkamath/mdetr/blob/main/models/mdetr.py#L464

    Args:
        pred_logits (Tensor): Logits predicted by the model.
            Shape: (batch_size, num_queries, num_tokens)
        n_target_boxes (List[int]): Number of boxes in each target
        positive_map (Tensor): Map from boxes to tokens for the entire batch.
            positive_map[i,j] = 1 iff box i is associated to token j.
            Shape: (sum([len(target["boxes"]) for target in batch]), num_tokens)
        indices (List[Tuple[Tensor, Tensor]]): A list of size batch_size, containing
            tuples of ``(index_i, index_j)`` where:
            - ``index_i`` is the indices of the selected predictions (in order)
            - ``index_j`` is the indices of the corresponding selected targets
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        num_boxes (int): Normalization factor. Should equal the average number of
            boxes per local batch.
        no_object_weight (float): Relative classification weight of the no-object class.
    Returns:
        Negative log likelihood for the batch normalized by num_boxes
    """

    logits = pred_logits.log_softmax(-1)

    src_idx = _get_src_permutation_idx(indices)
    tgt_idx = []
    offset = 0
    for i, (_, tgt) in enumerate(indices):
        tgt_idx.append(tgt + offset)
        offset += n_target_boxes[i]

    # tgt_idx concatenates the target indices across samples in the batch,
    # giving each box a unique value which will be used to permute positive_map
    tgt_idx = torch.cat(tgt_idx)

    # Permute the rows of positive map based on target box indices
    tgt_pos = positive_map[tgt_idx]

    target_sim = torch.zeros_like(logits)

    # Default is the no match value
    target_sim[:, :, -1] = 1

    # Fill each of the corresponding rows of target_sim with the ground truth
    target_sim[src_idx] = tgt_pos

    loss_ce = -(logits * target_sim).sum(-1)

    # Downweight the loss for unmatched boxes by no_object_weight
    no_object_tensor = torch.full(
        loss_ce.shape, no_object_weight, device=target_sim.device
    )
    no_object_tensor[src_idx] = 1
    loss_ce = loss_ce * no_object_tensor
    loss_ce = loss_ce.sum() / num_boxes

    return loss_ce


class BoxLosses(NamedTuple):
    l1_loss: torch.Tensor
    giou_loss: torch.Tensor


def box_losses(
    pred_boxes: Tensor,
    target_boxes: List[Tensor],
    indices: List[Tuple[Tensor, Tensor]],
    num_boxes: int,
) -> BoxLosses:
    """Box losses: L1 loss and GIoU loss

    Inputs: pred_boxes (Tensor): Bounding boxes predicted by the model.
                Shape: (batch_size, num_queries, 4)
            target_boxes (List[Tensor]): List of box coordinates for each sample in batch.
                Length = batch_size, Tensor size = [len(target["boxes"]), 4]
            indices (List[Tuple[Tensor, Tensor]]): A list of size batch_size,
                containing tuples of (index_i, index_j) where:
                    - index_i is the indices of the selected predictions (in order)
                    - index_j is the indices of the corresponding selected targets
                For each batch element, it holds:
                    len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
            num_boxes (int): Normalization factor. Should equal the average number of
                boxes per local batch.
    Returns: BoxLosses NamedTuple with elements l1_loss and giou_loss
    """
    idx = _get_src_permutation_idx(indices)
    src_boxes = pred_boxes[idx]
    target_boxes = torch.cat([t[i] for t, (_, i) in zip(target_boxes, indices)], dim=0)

    l1_loss = F.l1_loss(src_boxes, target_boxes, reduction="sum") / num_boxes
    giou_loss = 1 - torch.diag(
        generalized_box_iou(
            box_convert(src_boxes, in_fmt="cxcywh", out_fmt="xyxy"),
            box_convert(target_boxes, in_fmt="cxcywh", out_fmt="xyxy"),
        )
    )
    giou_loss = giou_loss.sum() / num_boxes
    return BoxLosses(l1_loss=l1_loss, giou_loss=giou_loss)
