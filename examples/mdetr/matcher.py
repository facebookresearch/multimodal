# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Tuple

import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torch import nn, Tensor
from torchvision.ops.boxes import box_convert, generalized_box_iou


class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network.

    For efficiency reasons, the targets don't include the no_object. Because of this,
    in general, there are more predictions than targets. In this case, we do a 1-to-1
    matching of the best predictions, while the others are un-matched (and thus treated
    as non-objects). This implementation is based on the MDETR repo:
    https://github.com/ashkamath/mdetr/blob/main/models/matcher.py#L13

    Attributes:
        cost_class (float): Relative weight of the classification error in the
            matching cost. Default: ``1``
        cost_bbox (float): Relative weight of the L1 error of the bounding box
            coordinates in the matching cost. Default: ``1``
        cost_giou (float): Relative weight of the giou loss of the bounding box in
            the matching cost. Default: ``1``


    Args:
        pred_logits (Tensor): Classification logits.
            Size: (batch_size, num_queries, num_classes)
        pred_boxes (Tensor): Predicted box coordinates.
            Size: (batch_size, num_queries, 4)
        target_boxes_per_sample (List[Tensor]): A list of target bounding boxes.
            Length = batch_size.
            Each element is a tensor of size (n_boxes_for_sample, 4).
        positive_map (Tensor): :math:`\text{positive_map}[i,j] = 1` when box i maps to class j.
            Size: (total_boxes, num_classes) where total_boxes is the sum of
            n_boxes_for_sample over every sample in the batch.

    Returns:
        A list of size batch_size, containing tuples of ``(index_i, index_j)`` where:
            - ``index_i`` is the indices of the selected predictions (in order)
            - ``index_j`` is the indices of the corresponding selected targets
        For each batch element, it holds:
            len(index_i) = len(index_j) = min(num_queries, num_target_boxes)

    Raises:
        ValueError: If all costs are zero or first dim of target boxes and positive map
            don't match or classification cost and bbox cost shapes don't match.
    """

    def __init__(
        self, cost_class: float = 1, cost_bbox: float = 5, cost_giou: float = 2
    ):
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        if cost_class == 0 and cost_bbox == 0 and cost_giou == 0:
            raise ValueError("At least one cost must be nonzero")

    @torch.no_grad()
    def forward(
        self,
        pred_logits: Tensor,
        pred_boxes: Tensor,
        target_boxes_per_sample: List[Tensor],
        positive_map: Tensor,
    ) -> List[Tuple[Tensor, Tensor]]:
        bs, num_queries = pred_logits.shape[:2]
        target_boxes = torch.cat(target_boxes_per_sample)
        # We flatten to compute the cost matrices in a batch
        out_prob = F.softmax(
            pred_logits.flatten(0, 1), dim=-1
        )  # [batch_size * num_queries, num_classes]
        out_bbox = pred_boxes.flatten(0, 1)  # [batch_size * num_queries, 4]
        if target_boxes.size(0) != positive_map.size(0):
            raise ValueError(
                "Total of target boxes should match first dim of positive map"
            )

        # Compute the soft-cross entropy between the predicted token alignment
        # and the ground truth one for each box
        cost_class = -(out_prob.unsqueeze(1) * positive_map.unsqueeze(0)).sum(-1)

        # Compute the L1 cost between boxes
        cost_bbox = torch.cdist(out_bbox, target_boxes, p=1)
        if cost_class.shape != cost_bbox.shape:
            raise ValueError(
                f"""
            Classification and bounding box cost shapes do not match.
            Classification cost shape: {cost_class.shape},
            Bounding box cost shape: {cost_bbox.shape}
            """
            )

        # Compute the giou cost betwen boxes
        cost_giou = -generalized_box_iou(
            box_convert(out_bbox, in_fmt="cxcywh", out_fmt="xyxy"),
            box_convert(target_boxes, in_fmt="cxcywh", out_fmt="xyxy"),
        )

        # Final cost matrix
        cost_matrix = (
            self.cost_bbox * cost_bbox
            + self.cost_class * cost_class
            + self.cost_giou * cost_giou
        )
        cost_matrix = cost_matrix.view(bs, num_queries, -1).cpu()

        sizes = [x.size(0) for x in target_boxes_per_sample]
        indices = [
            linear_sum_assignment(c[i])
            for i, c in enumerate(cost_matrix.split(sizes, -1))
        ]
        return [
            (
                torch.as_tensor(i, dtype=torch.int64),
                torch.as_tensor(j, dtype=torch.int64),
            )
            for i, j in indices
        ]
