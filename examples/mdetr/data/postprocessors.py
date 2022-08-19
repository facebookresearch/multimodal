# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import List

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torchvision.ops.boxes import box_convert


class PostProcessFlickr:
    """This module converts the model's output for Flickr30k entities evaluation.

    This processor is intended for recall@k evaluation with respect to each phrase
    in the sentence. It requires a description of each phrase (as a binary mask),
    and returns a sorted list of boxes for each phrase. Based on MDETR repo:
    https://github.com/ashkamath/mdetr/blob/main/models/postprocessors.py#L13.


    Inputs: outputs (Dict[str, Tensor]): raw outputs of the model. Should contain keys
                pred_logits and pred_boxes.
            target_sizes (Tensor) Size of each image in the batch. For evaluation, this
                must be the original image size (before any data augmentation).
                Size: (2, batch_size)
            positive_map (Tensor): For each phrase in the batch, contains a binary mask
                of the tokens that correspond to that sentence. Note that this is a
                "collapsed" batch, meaning that all the phrases of all the batch
                elements are stored sequentially.
                Size: (total_num_phrases, max_seq_len)
            phrases_per_sample (List[int]): Number of phrases corresponding
                to each batch element.

    Returns: a List[List[List[float]]]: List of bounding box coordinates for each
                phrase in each sample sorted by probabilities.


    """

    def __call__(
        self,
        output_logits: Tensor,
        output_bbox: Tensor,
        target_sizes: Tensor,
        positive_map: Tensor,
        phrases_per_sample: List[int],
    ) -> List[List[List[float]]]:

        assert output_logits.size(0) == target_sizes.size(
            0
        ), "Logits and target sizes should both have first dim = batch_size"
        assert target_sizes.size(1) == 2, "Target sizes should have second dim = 2"

        batch_size = target_sizes.shape[0]
        prob = F.softmax(output_logits, -1)

        # convert to [x0, y0, x1, y1] format
        boxes = box_convert(output_bbox, in_fmt="cxcywh", out_fmt="xyxy")
        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        # and from relative [0, 1] to absolute [0, height] coordinates
        boxes = boxes * scale_fct[:, None, :]
        cum_sum = np.cumsum(phrases_per_sample)

        curr_batch_index = 0
        # binarize the map if not already binary
        pos = positive_map > 1e-6

        predicted_boxes: List[List[List[float]]] = [[] for _ in range(batch_size)]

        # The collapsed batch dimension must match the number of items
        assert (
            pos.size(0) == cum_sum[-1]
        ), "First dimension of positive map must equal sum of phrases per sample"

        if len(pos) == 0:
            return predicted_boxes

        # if the first batch elements don't contain elements, skip them.
        while cum_sum[curr_batch_index] == 0:
            curr_batch_index += 1

        for i in range(len(pos)):
            # scores are computed by taking the max over the scores assigned to the positive tokens
            scores, _ = torch.max(
                pos[i].unsqueeze(0) * prob[curr_batch_index, :, :], dim=-1
            )
            _, indices = torch.sort(scores, descending=True)

            assert (
                phrases_per_sample[curr_batch_index] > 0
            ), "Each sample must have at least one phrase"
            predicted_boxes[curr_batch_index].append(
                boxes[curr_batch_index][indices].to("cpu").tolist()
            )
            if i == len(pos) - 1:
                break

            # check if we need to move to the next batch element
            while i >= cum_sum[curr_batch_index] - 1:
                curr_batch_index += 1
                assert curr_batch_index < len(
                    cum_sum
                ), "Current batch index is not less than total number of phrases"

        return predicted_boxes
