# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn, Tensor


class ImageTextContrastiveLoss(nn.Module):
    """
    Compute the image-text contrastive loss from image-text similarity, as used in ALBEF.
    Support loss distillation with pseudo-targets for non-zero alpha. Compute standard contrastive loss for zero alpha.

    Inputs:
        image_to_text_sim (Tensor): Image to text similarity.
        text_to_image_sim (Tensor): Text to image similarity.
        image_to_text_sim_m (Optional[Tensor]): Image to text similarity from momentum models.
            Required if alpha is non-zero.
        text_to_image_sim_m (Optional[Tensor]): Text to image similarity from momentum models.
            Required if alpha is non-zero.
        sim_targets (Optional[Tensor]): Similarity pseudo-targets from momentum models. Default is the diagonal matrix.
            Requires all Tensor inputs to have the same size.
        alpha (Optional[float]): The interpolation value of momentum similarity and sim_targets. Default is 0.
    """

    def __init__(
        self,
    ) -> None:
        super().__init__()

    def forward(
        self,
        image_to_text_sim: Tensor,
        text_to_image_sim: Tensor,
        image_to_text_sim_m: Optional[Tensor] = None,
        text_to_image_sim_m: Optional[Tensor] = None,
        sim_targets: Optional[Tensor] = None,
        alpha: Optional[float] = 0.0,
    ) -> Tensor:
        if sim_targets is None:
            sim_targets = torch.zeros(image_to_text_sim.size()).to(
                image_to_text_sim.device
            )
            sim_targets.fill_diagonal_(1)

        if alpha != 0:
            assert (
                image_to_text_sim_m is not None and text_to_image_sim_m is not None
            ), "sim_i2t_m and sim_t2i_m cannot be none for non-zero alpha"

            with torch.no_grad():
                image_to_text_sim_targets = (
                    alpha * F.softmax(image_to_text_sim_m, dim=1)
                    + (1 - alpha) * sim_targets
                )
                text_to_image_sim_targets = (
                    alpha * F.softmax(text_to_image_sim_m, dim=1)
                    + (1 - alpha) * sim_targets
                )
        else:
            image_to_text_sim_targets = sim_targets
            text_to_image_sim_targets = sim_targets

        loss_i2t = -torch.sum(
            F.log_softmax(image_to_text_sim, dim=1) * image_to_text_sim_targets, dim=1
        ).mean()
        loss_t2i = -torch.sum(
            F.log_softmax(text_to_image_sim, dim=1) * text_to_image_sim_targets, dim=1
        ).mean()

        loss_itc = (loss_i2t + loss_t2i) / 2
        return loss_itc


class CausalLanguageModelingLoss(nn.Module):
    """
    Compute the autoregressive masked language modeling loss by predicting the next token, as used in VQA.
    Support loss distillation for non-zero alpha. Compute standard mlm loss for zero alpha.

    Args:
        mask_token_id (int): The token id indicating a masked token. Default is -100.

    Inputs:
        labels (Tensor of shape (batch_size, seq_length)): The masked output tokens.
        prediction_scores (Tensor of shape (batch_size, seq_length, vocab_size)):
            The prediction scores from a prediction head.
        prediction_scores_m (Optional[Tensor] of shape (batch_size, seq_length, vocab_size)):
            The prediction scores from a momentum prediction head.
            Required if alpha is non-zero.
        alpha (float): The interpolation value between mlm_loss and loss_distill. Default is 0.
    """

    def __init__(
        self,
        mask_token_id: int = -100,
    ) -> None:
        super().__init__()
        self.mask_token_id = mask_token_id

    def forward(
        self,
        labels: Tensor,
        prediction_scores: Tensor,
        prediction_scores_m: Optional[Tensor] = None,
        alpha: Optional[float] = 0.0,
    ) -> Tensor:
        batch_size = labels.size(0)
        # shift prediction scores and labels by one for next-token predict
        prediction_scores = prediction_scores[:, :-1, :].contiguous()
        labels = labels[:, 1:].contiguous()
        mlm_loss = F.cross_entropy(
            prediction_scores.view(-1, prediction_scores.shape[-1]),
            labels.view(-1),
            reduction="none",
        )
        mlm_loss = mlm_loss.view(batch_size, -1).sum(1)

        if alpha != 0:
            assert (
                prediction_scores_m is not None
            ), "prediction_scores_m cannot be None for non-zero alpha"

            with torch.no_grad():
                prediction_scores_m = prediction_scores_m[:, :-1, :].contiguous()
            loss_distill = -torch.sum(
                F.log_softmax(prediction_scores, dim=-1)
                * F.softmax(prediction_scores_m, dim=-1),
                dim=-1,
            )
            loss_distill = (loss_distill * (labels != self.mask_token_id)).sum(1)
            mlm_loss = (1 - alpha) * mlm_loss + alpha * loss_distill

        return mlm_loss
