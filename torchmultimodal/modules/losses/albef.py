# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Callable, Optional

import torch
import torch.nn.functional as F
from torch import nn, Tensor


class ImageTextContrastiveLoss(nn.Module):
    """
    Compute the image-text contrastive loss from image-text similarity, as used in ALBEF.
    Support loss distillation with pseudo-targets for non-zero alpha. Compute standard contrastive loss for zero alpha.

    Args:
        alpha (float): The interpolation value of momentum similarity and sim_targets. Default is 0.

    Inputs:
        image_to_text_sim (Tensor): Image to text similarity.
        text_to_image_sim (Tensor): Text to image similarity.
        image_to_text_sim_m (Optional[Tensor]): Image to text similarity from momentum models.
            Required if alpha is non-zero.
        text_to_image_sim_m (Optional[Tensor]): Text to image similarity from momentum models.
            Required if alpha is non-zero.
        sim_targets (Optional[Tensor]): Similarity pseudo-targets from momentum models. Default is the diagonal matrix.
            Requires all inputs to have the same size.
    """

    def __init__(
        self,
        alpha: float = 0.0,
    ) -> None:
        super().__init__()
        self.alpha = alpha

    def forward(
        self,
        image_to_text_sim: Tensor,
        text_to_image_sim: Tensor,
        image_to_text_sim_m: Optional[Tensor] = None,
        text_to_image_sim_m: Optional[Tensor] = None,
        sim_targets: Optional[Tensor] = None,
    ) -> Tensor:
        if sim_targets is None:
            sim_targets = torch.zeros(image_to_text_sim.size()).to(
                image_to_text_sim.device
            )
            sim_targets.fill_diagonal_(1)

        if self.alpha != 0:
            assert (
                image_to_text_sim_m is not None and text_to_image_sim_m is not None
            ), "sim_i2t_m and sim_t2i_m cannot be none for non-zero alpha"

            with torch.no_grad():
                image_to_text_sim_targets = (
                    self.alpha * F.softmax(image_to_text_sim_m, dim=1)
                    + (1 - self.alpha) * sim_targets
                )
                text_to_image_sim_targets = (
                    self.alpha * F.softmax(text_to_image_sim_m, dim=1)
                    + (1 - self.alpha) * sim_targets
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


class ImageTextMatchingLoss(nn.Module):
    """
    Compute the image-text matching loss by predicting whether an image-text pair is matched or not.

    Args:
        hidden_size (int): The image-text multimodal embedding hidden size. Default is 768.

    Inputs:
        embeddings_pos (Tensor of shape (batch_size_pos, hidden_size)):
            The multimodal embeddings for positive image-text pairs.
        embeddings_neg (Tensor of shape (batch_size_neg, hidden_size)):
            The multimodal embeddings for negative image-text pairs.
    """

    def __init__(
        self,
        hidden_size: int = 768,
    ) -> None:
        super().__init__()
        self.itm_head = nn.Linear(
            hidden_size, 2
        )  # binary output indicating image-text matches

    def forward(
        self,
        embeddings_pos: Tensor,
        embeddings_neg: Tensor,
    ) -> Tensor:
        vl_embeddings = torch.cat([embeddings_pos, embeddings_neg], dim=0)
        vl_output = self.itm_head(vl_embeddings)
        itm_labels = torch.cat(
            [
                torch.ones(embeddings_pos.size(0), dtype=torch.long),
                torch.zeros(embeddings_neg.size(0), dtype=torch.long),
            ],
            dim=0,
        ).to(vl_embeddings.device)
        loss_itm = F.cross_entropy(vl_output, itm_labels)
        return loss_itm


class MaskedLanguageModelingLoss(nn.Module):
    """
    Compute the autoregressive masked language modeling loss by predicting the next token, as used in VQA.
    Support loss distillation for non-zero alpha. Compute standard mlm loss for zero alpha.

    Args:
        masked_token_id (int): The token id indicating a masked token. Default is -100.
        vocab_size (int): The number of different tokens the prediction_head can predict. Default is 30522.
        hidden_size (int): The hidden size of the prediction_head. Default is 768.
        layer_norm_eps (float): The epsilon used by the prediction_head normalization layer. Default is 1e-12.
        alpha (float): The interpolation value between mlm_loss and loss_distill. Default is 0.
        transform_act_fn (Callable[[Tensor], Tensor]): The activation function in the prediction_head. Default is GELU.

    Inputs:
        labels (Tensor of shape (batch_size, seq_length)): The masked output tokens.
        hidden_states (Tensor of shape (batch_size, seq_length, hidden_size)):
            The hidden states of preceding tokens.
        hidden_states_m (Optional[Tensor] of shape (batch_size, seq_length, hidden_size)):
            The hidden states of preceding tokens from momentum models.
            Required if alpha is non-zero.
    """

    def __init__(
        self,
        mask_token_id: int = -100,
        vocab_size: int = 30522,
        hidden_size: int = 768,
        layer_norm_eps: float = 1e-12,
        alpha: float = 0.0,
        transform_act_fn: Callable[[Tensor], Tensor] = nn.functional.gelu,
    ) -> None:
        super().__init__()
        self.mask_token_id = mask_token_id
        self.vocab_size = vocab_size
        self.alpha = alpha
        self.prediction_head = PredictionHead(
            vocab_size, hidden_size, layer_norm_eps, transform_act_fn
        )

    def forward(
        self,
        labels: Tensor,
        hidden_states: Tensor,
        hidden_states_m: Optional[Tensor] = None,
    ) -> Tensor:
        batch_size = labels.size(0)
        prediction_scores = self.prediction_head(hidden_states)
        # shift prediction scores and labels by one for next-token prediction
        prediction_scores = prediction_scores[:, :-1, :].contiguous()
        labels = labels[:, 1:].contiguous()
        mlm_loss = F.cross_entropy(
            prediction_scores.view(-1, self.vocab_size),
            labels.view(-1),
            reduction="none",
        )
        mlm_loss = mlm_loss.view(batch_size, -1).sum(1)

        if self.alpha != 0:
            assert (
                hidden_states_m is not None
            ), "hidden_states_m cannot be None for non-zero alpha"

            with torch.no_grad():
                prediction_scores_m = self.prediction_head(hidden_states_m)
                prediction_scores_m = prediction_scores_m[:, :-1, :].contiguous()
            loss_distill = -torch.sum(
                F.log_softmax(prediction_scores, dim=-1)
                * F.softmax(prediction_scores_m, dim=-1),
                dim=-1,
            )
            loss_distill = (loss_distill * (labels != self.mask_token_id)).sum(1)
            mlm_loss = (1 - self.alpha) * mlm_loss + self.alpha * loss_distill

        return mlm_loss


class PredictionHead(nn.Module):
    """
    Predict the following token autoregressively.

    Args:
        vocab_size (int): The number of different tokens the prediction_head can predict.
        hidden_size (int): The hidden size of the prediction_head.
        layer_norm_eps (float): The epsilon used by the prediction_head normalization layer.
        transform_act_fn (Callable[[Tensor], Tensor]): The activation function in the prediction_head.

    Inputs:
        hidden_states (Tensor): The hidden states of preceding tokens.
    """

    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        layer_norm_eps: float,
        transform_act_fn: Callable[[Tensor], Tensor],
    ) -> None:
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.transform_act_fn = transform_act_fn
        self.layer_norm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.decoder = nn.Linear(hidden_size, vocab_size)

    def forward(self, hidden_states: Tensor) -> Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.layer_norm(hidden_states)
        hidden_states = self.decoder(hidden_states)
        return hidden_states
