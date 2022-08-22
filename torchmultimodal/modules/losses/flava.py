# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import math
import warnings
from dataclasses import dataclass, field
from typing import Any, Optional, Union

import torch
from torch import nn, Tensor
from torchmultimodal.modules.losses.contrastive_loss_with_temperature import (
    contrastive_loss_with_temperature,
    ContrastiveLossOutput,
)
from torchmultimodal.utils.common import ModelOutput


def assert_labels_are_present(
    labels: Optional[Tensor], category: str = "labels"
) -> None:
    assert (
        labels is not None
    ), f"Model is in training model but {category} are not passed"


@dataclass
class ITMLossOutput(ModelOutput):
    logits: Tensor
    loss: Tensor


@dataclass
class MaskedPredictionLossOutput(ModelOutput):
    logits: Tensor
    loss: Tensor


@dataclass
class FLAVAGlobalContrastiveLossOutput(ContrastiveLossOutput):
    text_embedding: Tensor
    image_embedding: Tensor
    logit_scale: Tensor


@dataclass
class FLAVAPretrainingLossesCollection(ModelOutput):
    mmm_text_loss: Optional[Tensor] = None
    mmm_image_loss: Optional[Tensor] = None
    mim_loss: Optional[Tensor] = None
    mlm_loss: Optional[Tensor] = None
    itm_loss: Optional[Tensor] = None
    global_contrastive_loss: Optional[Tensor] = None


@dataclass
class FLAVAPretrainingLossOutput(ModelOutput):
    losses: FLAVAPretrainingLossesCollection = field(
        default_factory=FLAVAPretrainingLossesCollection
    )
    mlm_output: Optional[MaskedPredictionLossOutput] = None
    mim_output: Optional[MaskedPredictionLossOutput] = None
    mmm_text_output: Optional[MaskedPredictionLossOutput] = None
    mmm_image_output: Optional[MaskedPredictionLossOutput] = None
    itm_output: Optional[ITMLossOutput] = None
    global_contrastive_output: Optional[FLAVAGlobalContrastiveLossOutput] = None
    image_sequence: Optional[Tensor] = None
    text_sequence: Optional[Tensor] = None
    image_masked_sequence: Optional[Tensor] = None
    text_masked_sequence: Optional[Tensor] = None
    multimodal_sequence: Optional[Tensor] = None
    multimodal_masked_sequence: Optional[Tensor] = None


# TODO(asg): Replace later with MLP classifier if checkpoint permits
class Pooler(nn.Module):
    def __init__(self, hidden_size: int = 768, **kwargs: Any):
        super().__init__()

        self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states: Tensor) -> Tensor:
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class ITMLoss(nn.Module):
    def __init__(
        self,
        ignore_index: int = -1,
        **kwargs: Any,
    ):
        super().__init__()
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=ignore_index)

    def forward(
        self,
        scores: Tensor,
        labels: Tensor,
    ) -> ITMLossOutput:
        if self.training:
            assert_labels_are_present(labels, "itm labels")

        if labels is None:
            loss = scores.sum() * 0
        else:
            loss = self.ce_loss(
                scores.view(-1, 2),
                labels.view(-1),
            )
        return ITMLossOutput(logits=scores, loss=loss)


class MaskedPredictionLoss(nn.Module):
    def __init__(
        self,
        vocab_size: int = 30522,
        ignore_index: int = -1,
        ignore_nan: bool = False,
        **kwargs: Any,
    ):
        super().__init__()
        self.ignore_index = ignore_index
        self.vocab_size = vocab_size
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=ignore_index)
        self.ignore_nan = ignore_nan

    def forward(
        self,
        prediction: Tensor,
        masked_labels: Optional[Tensor] = None,
    ) -> MaskedPredictionLossOutput:

        masked_tokens = masked_labels.ne(self.ignore_index)
        masked_labels = masked_labels[masked_tokens]

        if masked_labels is None:
            masked_loss = prediction.sum() * 0
        else:
            masked_loss = self.ce_loss(
                prediction.view(-1, self.vocab_size),
                masked_labels.view(-1),
            )

        # When masked_labels are all ignore_index then masked_lm_loss is NaN,
        # so we replace NaN with 0.
        if torch.isnan(masked_loss) and self.ignore_nan:
            warnings.warn("NaN detected in masked_loss. Replacing it with 0.")
            masked_loss = torch.nan_to_num(masked_loss, nan=0.0)

        return MaskedPredictionLossOutput(
            logits=prediction,
            loss=masked_loss,
        )


class FLAVAGlobalContrastiveLoss(nn.Module):
    def __init__(
        self,
        logit_scale: Union[float, nn.Parameter] = None,
    ):
        super().__init__()
        if logit_scale is None:
            logit_scale = math.log(1 / 0.07)

        # If already initialized, set to what was passed
        if isinstance(logit_scale, nn.Parameter):
            self.logit_scale = logit_scale
        else:
            self.logit_scale = nn.Parameter(logit_scale * torch.ones([]))

    def forward(
        self,
        image_sequence: Tensor,
        text_sequence: Tensor,
        mask: Tensor,
    ) -> FLAVAGlobalContrastiveLossOutput:

        text_embedding = nn.functional.normalize(text_sequence, dim=-1)
        image_embedding = nn.functional.normalize(
            image_sequence,
            dim=-1,
        )

        self.logit_scale.data.clamp_(0, 4.6052)

        output = contrastive_loss_with_temperature(
            image_embeddings=image_embedding,
            text_embeddings=text_embedding,
            logit_scale=self.logit_scale,
            mask=mask,
            # Always true for FLAVA global contrastive loss
            backprop_in_gather=True,
        )

        return FLAVAGlobalContrastiveLossOutput(
            loss=output.loss,
            image_logits=output.image_logits,
            text_logits=output.text_logits,
            image_loss=output.image_loss,
            text_loss=output.text_loss,
            text_embedding=text_embedding,
            image_embedding=image_embedding,
            logit_scale=self.logit_scale.data,
        )


class FLAVAPretrainingLoss(nn.Module):
    def __init__(
        self,
        logit_scale: Union[float, nn.Parameter] = None,
        text_vocab_size: int = 30522,
        image_vocab_size: int = 8192,
        ignore_index: int = -1,
        mlm_weight: float = 1.0,
        mim_weight: float = 1.0,
        contrastive_loss_weight: float = 1.0,
        mmm_image_loss_weight: float = 1.0,
        mmm_text_loss_weight: float = 1.0,
        itm_loss_weight: float = 1.0,
        **kwargs: Any,
    ):
        super().__init__()
        self.itm_loss = ITMLoss(
            ignore_index=ignore_index,
        )
        self.contrastive_loss = FLAVAGlobalContrastiveLoss(
            logit_scale=logit_scale,
        )
        self.mlm_loss = MaskedPredictionLoss(
            vocab_size=text_vocab_size,
            ignore_index=ignore_index,
        )
        self.mim_loss = MaskedPredictionLoss(
            vocab_size=image_vocab_size,
            ignore_index=ignore_index,
        )
        # Create separate weights for MMM loss
        self.mmm_loss = nn.ModuleDict(
            {
                "mlm": MaskedPredictionLoss(
                    vocab_size=text_vocab_size,
                    ignore_index=ignore_index,
                ),
                "mim": MaskedPredictionLoss(
                    vocab_size=image_vocab_size,
                    ignore_index=ignore_index,
                ),
            }
        )

        self.mim_weight = mim_weight
        self.mlm_weight = mlm_weight
        self.contrastive_loss_weight = contrastive_loss_weight
        self.mmm_image_loss_weight = mmm_image_loss_weight
        self.mmm_text_loss_weight = mmm_text_loss_weight
        self.itm_loss_weight = itm_loss_weight

    # TODO: Some refactoring is needed in this function to make it look better
    # TODO: Possibly refactor this into functional and class component
    # for better usability
    def forward(
        self,
        multimodal_masked_sequence: Optional[Tensor] = None,
        pos_mask: Optional[Tensor] = None,
        itm_labels: Optional[Tensor] = None,
        mim_labels: Optional[Tensor] = None,
        mlm_labels: Optional[Tensor] = None,
        projected_image_embeddings: Optional[Tensor] = None,
        projected_text_embeddings: Optional[Tensor] = None,
        itm_logits: Optional[Tensor] = None,
        mlm_head_output: Optional[Tensor] = None,
        mim_head_output: Optional[Tensor] = None,
        mmm_mlm_head_output: Optional[Tensor] = None,
        mmm_mim_head_output: Optional[Tensor] = None,
    ) -> FLAVAPretrainingLossOutput:
        outputs = FLAVAPretrainingLossOutput()

        # Check multimodal_masked_sequence to make sure this is unimodal case
        # This specific case can though be backpropagated directly as MIM is independent of
        # text, but that is a research question :)

        if (
            mim_head_output is not None
            and self.mim_weight > 0
            and multimodal_masked_sequence is None
        ):
            outputs.mim_output = self.mim_loss(mim_head_output, mim_labels)
            outputs.mim_output.loss *= self.mim_weight
            outputs.losses.mim_loss = outputs.mim_output.loss

        # Check multimodal_masked_sequence to make sure this is unimodal case

        if (
            mlm_head_output is not None
            and self.mlm_weight > 0
            and multimodal_masked_sequence is None
        ):
            outputs.mlm_output = self.mlm_loss(mlm_head_output, mlm_labels)
            outputs.mlm_output.loss *= self.mlm_weight
            outputs.losses.mlm_loss = outputs.mlm_output.loss

        if multimodal_masked_sequence is not None and self.itm_loss_weight > 0:
            assert itm_logits is not None
            outputs.itm_output = self.itm_loss(itm_logits, itm_labels)
            outputs.itm_output.loss *= self.itm_loss_weight
            outputs.losses.itm_loss = outputs.itm_output.loss

        if mmm_mlm_head_output is not None and self.mmm_text_loss_weight > 0:
            outputs.mmm_text_output = self.mmm_loss.mlm(
                mmm_mlm_head_output, mlm_labels
            )  # type: ignore
            outputs.mmm_text_output.loss *= self.mmm_text_loss_weight
            outputs.losses.mmm_text_loss = outputs.mmm_text_output.loss

        if mmm_mim_head_output is not None and self.mmm_image_loss_weight > 0:
            outputs.mmm_image_output = self.mmm_loss.mim(
                mmm_mim_head_output, mim_labels
            )  # type: ignore
            outputs.mmm_image_output.loss *= self.mmm_image_loss_weight
            outputs.losses.mmm_image_loss = outputs.mmm_image_output.loss

        if (
            projected_image_embeddings is not None
            and projected_text_embeddings is not None
            and self.contrastive_loss_weight > 0
        ):
            outputs.global_contrastive_output = self.contrastive_loss(
                projected_image_embeddings,
                projected_text_embeddings,
                pos_mask,
            )
            outputs.global_contrastive_output.loss *= self.contrastive_loss_weight
            outputs.losses.global_contrastive_loss = (
                outputs.global_contrastive_output.loss
            )

        return outputs
