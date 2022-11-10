# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import math
import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, OrderedDict, Union

import torch
from torch import nn, Tensor
from torchmultimodal.modules.layers.normalizations import Fp32LayerNorm
from torchmultimodal.modules.losses.contrastive_loss_with_temperature import (
    contrastive_loss_with_temperature,
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
class FLAVAGlobalContrastiveLossOutput(OrderedDict):
    text_embedding: Tensor
    image_embedding: Tensor
    logit_scale: Tensor
    image_logits: Tensor
    text_logits: Tensor
    image_loss: Tensor
    text_loss: Tensor
    loss: Tensor


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


class TwoWayHead(nn.Module):
    def __init__(self, hidden_size: int = 768, **kwargs: Any):
        super().__init__()

        self.seq_relationship = nn.Linear(hidden_size, 2)

    def forward(self, pooled_output: Tensor) -> Tensor:
        return self.seq_relationship(pooled_output)


class ITMLoss(nn.Module):
    def __init__(
        self,
        hidden_size: int = 768,
        ignore_index: int = -1,
        **kwargs: Any,
    ):
        super().__init__()
        self.pooler = Pooler(hidden_size=hidden_size)
        self.cls = TwoWayHead(hidden_size=hidden_size)
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=ignore_index)

    def forward(
        self,
        hidden_states: Tensor,
        labels: Tensor,
    ) -> ITMLossOutput:
        if self.training:
            assert_labels_are_present(labels, "itm labels")

        pooled_output = self.pooler(hidden_states)
        scores = self.cls(pooled_output)

        if labels is None:
            loss = pooled_output.sum() * 0
        else:
            loss = self.ce_loss(
                scores.view(-1, 2),
                labels.view(-1),
            )
        return ITMLossOutput(logits=scores, loss=loss)


class MaskedPredictionHead(nn.Module):
    def __init__(
        self,
        hidden_size: int = 768,
        vocab_size: int = 30522,
        transform_act_fn: Callable[[Tensor], Tensor] = nn.functional.gelu,
        layer_norm_eps: float = 1e-5,
        use_fp32_layer_norm: bool = True,
        **kwargs: Any,
    ):
        super().__init__()

        self.dense = nn.Linear(hidden_size, hidden_size)
        self.transform_act_fn = transform_act_fn

        self.layer_norm: nn.LayerNorm
        if use_fp32_layer_norm:
            self.layer_norm = Fp32LayerNorm(hidden_size, eps=layer_norm_eps)
        else:
            self.layer_norm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = nn.Linear(hidden_size, vocab_size, bias=False)

        self.bias = nn.Parameter(torch.zeros(vocab_size))

        # Need a link between the two variables so that the bias is
        # correctly resized with `resize_token_embeddings`
        self.decoder.bias = self.bias

    def forward(self, hidden_states: Tensor) -> Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.layer_norm(hidden_states)
        hidden_states = self.decoder(hidden_states)
        return hidden_states


class MaskedPredictionLoss(nn.Module):
    def __init__(
        self,
        hidden_size: int = 768,
        vocab_size: int = 30522,
        transform_act_fn: Callable[[Tensor], Tensor] = nn.functional.gelu,
        layer_norm_eps: float = 1e-5,
        ignore_index: int = -1,
        ignore_nan: bool = False,
        **kwargs: Any,
    ):
        super().__init__()

        self.cls = MaskedPredictionHead(
            hidden_size=hidden_size,
            vocab_size=vocab_size,
            transform_act_fn=transform_act_fn,
            layer_norm_eps=layer_norm_eps,
        )
        self.ignore_index = ignore_index
        self.vocab_size = vocab_size
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=ignore_index)
        self.ignore_nan = ignore_nan

    def forward(
        self, hidden_states: Tensor, masked_labels: Optional[Tensor] = None
    ) -> MaskedPredictionLossOutput:
        if self.training:
            assert_labels_are_present(masked_labels, "masked labels")

        if masked_labels is not None:
            masked_tokens = masked_labels.ne(self.ignore_index)
            masked_labels = masked_labels[masked_tokens]
            sequence_output = hidden_states[masked_tokens, :]
        else:
            sequence_output = hidden_states

        prediction = self.cls(sequence_output)

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
        image_embedding_size: int = 768,
        text_embedding_size: int = 768,
        projection_size: int = 768,
        image_embedding_index: int = 0,
        text_embedding_index: int = 0,
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
            embeddings_a=image_embedding,
            embeddings_b=text_embedding,
            logit_scale=self.logit_scale,
            mask=mask,
            # Always true for FLAVA global contrastive loss
            backprop_in_gather=True,
        )

        return FLAVAGlobalContrastiveLossOutput(
            loss=output.loss,
            image_logits=output.logits_a,
            text_logits=output.logits_b,
            image_loss=output.loss_a,
            text_loss=output.loss_b,
            text_embedding=text_embedding,
            image_embedding=image_embedding,
            logit_scale=self.logit_scale.data,
        )


class FLAVAPretrainingLoss(nn.Module):
    def __init__(
        self,
        logit_scale: Union[float, nn.Parameter] = None,
        hidden_size: int = 768,
        text_vocab_size: int = 30522,
        image_vocab_size: int = 8192,
        transform_act_fn: Callable[[Tensor], Tensor] = nn.functional.gelu,
        layer_norm_eps: float = 1e-5,
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

        self.contrastive_loss = FLAVAGlobalContrastiveLoss(
            logit_scale=logit_scale,
            image_embedding_size=hidden_size,
            text_embedding_size=hidden_size,
            projection_size=hidden_size,
        )
        self.mlm_loss = MaskedPredictionLoss(
            hidden_size=hidden_size,
            vocab_size=text_vocab_size,
            transform_act_fn=transform_act_fn,
            layer_norm_eps=layer_norm_eps,
            ignore_index=ignore_index,
        )
        self.mim_loss = MaskedPredictionLoss(
            hidden_size=hidden_size,
            vocab_size=image_vocab_size,
            transform_act_fn=transform_act_fn,
            layer_norm_eps=layer_norm_eps,
            ignore_index=ignore_index,
        )
        # Create separate weights for MMM loss
        self.mmm_loss = nn.ModuleDict(
            {
                "mlm": MaskedPredictionLoss(
                    hidden_size=hidden_size,
                    vocab_size=text_vocab_size,
                    transform_act_fn=transform_act_fn,
                    layer_norm_eps=layer_norm_eps,
                    ignore_index=ignore_index,
                ),
                "mim": MaskedPredictionLoss(
                    hidden_size=hidden_size,
                    vocab_size=image_vocab_size,
                    transform_act_fn=transform_act_fn,
                    layer_norm_eps=layer_norm_eps,
                    ignore_index=ignore_index,
                ),
            }
        )
        self.itm_loss = ITMLoss(
            hidden_size=hidden_size,
            ignore_index=ignore_index,
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
        image_sequence: Optional[Tensor] = None,
        text_sequence: Optional[Tensor] = None,
        image_masked_sequence: Optional[Tensor] = None,
        text_masked_sequence: Optional[Tensor] = None,
        multimodal_sequence: Optional[Tensor] = None,
        multimodal_masked_sequence: Optional[Tensor] = None,
        itm_labels: Optional[Tensor] = None,
        mim_labels: Optional[Tensor] = None,
        mlm_labels: Optional[Tensor] = None,
        projected_image_embeddings: Optional[Tensor] = None,
        projected_text_embeddings: Optional[Tensor] = None,
    ) -> FLAVAPretrainingLossOutput:
        outputs = FLAVAPretrainingLossOutput()
        pos_mask = None

        # Check multimodal_masked_sequence to make sure this is unimodal case
        # This specific case can though be backpropagated directly as MIM is independent of
        # text, but that is a research question :)

        if (
            image_masked_sequence is not None
            and self.mim_weight > 0
            and multimodal_masked_sequence is None
        ):
            # Remove CLS token from image_masked_sequence

            start_index = -mim_labels.size(1) if mim_labels is not None else 1
            outputs.mim_output = self.mim_loss(
                image_masked_sequence[:, start_index:, :], mim_labels
            )
            outputs.mim_output.loss *= self.mim_weight
            outputs.losses.mim_loss = outputs.mim_output.loss

        # Check multimodal_masked_sequence to make sure this is unimodal case
        if (
            text_masked_sequence is not None
            and self.mlm_weight > 0
            and multimodal_masked_sequence is None
        ):
            start_index = -mlm_labels.size(1) if mlm_labels is not None else 1
            outputs.mlm_output = self.mlm_loss(
                text_masked_sequence[:, start_index:, :], mlm_labels
            )
            outputs.mlm_output.loss *= self.mlm_weight
            outputs.losses.mlm_loss = outputs.mlm_output.loss

        if multimodal_masked_sequence is not None and self.itm_loss_weight > 0:
            if itm_labels is not None:
                pos_pairs = itm_labels.ne(0)
                pos_mask = torch.where(
                    pos_pairs.any(), pos_pairs, pos_pairs.new([True])
                )
            else:
                pos_mask = torch.ones(
                    multimodal_masked_sequence.size(0),
                    device=multimodal_masked_sequence.device,
                ).bool()
            outputs.itm_output = self.itm_loss(multimodal_masked_sequence, itm_labels)
            outputs.itm_output.loss *= self.itm_loss_weight
            outputs.losses.itm_loss = outputs.itm_output.loss

            multimodal_masked_sequence = multimodal_masked_sequence[pos_mask]
            if mlm_labels is not None:
                mlm_labels = mlm_labels[pos_mask]
            if mim_labels is not None:
                mim_labels = mim_labels[pos_mask]

        if multimodal_masked_sequence is not None and self.mmm_text_loss_weight > 0:
            start_index = (
                -mlm_labels.size(1)
                if mlm_labels is not None
                else -(text_masked_sequence.size(1) - 1)
            )
            sequence_for_text = multimodal_masked_sequence[:, start_index:, :]
            outputs.mmm_text_output = self.mmm_loss.mlm(
                sequence_for_text,
                mlm_labels,
            )  # type: ignore
            outputs.mmm_text_output.loss *= self.mmm_text_loss_weight
            outputs.losses.mmm_text_loss = outputs.mmm_text_output.loss

        if multimodal_masked_sequence is not None and self.mmm_image_loss_weight > 0:
            # Starts from 2 because of 2 CLS, one for multimodal encoder and one
            # that comes from image encoder.
            total_indices = (
                mim_labels.size(1)
                if mlm_labels is not None
                else (image_masked_sequence.size(1) - 1)
            )
            sequence_for_image = multimodal_masked_sequence[:, 2 : 2 + total_indices, :]
            outputs.mmm_image_output = self.mmm_loss.mim(
                sequence_for_image,
                mim_labels,
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
