# Copyright (c) Facebook, Inc. and its affiliates.
import math
import warnings
from typing import Any, Callable, Optional, Union

import torch
from torch import nn, Tensor

from torchmultimodal.modules.losses.contrastive_loss_with_temperature import (
    contrastive_loss_with_temperature,
)


# TODO(asg): Replace later with MLP classifier if checkpoint permits
class Pooler(nn.Module):
    def __init__(self, hidden_size: int = 756, **kwargs: Any):
        super().__init__()

        self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


# TODO(asg): Simplify with existing checkpoints later
class TwoWayHead(nn.Module):
    def __init__(self, hidden_size: int = 756, **kwargs: Any):
        super().__init__()

        self.seq_relationship = nn.Linear(hidden_size, 2)

    def forward(self, pooled_output):
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
    ):
        pooled_output = self.pooler(hidden_states)
        scores = self.cls(pooled_output)

        return self.ce_loss(
            scores.view(-1, 2),
            labels.view(-1),
        )


class MaskedPredictionHead(nn.Module):
    def __init__(
        self,
        hidden_size: int = 756,
        vocab_size: int = 30522,
        transform_act_fn: Callable[..., Tensor] = nn.functional.gelu,
        layer_norm_eps: float = 1e-5,
        **kwargs: Any,
    ):
        super().__init__()

        self.dense = nn.Linear(hidden_size, hidden_size)
        self.transform_act_fn = transform_act_fn
        self.layer_norm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = nn.Linear(hidden_size, vocab_size, bias=False)

        self.bias = nn.Parameter(torch.zeros(vocab_size))

        # Need a link between the two variables so that the bias is
        # correctly resized with `resize_token_embeddings`
        self.decoder.bias = self.bias

    def forward(self, hidden_states: Tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.layer_norm(hidden_states)
        hidden_states = self.decoder(hidden_states)
        return hidden_states


class MaskedPredictionLoss(nn.Module):
    def __init__(
        self,
        hidden_size: int = 756,
        vocab_size: int = 30522,
        transform_act_fn: Callable[..., Tensor] = nn.functional.gelu,
        layer_norm_eps: float = 1e-5,
        ignore_index: int = -1,
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

    def forward(self, hidden_states: Tensor, masked_labels: Tensor):
        masked_tokens = masked_labels.ne(self.ignore_index)

        masked_labels = masked_labels[masked_tokens]
        sequence_output = hidden_states[masked_tokens, :]

        prediction = self.cls(sequence_output)
        masked_loss = self.ce_loss(
            prediction.view(-1, self.vocab_size),
            masked_labels.view(-1),
        )
        # When masked_labels are all ignore_index then masked_lm_loss is NaN,
        # so we replace NaN with 0.
        if torch.isnan(masked_loss):
            warnings.warn("NaN detected in masked_loss. Replacing it with 0.")
            masked_loss = torch.nan_to_num(masked_loss, nan=0.0)
        return masked_loss


class FLAVAGlobalContrastiveLoss(nn.Module):
    def __init__(
        self,
        logit_scale: Union[float, nn.Parameter] = None,
        image_embedding_size: int = 768,
        text_embedding_size: int = 768,
        projection_size: int = 768,
        image_embedding_index: int = 0,
        text_embedding_index: int = 0,
        **kwargs,
    ):
        super().__init__()
        if logit_scale is None:
            logit_scale = math.log(1 / 0.07)

        # If already initialized, set to what was passed
        if isinstance(logit_scale, nn.Parameter):
            self.logit_scale = logit_scale
        else:
            self.logit_scale = nn.Parameter(logit_scale * torch.ones([]))

        self.image_projection = nn.Linear(image_embedding_size, projection_size)
        self.text_projection = nn.Linear(text_embedding_size, projection_size)
        self.image_embedding_index = image_embedding_index
        self.text_embedding_index = text_embedding_index

    def forward(
        self,
        image_sequence: Tensor,
        text_sequence: Tensor,
        mask: Tensor,
    ):
        text_embedding = nn.functional.normalize(
            self.text_projection(text_sequence[:, self.text_embedding_index, :]), dim=-1
        )
        image_embedding = nn.functional.normalize(
            self.image_projection(image_sequence[:, self.image_embedding_index, :]),
            dim=-1,
        )

        self.logit_scale.data.clamp_(0, 4.6052)

        return contrastive_loss(
            image_embeddings=image_embedding,
            text_embeddings=text_embedding,
            logit_scale=self.logit_scale,
            mask=mask,
            # Always true for FLAVA global contrastive loss
            backprop_in_gather=True,
        )


class FLAVAPretrainingLoss(nn.Module):
    def __init__(
        self,
        logit_scale: Union[float, nn.Parameter] = None,
        hidden_size: int = 768,
        text_vocab_size: int = 30522,
        image_vocab_size: int = 8192,
        transform_act_fn: Callable[..., Tensor] = nn.functional.gelu,
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
        self.itm = ITMLoss(
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
    ):
        # TODO(asg): Add proper checks and only calculate losses which can
        # be calculated
        outputs = {}
        pos_mask = None

        # Check multimodal_masked_sequence to make sure this is unimodal case
        # This specific case can though be backpropagated directly as MIM is independent of
        # text, but that is a research question :)
        if (
            mim_labels is not None
            and self.mim_weight > 0
            and multimodal_masked_sequence is None
        ):
            assert image_masked_sequence is not None, (
                "image_masked_sequence must be passed with "
                + "mim_labels when it is not a multimodal case"
            )
            # Remove CLS token from image_masked_sequence
            outputs["mim_loss"] = self.mim_loss(
                image_masked_sequence[:, -mim_labels.size(1) :, :], mim_labels
            )
            outputs["mim_loss"] *= self.mim_weight

        # Check multimodal_masked_sequence to make sure this is unimodal case
        if (
            mlm_labels is not None
            and self.mlm_weight > 0
            and multimodal_masked_sequence is None
        ):
            assert text_masked_sequence is not None, (
                "text_masked_sequence must be passed with "
                + "mlm_labels when it is not a multimodal case"
            )
            outputs["mlm_loss"] = self.mlm_loss(
                text_masked_sequence[:, -mlm_labels.size(1) :, :], mlm_labels
            )
            outputs["mlm_loss"] *= self.mlm_weight

        if (
            multimodal_sequence is not None
            and itm_labels is not None
            and self.itm_loss_weight > 0
        ):
            pos_pairs = itm_labels.ne(0)
            pos_mask = torch.where(pos_pairs.any(), pos_pairs, pos_pairs.new([True]))
            itm_loss = self.itm(multimodal_sequence, itm_labels)
            outputs["itm_loss"] = self.itm_loss_weight * itm_loss
            multimodal_sequence = multimodal_sequence[pos_mask]
            if multimodal_masked_sequence is not None:
                multimodal_masked_sequence = multimodal_masked_sequence[pos_mask]
            if mlm_labels is not None:
                mlm_labels = mlm_labels[pos_mask]
            if mim_labels is not None:
                mim_labels = mim_labels[pos_mask]

        if multimodal_masked_sequence is not None and self.mmm_text_loss_weight > 0:
            assert mlm_labels is not None, "mlm_labels must be passed for mmm_text_loss"
            sequence_for_text = multimodal_masked_sequence[:, -mlm_labels.size(1) :, :]
            outputs["mmm/text_loss"] = self.mlm_loss(
                sequence_for_text,
                mlm_labels,
            )
            outputs["mmm/text_loss"] *= self.mmm_text_loss_weight

        if multimodal_masked_sequence is not None and self.mmm_image_loss_weight > 0:
            assert (
                mim_labels is not None
            ), "mim_labels must be passed for mmm_image_loss"
            # Starts from 2 because of 2 CLS, one for multimodal encoder and one
            # that comes from image encoder.
            sequence_for_image = multimodal_masked_sequence[
                :, 2 : 2 + mim_labels.size(1), :
            ]
            outputs["mmm/image_loss"] = self.mim_loss(
                sequence_for_image,
                mim_labels,
            )
            outputs["mmm/image_loss"] *= self.mmm_image_loss_weight

        if (
            image_sequence is not None
            and text_sequence is not None
            and self.contrastive_loss_weight > 0
        ):
            outputs["global_contrastive_loss"] = self.contrastive_loss(
                image_sequence,
                text_sequence,
                pos_mask,
            )
            outputs["global_contrastive_loss"] *= self.contrastive_loss_weight

        return outputs
