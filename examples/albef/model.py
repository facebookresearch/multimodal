# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import copy
from typing import Callable, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torchmultimodal.models.albef import ALBEFModel, ALBEFModelWithSimilarity
from torchmultimodal.modules.encoders.albef_multimodal_encoder import (
    ALBEFMultimodalEncoder,
)
from torchmultimodal.modules.encoders.albef_text_encoder import (
    ALBEFTextEmbeddings,
    ALBEFTextEncoder,
)
from torchmultimodal.modules.encoders.albef_vision_encoder import ALBEFVisionEncoder
from torchmultimodal.modules.losses.albef import (
    ImageTextContrastiveLoss,
    ImageTextMatchingLoss,
    MaskedLanguageModelingLoss,
)
from torchmultimodal.utils.attention import get_causal_attention_mask
from torchmultimodal.utils.common import (
    load_module_from_url,
    momentum_update,
    remove_grad,
)


_ALBEF_PRETRAINED_URLS = {
    "vqa": "https://download.pytorch.org/models/multimodal/albef/pretrained_vqa_checkpoint.pt",
    "retrieval": "https://download.pytorch.org/models/multimodal/albef/pretrained_retrieval_checkpoint.pt",
}


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

    Returns:
        Tensor: Prediction scores for the following token.
    """

    def __init__(
        self,
        vocab_size: int = 30522,
        hidden_size: int = 768,
        layer_norm_eps: float = 1e-12,
        transform_act_fn: Callable[[Tensor], Tensor] = nn.functional.gelu,
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


class ALBEFDecoder(nn.Module):
    """
    Generate the prediction scores for answers from image and question hidden states.

    Args:
        text_embeddings (ALBEFTextEmbeddings): Instantiated ALBEFTextEmbeddings.
        multimodal_encoder (ALBEFMultimodalEncoder): Instantiated ALBEFMultimodalEncoder.
        prediction_head (PredictionHead): Instantiated PredictionHead.

    Inputs:
        input_ids (Tensor of shape (batch_size, seq_len)):
            Input ids for input text tokens.
        attention_mask (Tensor of shape (batch_size, seq_len)):
            Input attention mask to avoid performing attention on padding token indices.
        encoder_hidden_states (Tensor of shape (batch_size, encoder_seq_len, hidden_size)):
            The encoder hidden states.
        encoder_attention_mask (Tensor of shape (batch_size, encoder_seq_len)):
            The attention mask for encoder hidden states.

    Returns:
        Tensor: Prediction scores for answers.
    """

    def __init__(
        self,
        text_embeddings: ALBEFTextEmbeddings,
        multimodal_encoder: ALBEFMultimodalEncoder,
        prediction_head: PredictionHead,
    ) -> None:
        super().__init__()
        self.text_embeddings = text_embeddings
        self.multimodal_encoder = multimodal_encoder
        self.prediction_head = prediction_head

    def get_extended_attention_mask_for_decoder(self, attention_mask: Tensor) -> Tensor:
        """
        Apply a causal mask in addition to the padding mask and make the mask broadcastable,
        such that future and masked tokens are ignored.

        Args:
            attention_mask (Tensor):
                Padding mask with ones indicating tokens to attend to, zeros for tokens to ignore.

        Returns:
            extended_attention_mask (Tensor):
                The broadcastable attention mask, with the same dtype as ``attention_mask.dtype``.
        """
        device = attention_mask.device
        batch_size, seq_length = attention_mask.shape
        causal_mask = get_causal_attention_mask(seq_length).to(device)
        causal_mask = causal_mask.repeat(batch_size, 1).view(
            batch_size, seq_length, seq_length
        )
        extended_attention_mask = (
            causal_mask[:, None, :, :] * attention_mask[:, None, None, :]
        )
        extended_attention_mask = extended_attention_mask.to(dtype=attention_mask.dtype)
        return extended_attention_mask

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        encoder_hidden_states: Tensor,
        encoder_attention_mask: Tensor,
    ) -> Tensor:
        hidden_states = self.text_embeddings(input_ids)
        attention_mask = self.get_extended_attention_mask_for_decoder(attention_mask)
        decoder_output = self.multimodal_encoder(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
        )
        prediction_scores = self.prediction_head(decoder_output)
        return prediction_scores


class ALBEFModelForVQA(nn.Module):
    """
    ALBEF Model for VQA finetuning and inference.

    Args:
        model (ALBEFModel): Instantiated ALBEFModel.
        answer_decoder (ALBEFDecoder): Instantiated ALBEFDecoder.
        loss (MaskedLanguageModelingLoss): Instantiated MaskedLanguageModelingLoss.

    Inputs:
        image (Tensor of shape (B, C, H, W)): Image features.
        question (Tensor of shape (B, L)): Question text features.
        question_atts (Tensor of shape (B, L)): Question attention mask.
        answers (Tensor of shape (N, M)): Answer text features.
        answers_atts (Tensor of shape (N, M)): Answer attention mask.
        ans_weights (Optional[Tensor] of shape (N)): Weights for each answer.
            Required if is_train is True.
        ans_lengths (Optional[List[int]] of length B): Number of answers for each question.
            ans_lengths should sum to N.
            Required if is_train is True.
        alpha (Optional[float]): The interpolation value between mlm_loss and loss_distill.
            Required if is_train is True.
        k (Optional[int]): The number of answers to return for inference.
            Required if is_train is False.
        is_train (Optional[bool]): Whether the model is in training.

    Returns:
        is_train is True:
            Tensor: The masked language modeling loss for input.
        is_train is False:
            Tuple[Tensor, Tensor]: The ids and probabilities for the top k predicted answers.
    """

    def __init__(
        self,
        model: ALBEFModel,
        answer_decoder: ALBEFDecoder,
        loss: MaskedLanguageModelingLoss,
    ) -> None:
        super().__init__()
        self.model = model
        self.answer_decoder = answer_decoder
        self.loss = loss
        self.answer_decoder_m = copy.deepcopy(self.answer_decoder)
        remove_grad(
            self.answer_decoder_m
        )  # remove gradient for the momentum decoder model

    def _train_forward(
        self,
        image: Tensor,
        question: Tensor,
        question_atts: Tensor,
        answers: Tensor,
        answers_atts: Tensor,
        ans_weights: Tensor,
        ans_lengths: List[int],
        alpha: float,
    ) -> Tensor:
        """
        Forward step for training. Encode the inputs with the ALBEFModel.
        Generate pseudo-targets using answer_decoder_m (momentum decoder model).
        Generate answer predictions using answer_decoder.
        Compute masked language modeling loss of the predictions using answers as labels,
            pseudo-targets as soft-labels, and alpha as their interpolation value.

        Inputs:
            image (Tensor of shape (B, C, H, W)): Image features.
            question (Tensor of shape (B, L)): Question text features.
            question_atts (Tensor of shape (B, L)): Question attention mask.
            answers (Tensor of shape (N, M)): Answer text features.
            answers_atts (Tensor of shape (N, M)): Answer attention mask.
            ans_weights (Tensor of shape (N)): Weights for each answer.
            ans_lengths (List[int] of length B): Number of answers for each question.
                ans_lengths should sum to N.
            alpha (float): The interpolation value between mlm_loss and loss_distill.

        Returns:
            Tensor: The masked language modeling loss for input.
        """
        # get image-question embeddings from the ALBEFModel and format it to match the ans_lengths
        encoder_outputs = self.model(image, question, question_atts)
        (
            encoder_hidden_states,
            encoder_hidden_states_m,
            encoder_attention_mask,
        ) = self._encoder_hidden_states(
            encoder_outputs.multimodal_embeddings,
            encoder_outputs.multimodal_embeddings_m,
            question_atts,
            ans_lengths,
        )

        # use the momentum model to generate pseudo-targets
        with torch.no_grad():
            momentum_update(
                self.answer_decoder, self.answer_decoder_m, self.model.momentum
            )
            prediction_scores_m = self.answer_decoder_m(
                input_ids=answers,
                attention_mask=answers_atts,
                encoder_hidden_states=encoder_hidden_states_m,
                encoder_attention_mask=encoder_attention_mask,
            )

        # generate answer predictions
        prediction_scores = self.answer_decoder(
            input_ids=answers,
            attention_mask=answers_atts,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
        )

        # compute masked language modeling loss from the prediction scores
        labels = answers.masked_fill(answers == 0, self.loss.mask_token_id)
        loss = self.loss(labels, prediction_scores, prediction_scores_m, alpha)
        loss = ans_weights * loss
        loss = loss.sum() / image.size(0)
        return loss

    def _eval_forward(
        self,
        image: Tensor,
        question: Tensor,
        question_atts: Tensor,
        answers: Tensor,
        answer_atts: Tensor,
        k: int = 128,
    ) -> Tuple[Tensor, Tensor]:
        """
        Forward step for evaluation. Encode the inputs with the ALBEFModel.
        Generate answer autoregressively using the decoder, starting with the [CLS] token.
        Compute the answer ids and their perspective probabilities of the top k predictions.

        Inputs:
            image (Tensor of shape (B, C, H, W)): Image features.
            question (Tensor of shape (B, L)): Question text features.
            question_atts (Tensor of shape (B, L)): Question attention mask.
            answers (Tensor of shape (N, M)): Answer text features.
            answer_atts (Tensor of shape (N, M)): Answer attention mask.
            k (int): The number of answers to return for inference.

        Returns:
            Tuple[Tensor, Tensor]: The ids and probabilities for the top k predicted answers.
        """
        # get multimodal embeddings from the ALBEFModel and
        # feed it to the decoder as cross attention
        encoder_outputs = self.model(image, question, question_atts)

        # use cls token as the decoder's initial input token
        num_ques = question.size(0)
        start_ids = answers[0, 0].repeat(num_ques, 1)
        atts = torch.ones(start_ids.shape).to(image.device)

        # auto-regressively generates the answer
        prediction_scores = self.answer_decoder(
            input_ids=start_ids,
            attention_mask=atts,
            encoder_hidden_states=encoder_outputs.multimodal_embeddings,
            encoder_attention_mask=question_atts,
        )

        logits = prediction_scores[:, 0, :]
        answer_first_token = answers[:, 1]
        prob_first_token = F.softmax(logits, dim=1).index_select(
            dim=1, index=answer_first_token
        )
        topk_probs, topk_ids = prob_first_token.topk(k, dim=1)

        input_ids = []
        input_atts = []
        for topk_id in topk_ids:
            input_ids.append(answers.index_select(dim=0, index=topk_id))
            input_atts.append(answer_atts.index_select(dim=0, index=topk_id))
        input_ids = torch.cat(input_ids)
        input_atts = torch.cat(input_atts)
        targets_ids = input_ids.masked_fill(input_ids == 0, self.loss.mask_token_id)

        question_states = encoder_outputs.multimodal_embeddings.repeat_interleave(
            k, dim=0
        )
        question_atts = question_atts.repeat_interleave(k, dim=0)

        prediction_scores = self.answer_decoder(
            input_ids=input_ids,
            attention_mask=input_atts,
            encoder_hidden_states=question_states,
            encoder_attention_mask=question_atts,
        )

        answer_loss = self.loss(targets_ids, prediction_scores)
        answer_loss = answer_loss.view(input_ids.size(0), -1)

        # topk_prob: first token probability
        topk_probs = topk_probs.view(-1, 1)
        log_probs = torch.cat([topk_probs.log(), -answer_loss], dim=1)

        # re-calculate log probabilities for the answer sequences using chain rule
        log_probs_sum = log_probs.sum(1)
        log_probs_sum = log_probs_sum.view(num_ques, k)

        topk_probs = F.softmax(log_probs_sum, dim=-1)

        # get top-k after re-ranking
        topk_probs, rerank_id = topk_probs.topk(k, dim=1)
        topk_ids = torch.gather(topk_ids, 1, rerank_id)

        return topk_ids, topk_probs

    def _encoder_hidden_states(
        self,
        multimodal_embeds: Tensor,
        multimodal_embeds_m: Tensor,
        question_atts: Tensor,
        ans_lengths: List[int],
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Repeat each image-question input, repeat its embedding and mask to match the number of answers it has.

        Args:
            multimodal_embeds (Tensor): Image-question embeddings.
            multimodal_embeds_m (Tensor): Image-question embeddings from the momentum model.
            question_atts (Tensor): Question attention mask.
            ans_lengths (List[int]): The number of answers each image-question input has.

        Returns:
            encoder_hidden_states (Tensor): Image-question embeddings after the repetition.
            encoder_hidden_states_m (Tensor): Image-question embeddings from the momentum model after the repetition.
            encoder_attention_mask (Tensor): Question attention mask after the repetition.
        """
        encoder_hidden_states = []
        encoder_attention_mask = []
        for b, n in enumerate(ans_lengths):
            encoder_hidden_states += [multimodal_embeds[b]] * n
            encoder_attention_mask += [question_atts[b]] * n
        encoder_hidden_states = torch.stack(encoder_hidden_states)
        encoder_attention_mask = torch.stack(encoder_attention_mask)

        with torch.no_grad():
            encoder_hidden_states_m = []
            for b, n in enumerate(ans_lengths):
                encoder_hidden_states_m += [multimodal_embeds_m[b]] * n
            encoder_hidden_states_m = torch.stack(encoder_hidden_states_m)

        return encoder_hidden_states, encoder_hidden_states_m, encoder_attention_mask

    def forward(
        self,
        image: Tensor,
        question: Tensor,
        question_atts: Tensor,
        answers: Tensor,
        answers_atts: Tensor,
        ans_weights: Optional[Tensor] = None,
        ans_lengths: Optional[List[int]] = None,
        alpha: Optional[float] = 0.0,
        k: Optional[int] = 128,
        is_train: Optional[bool] = True,
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        if is_train:
            return self._train_forward(
                image,
                question,
                question_atts,
                answers,
                answers_atts,
                ans_weights,
                ans_lengths,
                alpha,
            )
        else:
            return self._eval_forward(
                image,
                question,
                question_atts,
                answers,
                answers_atts,
                k,
            )


class ALBEFModelForRetrieval(nn.Module):
    """
    ALBEF Model for Retrieval finetuning and inference.

    Args:
        model_with_similarity (ALBEFModelWithSimilarity): Instantiated ALBEFModelWithSimilarity.
        itc_loss (ImageTextContrastiveLoss): Instantiated ImageTextContrastiveLoss.
        itm_loss (ImageTextMatchingLoss): Instantiated ImageTextMatchingLoss.

    Inputs:
        image (Optional[Tensor] of shape (B, C, H, W)): Image features.
            Required if is_train is True.
            Required if input_type is "image" or "multimodal".
        text (Optional[Tensor] of shape (B, L)): Text features.
            Required if is_train is True.
            Required if input_type is "text" or "multimodal".
        text_atts (Tensor of shape (B, L)): Text attention mask.
            Required if is_train is True.
            Required if input_type is "text" or "multimodal".
        idx (Tensor of shape (B)): Identifier for each image sample.
            Required if is_train is True.
        alpha (Optional[float]): The interpolation value between mlm_loss and loss_distill.
            Default is 0.
        input_type (Optional[str]): "image", "text", or "multimodal" indicating the encoding type.
            Required if is_train is False.
        is_train (Optional[bool]): Whether the model is in training.
            Default is True.

    Returns:
        is_train is True:
            Tensor: The sum of itc loss and itm loss.
        is_train is False:
            input_type is "image":
                Tuple[Tensor, Tensor]: Image embeddings and projected image features.
            input_type is "text":
                Tuple[Tensor, Tensor]: Text embeddings and projected text features.
            input_type is "multimodal"
                Tensor: Scores for the retrieval task.
    """

    def __init__(
        self,
        model_with_similarity: ALBEFModelWithSimilarity,
        itc_loss: ImageTextContrastiveLoss,
        itm_loss: ImageTextMatchingLoss,
    ) -> None:
        super().__init__()
        self.model_with_similarity = (
            model_with_similarity  # TODO: rename model to model_with_similarity
        )
        self.itc_loss = itc_loss
        self.itm_loss = itm_loss

    def _train_forward(
        self,
        image: Tensor,
        text: Tensor,
        text_atts: Tensor,
        idx: Tensor,
        alpha: float,
    ) -> Tensor:
        encoder_output = self.model_with_similarity(image, text, text_atts, idx)

        similarity_outputs = encoder_output.similarity
        similarity_targets = encoder_output.sim_targets
        itc_loss = self.itc_loss(
            similarity_outputs.sim_i2t,
            similarity_outputs.sim_t2i,
            similarity_outputs.sim_i2t_m,
            similarity_outputs.sim_t2i_m,
            similarity_targets,
            alpha,
        )

        pos_embeddings = encoder_output.multimodal_embeddings[:, 0, :]
        neg_embeddings = encoder_output.multimodal_embeddings_neg[:, 0, :]
        itm_loss = self.itm_loss(pos_embeddings, neg_embeddings)

        loss = itc_loss + itm_loss
        return loss

    def _eval_forward(
        self,
        input_type: str,
        image: Optional[Tensor],
        text: Optional[Tensor],
        text_atts: Optional[Tensor],
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        if input_type == "image":
            assert image is not None, "image input tensor cannot be None"
            image_embed = self.model_with_similarity.albef_model.vision_encoder(image)
            image_feat = F.normalize(
                self.model_with_similarity.vision_proj(image_embed[:, 0, :]), dim=-1
            )
            return image_embed, image_feat
        elif input_type == "text":
            assert (
                text is not None and text_atts is not None
            ), "text and text attention mask cannot be None"
            text_embed = self.model_with_similarity.albef_model.text_encoder(
                text, text_atts
            )
            text_feat = F.normalize(
                self.model_with_similarity.text_proj(text_embed[:, 0, :]), dim=-1
            )
            return text_embed, text_feat
        elif input_type == "multimodal":
            assert (
                image is not None and text is not None and text_atts is not None
            ), "image embeddings, text embeddings, and text attention mask cannot be None"
            multimodal_embeds = (
                self.model_with_similarity.albef_model.multimodal_encoder(
                    text,
                    text_atts,
                    image,
                )
            )
            score = self.itm_loss.itm_head(multimodal_embeds[:, 0, :])[:, 1]
            return score
        else:
            raise ValueError("input_type must be image, text, or multimodal")

    def forward(
        self,
        image: Optional[Tensor] = None,
        text: Optional[Tensor] = None,
        text_atts: Optional[Tensor] = None,
        idx: Optional[Tensor] = None,
        alpha: Optional[Tensor] = 0.0,
        input_type: Optional[str] = None,
        is_train: Optional[bool] = True,
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        if is_train:
            return self._train_forward(
                image,
                text,
                text_atts,
                idx,
                alpha,
            )
        else:
            return self._eval_forward(
                input_type,
                image,
                text,
                text_atts,
            )


def albef_model_for_vqa(config: dict, pretrained: bool = False) -> ALBEFModelForVQA:
    vision_encoder = ALBEFVisionEncoder(**config["vision_encoder_args"])
    text_encoder = ALBEFTextEncoder(**config["text_encoder_args"])
    question_multimodal_encoder = ALBEFMultimodalEncoder(
        **config["multimodal_encoder_args"]
    )
    text_embeddings = ALBEFTextEmbeddings(**config["text_embeddings_args"])
    answer_multimodal_encoder = ALBEFMultimodalEncoder(
        **config["multimodal_encoder_args"]
    )
    prediction_head = PredictionHead(**config["prediction_head_args"])
    albef_model = ALBEFModel(vision_encoder, text_encoder, question_multimodal_encoder)
    decoder = ALBEFDecoder(text_embeddings, answer_multimodal_encoder, prediction_head)
    loss = MaskedLanguageModelingLoss()
    model = ALBEFModelForVQA(albef_model, decoder, loss)

    if pretrained:
        load_module_from_url(model, _ALBEF_PRETRAINED_URLS["vqa"])
    return model


def albef_model_for_retrieval(config: dict) -> ALBEFModelForRetrieval:
    vision_encoder = ALBEFVisionEncoder(**config["vision_encoder_args"])
    text_encoder = ALBEFTextEncoder(**config["text_encoder_args"])
    multimodal_encoder = ALBEFMultimodalEncoder(**config["multimodal_encoder_args"])
    vision_proj = nn.Linear(**config["projection_args"])
    text_proj = nn.Linear(**config["projection_args"])

    albef_model = ALBEFModel(vision_encoder, text_encoder, multimodal_encoder)
    albef_model_with_sim = ALBEFModelWithSimilarity(
        albef_model, vision_proj, text_proj, **config["similarity_args"]
    )
    itc_loss = ImageTextContrastiveLoss()
    itm_loss = ImageTextMatchingLoss()

    return ALBEFModelForRetrieval(albef_model_with_sim, itc_loss, itm_loss)
