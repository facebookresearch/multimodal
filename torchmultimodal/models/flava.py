# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Code for some of the transformers components in this file are initialized
# from their counterparts in Hugging Face Transformers library.

import collections
import math
import warnings
from collections import namedtuple, OrderedDict
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union

import torch
from packaging import version
from torch import device, nn, Tensor
from torchmultimodal.modules.layers.mlp import MLP
from torchmultimodal.modules.layers.normalizations import Fp32LayerNorm
from torchmultimodal.modules.layers.transformer import (
    FLAVATransformerEncoder,
    FLAVATransformerOutput,
    FLAVATransformerWithoutEmbeddings,
    init_transformer_weights,
)
from torchmultimodal.modules.losses.flava import (
    FLAVAPretrainingLoss,
    FLAVAPretrainingLossOutput,
    Pooler,
)
from torchmultimodal.utils.common import ModelOutput, PretrainedMixin


EMBEDDING_OPTIONS = Literal["image", "text", "mm"]

FLAVAOutput = namedtuple(
    "FLAVAOutput",
    ["image", "image_masked", "text", "text_masked", "multimodal", "multimodal_masked"],
    defaults=(None, None, None, None, None, None),
)
FLAVAOutput.__annotations__ = {
    "image": FLAVATransformerOutput,
    "image_masked": FLAVATransformerOutput,
    "text": FLAVATransformerOutput,
    "text_masked": FLAVATransformerOutput,
    "multimodal": FLAVATransformerOutput,
    "multimodal_masked": FLAVATransformerOutput,
}


FLAVA_FOR_PRETRAINED_MAPPING = {
    "flava_full": "https://huggingface.co/aps/flava_full_pretrained_encoders_torchmm/resolve/main/pytorch_model.bin",
}


def to_2tuple(x: Any) -> Tuple:
    if isinstance(x, collections.abc.Iterable):
        return tuple(x)
    return (x, x)


@dataclass
class FLAVAForClassificationOutput(ModelOutput):
    logits: Tensor
    loss: Tensor


class FLAVAModel(nn.Module, PretrainedMixin):
    def __init__(
        self,
        image_encoder: nn.Module,
        text_encoder: nn.Module,
        mm_encoder: nn.Module,
        image_to_mm_projection: nn.Module,
        text_to_mm_projection: nn.Module,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.image_encoder = image_encoder
        self.text_encoder = text_encoder
        self.mm_encoder = mm_encoder
        self.image_to_mm_projection = image_to_mm_projection
        self.text_to_mm_projection = text_to_mm_projection

    def forward(
        self,
        image: Optional[Tensor] = None,
        text: Optional[Tensor] = None,
        image_patches_mask: Optional[Tensor] = None,
        text_masked: Optional[Tensor] = None,
        required_embedding: Optional[EMBEDDING_OPTIONS] = None,
        skip_unmasked_mm_encoder: bool = True,
    ) -> FLAVAOutput:
        if required_embedding is None:
            if image is not None and text is not None:
                required_embedding = "mm"
            elif image is not None:
                required_embedding = "image"
            else:
                required_embedding = "text"

        image_outputs = self._encode_data_to_embeddings(
            image,
            required_embedding,
            ["image", "mm"],
            self.encode_image,
        )
        text_outputs = self._encode_data_to_embeddings(
            text,
            required_embedding,
            ["text", "mm"],
            self.encode_text,
        )
        image_masked_outputs = self._encode_data_to_embeddings(
            image,
            required_embedding,
            ["image", "mm"],
            partial(self.encode_image, image_patches_mask=image_patches_mask),
        )
        text_masked_outputs = self._encode_data_to_embeddings(
            text_masked,
            required_embedding,
            ["text", "mm"],
            self.encode_text,
        )

        multimodal_outputs = FLAVATransformerOutput()
        multimodal_masked_outputs = FLAVATransformerOutput()

        if required_embedding == "mm":
            # Take last hidden state and not the last_hidden_state because
            # for flava we want the hidden state without final layernorm.
            if not skip_unmasked_mm_encoder:
                # Unmasked multimodal embedding is not currently used by any of the FLAVA losses.
                multimodal_outputs = self.encode_mm(
                    image_outputs.hidden_states[-1]
                    if image_outputs.hidden_states
                    else None,
                    text_outputs.hidden_states[-1]
                    if text_outputs.hidden_states
                    else None,
                )
            multimodal_masked_outputs = self.encode_mm(
                image_masked_outputs.hidden_states[-1]
                if image_masked_outputs.hidden_states
                else None,
                text_masked_outputs.hidden_states[-1]
                if text_masked_outputs.hidden_states
                else None,
            )

        return FLAVAOutput(
            image=image_outputs,
            image_masked=image_masked_outputs,
            text=text_outputs,
            text_masked=text_masked_outputs,
            multimodal=multimodal_outputs,
            multimodal_masked=multimodal_masked_outputs,
        )

    def encode_image(
        self, image: Tensor, image_patches_mask: Optional[Tensor] = None
    ) -> Optional[FLAVATransformerOutput]:
        if image_patches_mask is not None:
            return self.image_encoder(image, image_patches_mask)
        else:
            return self.image_encoder(image)

    def encode_text(
        self,
        text: Tensor,
        text_mask: Optional[Tensor] = None,
    ) -> Optional[FLAVATransformerOutput]:
        # TODO(asg): Give proper parameter names when implementing text encoder
        return self.text_encoder(
            input_ids=text,
            attention_mask=text_mask,
        )

    def _encode_data_to_embeddings(
        self,
        data: Optional[Tensor],
        selected_head_encoder: EMBEDDING_OPTIONS,
        encoder_options: List[EMBEDDING_OPTIONS],
        encode_callable: Callable[..., FLAVATransformerOutput],
    ) -> Optional[FLAVATransformerOutput]:
        output = FLAVATransformerOutput()

        if data is not None and selected_head_encoder in encoder_options:
            output = encode_callable(data)

        return output

    def encode_mm(
        self,
        image_embedding: Tensor,
        text_embedding: Tensor,
    ) -> FLAVATransformerOutput:
        if image_embedding is None or text_embedding is None:
            # Since nothing is passed, it might be case without
            # masked data let's say.
            return FLAVATransformerOutput()

        image_embedding = self.image_to_mm_projection(image_embedding)
        text_embedding = self.text_to_mm_projection(text_embedding)
        fused_state = torch.cat([image_embedding, text_embedding], dim=1)
        return self.mm_encoder(fused_state)


class FLAVAForPreTraining(nn.Module, PretrainedMixin):
    # TODOs:
    # 1. Expose logit scale
    # 2. For FLAVA model, allow interpolating the embeddings to
    # for patch embeddings
    def __init__(
        self, model: FLAVAModel, image_codebook: nn.Module, loss: FLAVAPretrainingLoss
    ) -> None:
        super().__init__()
        self.model = model
        self.image_codebook = image_codebook
        self.loss = loss

    def encode_image(
        self,
        image: Tensor,
        cls_index: int = 0,
    ) -> Tensor:
        transformer_output = self.model.encode_image(image)
        embeddings = transformer_output.last_hidden_state
        return self.loss.contrastive_loss.image_projection(embeddings[:, cls_index, :])

    def encode_text(
        self,
        text: Tensor,
        text_mask: Optional[Tensor] = None,
        cls_index: int = 0,
    ) -> Tensor:
        transformer_output = self.model.encode_text(text, text_mask)
        embeddings = transformer_output.last_hidden_state
        return self.loss.contrastive_loss.text_projection(embeddings[:, cls_index, :])

    # TODO: Add options to enable losses selectively
    def forward(
        self,
        image: Optional[Tensor] = None,
        text: Optional[Tensor] = None,
        image_for_codebook: Optional[Tensor] = None,
        image_patches_mask: Optional[Tensor] = None,
        text_masked: Optional[Tensor] = None,
        required_embedding: Optional[EMBEDDING_OPTIONS] = None,
        skip_unmasked_mm_encoder: bool = True,
        itm_labels: Optional[Tensor] = None,
        mlm_labels: Optional[Tensor] = None,
    ) -> FLAVAPretrainingLossOutput:
        image_labels = None
        if image_for_codebook is not None:
            image_labels = self.image_codebook(image_for_codebook).flatten(1)
            image_patches_mask = image_patches_mask.flatten(1).to(torch.bool)
            image_labels[~image_patches_mask] = -1

        flava_output: FLAVAOutput = self.model(
            image=image,
            text=text,
            image_patches_mask=image_patches_mask,
            text_masked=text_masked,
            required_embedding=required_embedding,
            skip_unmasked_mm_encoder=skip_unmasked_mm_encoder,
        )

        return self.loss(
            image_sequence=flava_output.image.last_hidden_state,
            text_sequence=flava_output.text.last_hidden_state,
            image_masked_sequence=flava_output.image_masked.last_hidden_state,
            text_masked_sequence=flava_output.text_masked.last_hidden_state,
            multimodal_sequence=flava_output.multimodal.last_hidden_state
            if not skip_unmasked_mm_encoder
            else None,
            multimodal_masked_sequence=flava_output.multimodal_masked.last_hidden_state,
            itm_labels=itm_labels,
            mim_labels=image_labels,
            mlm_labels=mlm_labels,
        )


class FLAVAForClassification(nn.Module, PretrainedMixin):
    def __init__(
        self,
        model: FLAVAModel,
        classifier: nn.Module,
        loss: Union[nn.Module, Callable[[Tensor, Tensor], Tensor]],
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.model = model
        self.classifier = classifier
        self.loss = loss

    def forward(
        self,
        image: Optional[Tensor] = None,
        text: Optional[Tensor] = None,
        required_embedding: Optional[EMBEDDING_OPTIONS] = None,
        labels: Optional[Tensor] = None,
        cls_index: int = 0,
    ) -> FLAVAForClassificationOutput:
        flava_output: FLAVAOutput = self.model(
            image=image,
            text=text,
            required_embedding=required_embedding,
            # Don't skip the encoder for classification
            skip_unmasked_mm_encoder=False,
        )

        hidden_state: Optional[Tensor] = None
        if required_embedding == "image":
            hidden_state = flava_output.image.last_hidden_state
        elif required_embedding == "text":
            hidden_state = flava_output.text.last_hidden_state
        else:
            hidden_state = flava_output.multimodal.last_hidden_state

        scores = self.classifier(hidden_state[:, cls_index])
        loss = self.loss(scores, labels)
        return FLAVAForClassificationOutput(
            logits=scores,
            loss=loss,
        )


# Based on timm implementation, which can be found here:
# https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
class PatchEmbeddings(nn.Module):
    """
    Image to Patch Embedding.
    """

    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 16,
        num_channels: int = 3,
        embed_dim: int = 768,
    ) -> None:
        super().__init__()
        image_size = to_2tuple(image_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (image_size[1] // patch_size[1]) * (
            image_size[0] // patch_size[0]
        )
        self.image_size: Tuple[int, int] = (image_size[0], image_size[1])
        self.patch_size: Tuple[int, int] = (patch_size[0], patch_size[1])
        self.num_patches = num_patches

        self.projection = nn.Conv2d(
            num_channels, embed_dim, kernel_size=self.patch_size, stride=self.patch_size
        )

    def forward(
        self, pixel_values: Tensor, interpolate_pos_encoding: bool = False
    ) -> Tensor:
        _, _, height, width = pixel_values.shape
        if not interpolate_pos_encoding:
            if height != self.image_size[0] or width != self.image_size[1]:
                raise ValueError(
                    f"Input image size ({height}*{width}) doesn't match model ({self.image_size[0]}*{self.image_size[1]})."
                )
        x = self.projection(pixel_values).flatten(2).transpose(1, 2)
        return x


class ImageEmbeddings(nn.Module):
    """
    Construct the CLS token, position and patch embeddings.
    """

    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 16,
        num_channels: int = 3,
        hidden_size: int = 768,
        hidden_dropout_prob: float = 0.0,
        use_image_masking: bool = True,
    ) -> None:
        super().__init__()

        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_size))
        self.patch_embeddings = PatchEmbeddings(
            image_size=image_size,
            patch_size=patch_size,
            num_channels=num_channels,
            embed_dim=hidden_size,
        )
        num_patches = self.patch_embeddings.num_patches
        self.position_embeddings = nn.Parameter(
            torch.zeros(1, num_patches + 1, hidden_size)
        )
        self.dropout = nn.Dropout(hidden_dropout_prob)

        if use_image_masking:
            self.mask_token = nn.Parameter(torch.zeros(1, 1, hidden_size))
        else:
            self.mask_token = None

    def interpolate_pos_encoding(
        self, embeddings: Tensor, height: int, width: int
    ) -> Tensor:
        """
        This method allows to interpolate the pre-trained position encodings, to be able to use the model on higher
        resolution images.
        Source:
        https://github.com/facebookresearch/dino/blob/de9ee3df6cf39fac952ab558447af1fa1365362a/vision_transformer.py#L174
        """

        npatch = embeddings.shape[1] - 1
        n = self.position_embeddings.shape[1] - 1
        if npatch == n and height == width:
            return self.position_embeddings
        class_pos_embed = self.position_embeddings[:, 0]
        patch_pos_embed = self.position_embeddings[:, 1:]
        dim = embeddings.shape[-1]
        h0 = height // self.patch_embeddings.patch_size[0]
        w0 = width // self.patch_embeddings.patch_size[1]
        # we add a small number to avoid floating point error in the interpolation
        # see discussion at https://github.com/facebookresearch/dino/issues/8
        h0, w0 = h0 + 0.1, w0 + 0.1
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(
                1, int(math.sqrt(n)), int(math.sqrt(n)), dim
            ).permute(0, 3, 1, 2),
            scale_factor=(h0 / math.sqrt(n), w0 / math.sqrt(n)),
            mode="bicubic",
            align_corners=False,
        )
        assert (
            int(h0) == patch_pos_embed.shape[-2]
            and int(w0) == patch_pos_embed.shape[-1]
        )
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)

    def forward(
        self,
        pixel_values: Tensor,
        image_patches_mask: Optional[Tensor] = None,
        interpolate_pos_encoding: bool = False,
    ) -> Tensor:
        batch_size, num_channels, height, width = pixel_values.shape
        embeddings = self.patch_embeddings(
            pixel_values, interpolate_pos_encoding=interpolate_pos_encoding
        )

        _, seq_len, _ = embeddings.size()
        if image_patches_mask is not None:
            if self.mask_token is not None:
                mask_tokens = self.mask_token.expand(batch_size, seq_len, -1)
                # replace the masked visual tokens by mask_tokens
                w = image_patches_mask.unsqueeze(-1).type_as(mask_tokens)
                embeddings = embeddings * (1 - w) + mask_tokens * w
            else:
                warnings.warn(
                    "image_patches_mask passed but use_image_masking in init was false. Ignoring."
                )
        # add the [CLS] token to the embedded patch tokens
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        embeddings = torch.cat((cls_tokens, embeddings), dim=1)

        # add positional encoding to each token
        if interpolate_pos_encoding:
            embeddings = embeddings + self.interpolate_pos_encoding(
                embeddings, height, width
            )
        else:
            embeddings = embeddings + self.position_embeddings

        embeddings = self.dropout(embeddings)

        return embeddings


class ImageTransformer(nn.Module):
    # TODO(asg): Add support for pretrained checkpoint loading
    def __init__(
        self,
        embeddings: nn.Module,
        encoder: nn.Module,
        layernorm: nn.Module,
        pooler: nn.Module,
        weight_init_fn: Optional[Callable] = None,
        initializer_range: float = 0.02,
        **kwargs: Any,
    ) -> None:
        super().__init__()

        self.embeddings = embeddings
        self.encoder = encoder
        self.layernorm = layernorm
        self.pooler = pooler

        if weight_init_fn is None:
            weight_init_fn = partial(
                init_transformer_weights, initializer_range=initializer_range
            )

        self.apply(weight_init_fn)

    def forward(
        self,
        pixel_values: Optional[Tensor] = None,
        image_patches_mask: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
    ) -> FLAVATransformerOutput:
        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        embedding_output = self.embeddings(
            pixel_values, image_patches_mask=image_patches_mask
        )

        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=attention_mask,
        )
        sequence_output = encoder_outputs[0]
        sequence_output = self.layernorm(sequence_output)
        pooled_output = (
            self.pooler(sequence_output) if self.pooler is not None else None
        )

        return FLAVATransformerOutput(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


class ImageTransformerWithVAE(nn.Module):
    def __init__(
        self,
        image_transformer: nn.Module,
        vae: nn.Module,
        **kwargs: Dict[str, Any],
    ) -> None:
        super().__init__()

        self.image_transformer = image_transformer
        self.vae = vae

    def forward(
        self,
        pixel_values: Optional[Tensor] = None,
        image_patches_mask: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
    ) -> FLAVATransformerOutput:
        image_labels = self.vae(pixel_values).flatten(1)
        image_patches_mask = image_patches_mask.flatten(1).to(torch.bool)
        image_labels[image_patches_mask == False] = -1  # noqa

        output = self.image_transformer(
            pixel_values=pixel_values,
            image_patches_mask=image_patches_mask,
            attention_mask=attention_mask,
        )
        return FLAVATransformerOutput(
            last_hidden_state=output.last_hidden_state,
            pooler_output=output.pooler_output,
            hidden_states=output.hidden_states,
            attentions=output.attentions,
            image_labels=image_labels,
        )


class TextEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings following BERT."""

    def __init__(
        self,
        hidden_size: int = 768,
        vocab_size: int = 30522,
        pad_token_id: int = 0,
        type_vocab_size: int = 2,
        max_position_embeddings: int = 512,
        layer_norm_eps: float = 1e-12,
        hidden_dropout_prob: float = 0.1,
    ) -> None:
        super().__init__()
        self.word_embeddings = nn.Embedding(
            vocab_size, hidden_size, padding_idx=pad_token_id
        )
        self.position_embeddings = nn.Embedding(max_position_embeddings, hidden_size)
        self.token_type_embeddings = nn.Embedding(type_vocab_size, hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.dropout = nn.Dropout(hidden_dropout_prob)
        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.register_buffer(
            "position_ids", torch.arange(max_position_embeddings).expand((1, -1))
        )
        self.position_ids: Tensor
        if version.parse(torch.__version__) > version.parse("1.6.0"):
            self.register_buffer(
                "token_type_ids",
                torch.zeros(self.position_ids.size(), dtype=torch.long),
                persistent=False,
            )
            self.token_type_ids: Tensor

    def forward(
        self,
        input_ids: Optional[Tensor] = None,
        token_type_ids: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
        inputs_embeds: Optional[Tensor] = None,
        past_key_values_length: int = 0,
    ) -> Tensor:
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]

        if position_ids is None:
            position_ids = self.position_ids[
                :, past_key_values_length : seq_length + past_key_values_length
            ]

        # Setting the token_type_ids to the registered buffer in constructor where it is all zeros, which usually occurs
        # when its auto-generated, registered buffer helps users when tracing the model without passing token_type_ids, solves
        # issue #5664
        if token_type_ids is None:
            if hasattr(self, "token_type_ids"):
                buffered_token_type_ids = self.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(
                    input_shape[0], seq_length
                )
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(
                    input_shape, dtype=torch.long, device=self.position_ids.device
                )

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = inputs_embeds + token_type_embeddings
        position_embeddings = self.position_embeddings(position_ids)
        embeddings += position_embeddings

        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class TextTransformer(nn.Module):
    # TODO(asg): Add support for pretrained checkpoint loading
    def __init__(
        self,
        embeddings: nn.Module,
        encoder: nn.Module,
        layernorm: nn.Module,
        pooler: nn.Module,
        weight_init_fn: Optional[Callable] = None,
        initializer_range: float = 0.02,
        pad_token_id: int = 0,
        **kwargs: Any,
    ) -> None:
        super().__init__()

        self.embeddings = embeddings
        self.encoder = encoder
        self.layernorm = layernorm
        self.pooler = pooler
        self.pad_token_id = pad_token_id

        if weight_init_fn is None:
            weight_init_fn = partial(
                init_transformer_weights, initializer_range=initializer_range
            )

        self.apply(weight_init_fn)

    def get_extended_attention_mask(
        self, attention_mask: Tensor, input_shape: Tuple[int, ...], device: device
    ) -> Tensor:
        """
        Makes broadcastable attention and causal masks so that future and masked tokens are ignored.
        Arguments:
            attention_mask (`torch.Tensor`):
                Mask with ones indicating tokens to attend to, zeros for tokens to ignore.
            input_shape (`Tuple[int]`):
                The shape of the input to the model.
            device: (`torch.device`):
                The device of the input to the model.
        Returns:
            `torch.Tensor` The extended attention mask, with a the same dtype as `attention_mask.dtype`.
        """
        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        if attention_mask.dim() == 3:
            extended_attention_mask = attention_mask[:, None, :, :]
        elif attention_mask.dim() == 2:
            # Provided a padding mask of dimensions [batch_size, seq_length]
            # - if the model is an encoder, make the mask broadcastable to [batch_size, num_heads, seq_length, seq_length]
            extended_attention_mask = attention_mask[:, None, None, :]
        else:
            raise ValueError(
                f"Wrong shape for input_ids (shape {input_shape}) or attention_mask (shape {attention_mask.shape})"
            )

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(
            dtype=attention_mask.dtype
        )  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        return extended_attention_mask

    def forward(
        self,
        input_ids: Optional[Tensor] = None,
        token_type_ids: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
    ) -> FLAVATransformerOutput:
        if input_ids is None:
            raise ValueError("You have to specify input_ids")
        input_shape = input_ids.size()
        device = input_ids.device

        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
            attention_mask[input_ids == self.pad_token_id] = 0
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # We can provide a self-attention mask of dimensions
        # [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(
            attention_mask, input_shape, device
        )

        embedding_output = self.embeddings(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
        )

        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
        )
        sequence_output = encoder_outputs[0]
        sequence_output = self.layernorm(sequence_output)
        pooled_output = (
            self.pooler(sequence_output) if self.pooler is not None else None
        )

        return FLAVATransformerOutput(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


class DalleConv2d(nn.Module):
    def __init__(self, n_in: int, n_out: int, kw: int) -> None:
        super().__init__()

        w = torch.empty((n_out, n_in, kw, kw), dtype=torch.float32)
        w.normal_(std=1 / math.sqrt(n_in * kw**2))

        b = torch.zeros((n_out,), dtype=torch.float32)
        self.w, self.b = nn.Parameter(w), nn.Parameter(b)
        self.kw = kw

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return nn.functional.conv2d(x, self.w, self.b, padding=(self.kw - 1) // 2)


class DalleEncoderBlock(nn.Module):
    def __init__(self, n_in: int, n_out: int, n_layers: int) -> None:
        super().__init__()
        n_hid = n_out // 4
        self.post_gain = 1 / (n_layers**2)

        self.id_path = DalleConv2d(n_in, n_out, 1) if n_in != n_out else nn.Identity()
        self.res_path = nn.Sequential(
            OrderedDict(
                [
                    ("relu_1", nn.ReLU()),
                    ("conv_1", DalleConv2d(n_in, n_hid, 3)),
                    ("relu_2", nn.ReLU()),
                    ("conv_2", DalleConv2d(n_hid, n_hid, 3)),
                    ("relu_3", nn.ReLU()),
                    ("conv_3", DalleConv2d(n_hid, n_hid, 3)),
                    ("relu_4", nn.ReLU()),
                    ("conv_4", DalleConv2d(n_hid, n_out, 1)),
                ]
            )
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.id_path(x) + self.post_gain * self.res_path(x)


class DalleEncoder(nn.Module):
    def __init__(
        self,
        group_count: int = 4,
        n_hid: int = 256,
        n_blk_per_group: int = 2,
        input_channels: int = 3,
        vocab_size: int = 8192,
        **kwargs: Any,
    ) -> None:
        super().__init__()

        self.input_channels = input_channels
        n_layers = group_count * n_blk_per_group
        output_conv = DalleConv2d(8 * n_hid, vocab_size, 1)

        self.blocks = nn.Sequential(
            OrderedDict(
                [
                    ("input", DalleConv2d(input_channels, 1 * n_hid, 7)),
                    (
                        "group_1",
                        self._create_group(
                            n_layers, n_blk_per_group, 1 * n_hid, 1 * n_hid
                        ),
                    ),
                    (
                        "group_2",
                        self._create_group(
                            n_layers, n_blk_per_group, 1 * n_hid, 2 * n_hid
                        ),
                    ),
                    (
                        "group_3",
                        self._create_group(
                            n_layers, n_blk_per_group, 2 * n_hid, 4 * n_hid
                        ),
                    ),
                    (
                        "group_4",
                        self._create_group(
                            n_layers,
                            n_blk_per_group,
                            4 * n_hid,
                            8 * n_hid,
                            use_pool=False,
                        ),
                    ),
                    (
                        "output",
                        nn.Sequential(
                            OrderedDict([("relu", nn.ReLU()), ("conv", output_conv)])
                        ),
                    ),
                ]
            )
        )

    def _create_group(
        self,
        n_layers: int,
        n_blk_per_group: int,
        n_in: int,
        n_hid: int,
        use_pool: bool = True,
    ) -> nn.Module:
        make_blk = partial(DalleEncoderBlock, n_layers=n_layers)
        blk_range = range(n_blk_per_group)
        blocks: OrderedDict[str, nn.Module] = OrderedDict()
        for i in blk_range:
            if i == 0:
                blocks[f"block_{i+1}"] = make_blk(n_in, n_hid)
            else:
                blocks[f"block_{i+1}"] = make_blk(n_hid, n_hid)

        if use_pool:
            blocks["pool"] = nn.MaxPool2d(kernel_size=2)

        return nn.Sequential(blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if len(x.shape) != 4:
            raise ValueError(f"input shape {x.shape} is not 4d")
        if x.shape[1] != self.input_channels:
            raise ValueError(
                f"input has {x.shape[1]} channels but model built for {self.input_channels}"
            )
        # if x.dtype != torch.float32:
        # 	raise ValueError('input must have dtype torch.float32')
        return self.blocks(x)


class DalleVAEEncoder(nn.Module, PretrainedMixin):
    def __init__(
        self, image_size: Union[int, Tuple[int, int]] = 112, pretrained: bool = True
    ) -> None:
        super().__init__()
        self.image_size = image_size
        self.encoder = DalleEncoder()
        if pretrained:
            self.load_model()

    def load_model(
        self,
        pretrained_url: str = None,
        load_state_dict: bool = False,
        state_dict_key: Optional[str] = None,
    ) -> Dict[str, Any]:
        # TODO (T116682215): Network error due to FLAVA model relying on access to openAI
        encoder = super().load_model(
            "https://cdn.openai.com/dall-e/encoder.pkl", load_state_dict=load_state_dict
        )
        if isinstance(encoder, dict):
            self.encoder.load_state_dict(encoder)
        else:
            self.encoder.load_state_dict(encoder.state_dict())
        return self.state_dict()

    def get_codebook_indices(self, images: Tensor) -> Tensor:
        z_logits = self.encoder(images)
        return torch.argmax(z_logits, dim=1)

    def get_codebook_probs(self, images: Tensor) -> Tensor:
        z_logits = self.encoder(images)
        return nn.Softmax(dim=1)(z_logits)

    def forward(self, img_seq_prob: Tensor) -> Tensor:
        return self.get_codebook_indices(img_seq_prob)


def flava_image_encoder(
    hidden_size: int = 768,
    num_attention_heads: int = 12,
    num_hidden_layers: int = 12,
    use_image_masking: bool = False,
    hidden_dropout_prob: float = 0.0,
    intermediate_size: int = 3072,
    intermediate_activation: Callable[..., Tensor] = nn.functional.gelu,
    attention_probs_dropout_prob: float = 0.0,
    layer_norm_eps: float = 1e-12,
    image_size: int = 224,
    patch_size: int = 16,
    num_channels: int = 3,
) -> ImageTransformer:

    embeddings = ImageEmbeddings(
        image_size=image_size,
        patch_size=patch_size,
        num_channels=num_channels,
        hidden_size=hidden_size,
        hidden_dropout_prob=hidden_dropout_prob,
        use_image_masking=use_image_masking,
    )
    encoder = FLAVATransformerEncoder(
        hidden_size=hidden_size,
        num_attention_heads=num_attention_heads,
        num_hidden_layers=num_hidden_layers,
        hidden_dropout_prob=hidden_dropout_prob,
        intermediate_size=intermediate_size,
        intermediate_activation=intermediate_activation,
        attention_probs_dropout_prob=attention_probs_dropout_prob,
        layer_norm_eps=layer_norm_eps,
    )

    layernorm = Fp32LayerNorm(hidden_size, eps=layer_norm_eps)
    pooler = Pooler(hidden_size=hidden_size)

    return ImageTransformer(
        embeddings=embeddings,
        encoder=encoder,
        layernorm=layernorm,
        pooler=pooler,
    )


def flava_text_encoder(
    hidden_size: int = 768,
    num_attention_heads: int = 12,
    num_hidden_layers: int = 12,
    hidden_dropout_prob: float = 0.0,
    intermediate_size: int = 3072,
    intermediate_activation: Callable[..., Tensor] = nn.functional.gelu,
    attention_probs_dropout_prob: float = 0.0,
    layer_norm_eps: float = 1e-12,
    vocab_size: int = 30522,
    pad_token_id: int = 0,
    type_vocab_size: int = 2,
    max_position_embeddings: int = 512,
) -> TextTransformer:
    embeddings = TextEmbeddings(
        hidden_size=hidden_size,
        vocab_size=vocab_size,
        pad_token_id=pad_token_id,
        type_vocab_size=type_vocab_size,
        max_position_embeddings=max_position_embeddings,
        layer_norm_eps=layer_norm_eps,
        hidden_dropout_prob=hidden_dropout_prob,
    )

    encoder = FLAVATransformerEncoder(
        hidden_size=hidden_size,
        num_attention_heads=num_attention_heads,
        num_hidden_layers=num_hidden_layers,
        hidden_dropout_prob=hidden_dropout_prob,
        intermediate_size=intermediate_size,
        intermediate_activation=intermediate_activation,
        attention_probs_dropout_prob=attention_probs_dropout_prob,
        layer_norm_eps=layer_norm_eps,
        pad_token_id=pad_token_id,
    )

    layernorm = Fp32LayerNorm(hidden_size, eps=layer_norm_eps)
    pooler = Pooler(hidden_size=hidden_size)

    return TextTransformer(
        embeddings=embeddings,
        encoder=encoder,
        layernorm=layernorm,
        pooler=pooler,
    )


def flava_multimodal_encoder(
    hidden_size: int = 768,
    num_attention_heads: int = 12,
    num_hidden_layers: int = 12,
    hidden_dropout_prob: float = 0.0,
    intermediate_size: int = 3072,
    intermediate_activation: Callable[..., Tensor] = nn.functional.gelu,
    attention_probs_dropout_prob: float = 0.0,
    layer_norm_eps: float = 1e-12,
) -> FLAVATransformerWithoutEmbeddings:
    encoder = FLAVATransformerEncoder(
        hidden_size=hidden_size,
        num_attention_heads=num_attention_heads,
        num_hidden_layers=num_hidden_layers,
        hidden_dropout_prob=hidden_dropout_prob,
        intermediate_size=intermediate_size,
        intermediate_activation=intermediate_activation,
        attention_probs_dropout_prob=attention_probs_dropout_prob,
        layer_norm_eps=layer_norm_eps,
    )
    layernorm = Fp32LayerNorm(hidden_size, eps=layer_norm_eps)
    pooler = Pooler(hidden_size=hidden_size)

    return FLAVATransformerWithoutEmbeddings(
        encoder=encoder,
        layernorm=layernorm,
        pooler=pooler,
    )


# NOTE:
# 1) There is a possibility of using dataclass for similar
#    style kwargs for encoders. Didn't explore due to readability.
def flava_model(
    # Image encoder specific parameters
    image_hidden_size: int = 768,
    image_num_attention_heads: int = 12,
    image_num_hidden_layers: int = 12,
    image_hidden_dropout_prob: float = 0.0,
    image_intermediate_size: int = 3072,
    image_intermediate_activation: Callable[..., Tensor] = nn.functional.gelu,
    image_attention_probs_dropout_prob: float = 0.0,
    image_layer_norm_eps: float = 1e-12,
    use_image_masking: bool = True,
    image_size: int = 224,
    patch_size: int = 16,
    num_channels: int = 3,
    # Text encoder specific parameters
    text_hidden_size: int = 768,
    text_num_attention_heads: int = 12,
    text_num_hidden_layers: int = 12,
    text_hidden_dropout_prob: float = 0.0,
    text_intermediate_size: int = 3072,
    text_intermediate_activation: Callable[..., Tensor] = nn.functional.gelu,
    text_attention_probs_dropout_prob: float = 0.0,
    text_layer_norm_eps: float = 1e-12,
    vocab_size: int = 30522,
    pad_token_id: int = 0,
    type_vocab_size: int = 2,
    max_position_embeddings: int = 512,
    # Multimodal encoder specific parameters
    multimodal_hidden_size: int = 768,
    multimodal_num_attention_heads: int = 12,
    multimodal_num_hidden_layers: int = 6,
    multimodal_hidden_dropout_prob: float = 0.0,
    multimodal_intermediate_size: int = 3072,
    multimodal_intermediate_activation: Callable[..., Tensor] = nn.functional.gelu,
    multimodal_attention_probs_dropout_prob: float = 0.0,
    multimodal_layer_norm_eps: float = 1e-12,
    **kwargs: Any,
) -> FLAVAModel:
    image_encoder = flava_image_encoder(
        hidden_size=image_hidden_size,
        num_attention_heads=image_num_attention_heads,
        num_hidden_layers=image_num_hidden_layers,
        use_image_masking=use_image_masking,
        hidden_dropout_prob=image_hidden_dropout_prob,
        intermediate_size=image_intermediate_size,
        intermediate_activation=image_intermediate_activation,
        attention_probs_dropout_prob=image_attention_probs_dropout_prob,
        layer_norm_eps=image_layer_norm_eps,
        image_size=image_size,
        patch_size=patch_size,
        num_channels=num_channels,
    )

    text_encoder = flava_text_encoder(
        hidden_size=text_hidden_size,
        num_attention_heads=text_num_attention_heads,
        num_hidden_layers=text_num_hidden_layers,
        hidden_dropout_prob=text_hidden_dropout_prob,
        intermediate_size=text_intermediate_size,
        intermediate_activation=text_intermediate_activation,
        attention_probs_dropout_prob=text_attention_probs_dropout_prob,
        layer_norm_eps=text_layer_norm_eps,
        vocab_size=vocab_size,
        pad_token_id=pad_token_id,
        type_vocab_size=type_vocab_size,
        max_position_embeddings=max_position_embeddings,
    )
    mm_encoder = flava_multimodal_encoder(
        hidden_size=multimodal_hidden_size,
        num_attention_heads=multimodal_num_attention_heads,
        num_hidden_layers=multimodal_num_hidden_layers,
        hidden_dropout_prob=multimodal_hidden_dropout_prob,
        intermediate_size=multimodal_intermediate_size,
        intermediate_activation=multimodal_intermediate_activation,
        attention_probs_dropout_prob=multimodal_attention_probs_dropout_prob,
        layer_norm_eps=multimodal_layer_norm_eps,
    )

    image_to_mm_projection = nn.Linear(image_hidden_size, multimodal_hidden_size)
    text_to_mm_projection = nn.Linear(text_hidden_size, multimodal_hidden_size)

    return FLAVAModel(
        image_encoder=image_encoder,
        text_encoder=text_encoder,
        mm_encoder=mm_encoder,
        image_to_mm_projection=image_to_mm_projection,
        text_to_mm_projection=text_to_mm_projection,
    )


def flava_model_for_pretraining(
    codebook_image_size: int = 112,
    pretrained_model_key: Optional[str] = None,
    **flava_model_kwargs: Any,
    # TODO: Add parameters for loss here
) -> FLAVAForPreTraining:
    model = flava_model(**flava_model_kwargs)

    codebook = DalleVAEEncoder(image_size=codebook_image_size)
    losses = FLAVAPretrainingLoss()

    flava = FLAVAForPreTraining(
        model=model,
        image_codebook=codebook,
        loss=losses,
    )

    if pretrained_model_key is not None:
        flava.load_model(FLAVA_FOR_PRETRAINED_MAPPING[pretrained_model_key])

    return flava


def flava_model_for_classification(
    num_classes: int,
    classifier_in_dim: int = 768,
    classifier_hidden_sizes: Union[int, List[int]] = 768,
    classifier_dropout: float = 0.5,
    classifier_activation: Callable[..., nn.Module] = nn.ReLU,
    classifier_normalization: Optional[Callable[..., nn.Module]] = None,
    loss_fn: Optional[Callable[..., Tensor]] = None,
    **flava_model_kwargs: Any,
) -> FLAVAForClassification:
    model = flava_model(**flava_model_kwargs)
    classifier = MLP(
        in_dim=classifier_in_dim,
        out_dim=num_classes,
        hidden_dims=classifier_hidden_sizes,
        dropout=classifier_dropout,
        activation=classifier_activation,
        normalization=classifier_normalization,
    )

    if loss_fn is None:
        loss_fn = nn.CrossEntropyLoss()

    return FLAVAForClassification(model=model, classifier=classifier, loss=loss_fn)
