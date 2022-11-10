# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Code for some of the transformers components in this file are initialized
# from their counterparts in Hugging Face Transformers library.

import math
from collections import namedtuple, OrderedDict
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, List, Optional, Tuple, Union

import torch
from torch import nn, Tensor
from torchmultimodal.models.flava.image_encoder import flava_image_encoder
from torchmultimodal.models.flava.text_encoder import flava_text_encoder
from torchmultimodal.models.flava.transformer import FLAVATransformerWithoutEmbeddings
from torchmultimodal.modules.layers.mlp import MLP
from torchmultimodal.modules.layers.normalizations import Fp32LayerNorm
from torchmultimodal.modules.layers.transformer import (
    TransformerEncoder,
    TransformerOutput,
)
from torchmultimodal.modules.losses.flava import (
    FLAVAPretrainingLoss,
    FLAVAPretrainingLossOutput,
    Pooler,
)
from torchmultimodal.utils.common import load_module_from_url, ModelOutput
from typing_extensions import Literal


EMBEDDING_OPTIONS = Literal["image", "text", "mm"]

FLAVAOutput = namedtuple(
    "FLAVAOutput",
    [
        "image",
        "image_masked",
        "text",
        "text_masked",
        "multimodal",
        "multimodal_masked",
        "projected_image_embeddings",
        "projected_text_embeddings",
    ],
    defaults=(None, None, None, None, None, None, None, None),
)
FLAVAOutput.__annotations__ = {
    "image": TransformerOutput,
    "image_masked": TransformerOutput,
    "text": TransformerOutput,
    "text_masked": TransformerOutput,
    "multimodal": TransformerOutput,
    "multimodal_masked": TransformerOutput,
}

CKPT_KEY = "flava_full"
FLAVA_FOR_PRETRAINED_MAPPING = {
    # This will no longer load with the updated model, but keeping here just in case
    # "flava_full": "https://huggingface.co/aps/flava_full_pretrained_encoders_torchmm/resolve/main/pytorch_model.bin",
    CKPT_KEY: "https://download.pytorch.org/models/multimodal/flava/flava_for_pretraining_unified_text_encoder.pt",
}

FLAVA_MODEL_MAPPING = {
    CKPT_KEY: "https://download.pytorch.org/models/multimodal/flava/flava_model_unified_text_encoder.pt",
}


def flava_multimodal_encoder(
    hidden_size: int = 768,
    num_attention_heads: int = 12,
    num_hidden_layers: int = 12,
    dropout: float = 0.0,
    intermediate_size: int = 3072,
    intermediate_activation: Callable[..., nn.Module] = nn.GELU,
    layer_norm_eps: float = 1e-12,
) -> FLAVATransformerWithoutEmbeddings:
    encoder = TransformerEncoder(
        n_layer=num_hidden_layers,
        d_model=hidden_size,
        n_head=num_attention_heads,
        dim_feedforward=intermediate_size,
        activation=intermediate_activation,
        layer_norm_eps=layer_norm_eps,
        dropout=dropout,
        norm_first=True,
    )
    layernorm = Fp32LayerNorm(hidden_size, eps=layer_norm_eps)
    pooler = Pooler(hidden_size=hidden_size)

    return FLAVATransformerWithoutEmbeddings(
        encoder=encoder, layernorm=layernorm, pooler=pooler, hidden_size=hidden_size
    )


@dataclass
class FLAVAForClassificationOutput(ModelOutput):
    logits: Tensor
    loss: Tensor


class FLAVAModel(nn.Module):
    def __init__(
        self,
        image_encoder: nn.Module,
        text_encoder: nn.Module,
        mm_encoder: nn.Module,
        image_to_mm_projection: nn.Module,
        text_to_mm_projection: nn.Module,
        text_projection: nn.Module,
        image_projection: nn.Module,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.image_encoder = image_encoder
        self.text_encoder = text_encoder
        self.mm_encoder = mm_encoder
        self.image_to_mm_projection = image_to_mm_projection
        self.text_to_mm_projection = text_to_mm_projection
        self.text_projection = text_projection
        self.image_projection = image_projection

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

        image_encoding_out = self._encode_data_to_embeddings(
            image,
            required_embedding,
            ["image", "mm"],
            partial(self.encode_image, projection=True),
        )
        if len(image_encoding_out) == 2:
            image_outputs, projected_image_embeddings = (
                image_encoding_out[0],
                image_encoding_out[1],
            )
        else:
            image_outputs = image_encoding_out  # type: ignore
            projected_image_embeddings = None

        text_encoding_out = self._encode_data_to_embeddings(
            text,
            required_embedding,
            ["text", "mm"],
            partial(self.encode_text, projection=True),
        )
        if len(text_encoding_out) == 2:
            text_outputs, projected_text_embeddings = (
                text_encoding_out[0],
                text_encoding_out[1],
            )
        else:
            text_outputs = text_encoding_out  # type: ignore
            projected_text_embeddings = None

        image_masked_outputs = self._encode_data_to_embeddings(
            image,
            required_embedding,
            ["image", "mm"],
            partial(self.encode_image, image_patches_mask=image_patches_mask),
        )
        assert type(image_masked_outputs) == TransformerOutput
        text_masked_outputs = self._encode_data_to_embeddings(
            text_masked,
            required_embedding,
            ["text", "mm"],
            self.encode_text,
        )
        assert type(text_masked_outputs) == TransformerOutput

        multimodal_outputs = TransformerOutput()
        multimodal_masked_outputs = TransformerOutput()

        if required_embedding == "mm":
            # Take last hidden state and not the last_hidden_state because
            # for flava we want the hidden state without final layernorm.
            if not skip_unmasked_mm_encoder:
                # Unmasked multimodal embedding is not currently used by any of the FLAVA losses.
                multimodal_outputs = self.encode_mm(
                    image_outputs.hidden_states[-1]  # type: ignore
                    if image_outputs.hidden_states  # type: ignore
                    else None,
                    text_outputs.hidden_states[-1]  # type: ignore
                    if text_outputs.hidden_states  # type: ignore
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
            projected_image_embeddings=projected_image_embeddings,
            projected_text_embeddings=projected_text_embeddings,
        )

    def encode_image(
        self,
        image: Tensor,
        image_patches_mask: Optional[Tensor] = None,
        projection: bool = False,
    ) -> Union[Tuple[TransformerOutput, Tensor], Optional[TransformerOutput]]:
        if image_patches_mask is not None:
            encoded_image = self.image_encoder(image, image_patches_mask)
        else:
            encoded_image = self.image_encoder(image)
        if projection:
            projected_embeddings = self.image_projection(
                encoded_image.last_hidden_state[:, 0, :]
            )
            return encoded_image, projected_embeddings
        return encoded_image

    def encode_text(
        self, text: Tensor, text_mask: Optional[Tensor] = None, projection: bool = False
    ) -> Union[Tuple[TransformerOutput, Tensor], Optional[TransformerOutput]]:
        # TODO(asg): Give proper parameter names when implementing text encoder
        encoded_text = self.text_encoder(
            input_ids=text,
            attention_mask=text_mask,
            return_attn_weights=True,
            return_hidden_states=True,
        )
        if projection:
            projected_embeddings = self.text_projection(
                encoded_text.last_hidden_state[:, 0, :]
            )
            return encoded_text, projected_embeddings
        return encoded_text

    def _encode_data_to_embeddings(
        self,
        data: Optional[Tensor],
        selected_head_encoder: EMBEDDING_OPTIONS,
        encoder_options: List[EMBEDDING_OPTIONS],
        encode_callable: Callable[
            ...,
            Union[Tuple[TransformerOutput, Tensor], Optional[TransformerOutput]],
        ],
    ) -> Union[Tuple[TransformerOutput, Tensor], Optional[TransformerOutput]]:
        output: Union[
            Tuple[TransformerOutput, Tensor], TransformerOutput
        ] = TransformerOutput()

        if data is not None and selected_head_encoder in encoder_options:
            output = encode_callable(data)
        return output

    def encode_mm(
        self,
        image_embedding: Tensor,
        text_embedding: Tensor,
    ) -> TransformerOutput:
        if image_embedding is None or text_embedding is None:
            # Since nothing is passed, it might be case without
            # masked data let's say.
            return TransformerOutput()

        image_embedding = self.image_to_mm_projection(image_embedding)
        text_embedding = self.text_to_mm_projection(text_embedding)
        fused_state = torch.cat([image_embedding, text_embedding], dim=1)
        return self.mm_encoder(fused_state)


class FLAVAForPreTraining(nn.Module):
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
        encoded_result = self.model.encode_image(image, projection=True)
        encoded_image = encoded_result[1]
        return encoded_image

    def encode_text(
        self,
        text: Tensor,
        text_mask: Optional[Tensor] = None,
        cls_index: int = 0,
    ) -> Tensor:
        encoded_result = self.model.encode_text(text, text_mask, projection=True)
        encoded_text = encoded_result[1]
        return encoded_text

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
            projected_image_embeddings=flava_output.projected_image_embeddings,
            projected_text_embeddings=flava_output.projected_text_embeddings,
        )


class FLAVAForClassification(nn.Module):
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


# NOTE:
# 1) There is a possibility of using dataclass for similar
#    style kwargs for encoders. Didn't explore due to readability.
def flava_model(
    # Image encoder specific parameters
    image_hidden_size: int = 768,
    image_num_attention_heads: int = 12,
    image_num_hidden_layers: int = 12,
    image_dropout: float = 0.0,
    image_intermediate_size: int = 3072,
    image_intermediate_activation: Callable[..., nn.Module] = nn.GELU,
    image_layer_norm_eps: float = 1e-12,
    use_image_masking: bool = True,
    image_size: int = 224,
    patch_size: int = 16,
    num_channels: int = 3,
    # Text encoder specific parameters
    text_hidden_size: int = 768,
    text_num_attention_heads: int = 12,
    text_num_hidden_layers: int = 12,
    text_dropout: float = 0.0,
    text_intermediate_size: int = 3072,
    text_intermediate_activation: Callable[..., nn.Module] = nn.GELU,
    text_layer_norm_eps: float = 1e-12,
    vocab_size: int = 30522,
    pad_token_id: int = 0,
    type_vocab_size: int = 2,
    max_position_embeddings: int = 512,
    # Multimodal encoder specific parameters
    multimodal_hidden_size: int = 768,
    multimodal_num_attention_heads: int = 12,
    multimodal_num_hidden_layers: int = 6,
    multimodal_dropout: float = 0.0,
    multimodal_intermediate_size: int = 3072,
    multimodal_intermediate_activation: Callable[..., nn.Module] = nn.GELU,
    multimodal_layer_norm_eps: float = 1e-12,
    # projection
    text_and_image_proj_size: int = 768,
    pretrained: bool = False,
    **kwargs: Any,
) -> FLAVAModel:
    image_encoder = flava_image_encoder(
        hidden_size=image_hidden_size,
        num_attention_heads=image_num_attention_heads,
        num_hidden_layers=image_num_hidden_layers,
        use_image_masking=use_image_masking,
        dropout=image_dropout,
        intermediate_size=image_intermediate_size,
        intermediate_activation=image_intermediate_activation,
        layer_norm_eps=image_layer_norm_eps,
        image_size=image_size,
        patch_size=patch_size,
        num_channels=num_channels,
    )
    text_encoder = flava_text_encoder(
        hidden_size=text_hidden_size,
        num_attention_heads=text_num_attention_heads,
        num_hidden_layers=text_num_hidden_layers,
        dropout=text_dropout,
        intermediate_size=text_intermediate_size,
        intermediate_activation=text_intermediate_activation,
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
        dropout=multimodal_dropout,
        intermediate_size=multimodal_intermediate_size,
        intermediate_activation=multimodal_intermediate_activation,
        layer_norm_eps=multimodal_layer_norm_eps,
    )

    image_to_mm_projection = nn.Linear(image_hidden_size, multimodal_hidden_size)
    text_to_mm_projection = nn.Linear(text_hidden_size, multimodal_hidden_size)

    image_projection = nn.Linear(image_hidden_size, text_and_image_proj_size)
    text_projection = nn.Linear(text_hidden_size, text_and_image_proj_size)

    flava = FLAVAModel(
        image_encoder=image_encoder,
        text_encoder=text_encoder,
        mm_encoder=mm_encoder,
        image_to_mm_projection=image_to_mm_projection,
        text_to_mm_projection=text_to_mm_projection,
        text_projection=text_projection,
        image_projection=image_projection,
    )

    if pretrained:
        load_module_from_url(flava, FLAVA_MODEL_MAPPING[CKPT_KEY])

    return flava


def flava_model_for_pretraining(
    codebook_image_size: int = 112,
    pretrained: bool = False,
    **flava_model_kwargs: Any,
    # TODO: Add parameters for loss here
) -> FLAVAForPreTraining:
    model = flava_model(**flava_model_kwargs)
    hidden_size = flava_model_kwargs.get("multimodal_hidden_size", 768)
    losses = FLAVAPretrainingLoss(hidden_size=hidden_size)
    codebook = DalleVAEEncoder(image_size=codebook_image_size)

    flava = FLAVAForPreTraining(
        model=model,
        image_codebook=codebook,
        loss=losses,
    )

    if pretrained:
        load_module_from_url(flava, FLAVA_FOR_PRETRAINED_MAPPING[CKPT_KEY])

    return flava


def flava_model_for_classification(
    num_classes: int,
    classifier_in_dim: int = 768,
    classifier_hidden_sizes: Union[int, List[int]] = 768,
    classifier_dropout: float = 0.5,
    classifier_activation: Callable[..., nn.Module] = nn.ReLU,
    classifier_normalization: Optional[Callable[..., nn.Module]] = None,
    loss_fn: Optional[Callable[..., Tensor]] = None,
    pretrained: bool = True,
    **flava_model_kwargs: Any,
) -> FLAVAForClassification:

    classifier = MLP(
        in_dim=classifier_in_dim,
        out_dim=num_classes,
        hidden_dims=classifier_hidden_sizes,
        dropout=classifier_dropout,
        activation=classifier_activation,
        normalization=classifier_normalization,
    )
    model = flava_model(**flava_model_kwargs)
    if loss_fn is None:
        loss_fn = nn.CrossEntropyLoss()

    classification_model = FLAVAForClassification(
        model=model, classifier=classifier, loss=loss_fn
    )

    if pretrained:
        load_module_from_url(
            classification_model,
            FLAVA_FOR_PRETRAINED_MAPPING[CKPT_KEY],
            strict=False,
        )
    return classification_model


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


class DalleVAEEncoder(nn.Module):
    def __init__(
        self, image_size: Union[int, Tuple[int, int]] = 112, pretrained: bool = True
    ):
        super().__init__()
        self.image_size = image_size
        self.encoder = DalleEncoder()
        if pretrained:
            self.load_model()

    def load_model(self) -> Any:  # type: ignore
        # TODO (T116682215): Network error due to FLAVA model relying on access to openAI

        encoder_state_dict = torch.hub.load_state_dict_from_url(
            "https://cdn.openai.com/dall-e/encoder.pkl"
        )
        self.encoder.load_state_dict(encoder_state_dict.state_dict())  # type: ignore
        return self.state_dict()

    def get_codebook_indices(self, images: Tensor) -> Tensor:
        z_logits = self.encoder(images)
        return torch.argmax(z_logits, axis=1)  # type: ignore

    def get_codebook_probs(self, images: Tensor) -> Tensor:
        z_logits = self.encoder(images)
        return nn.Softmax(dim=1)(z_logits)

    def forward(self, img_seq_prob: Tensor) -> Tensor:
        return self.get_codebook_indices(img_seq_prob)
