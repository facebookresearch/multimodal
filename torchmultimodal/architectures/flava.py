# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from collections import namedtuple
from functools import partial
from typing import Any, Callable, List, Literal, Optional, Tuple

import torch
from torch import nn, Tensor
from torchmultimodal.modules.layers.transformer import FLAVATransformerOutput
from torchmultimodal.utils.common import PretrainedMixin

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


class FLAVAArchitecture(nn.Module, PretrainedMixin):
    def __init__(
        self,
        image_encoder: nn.Module,
        text_encoder: nn.Module,
        mm_encoder: nn.Module,
        image_to_mm_projection: nn.Module,
        text_to_mm_projection: nn.Module,
        **kwargs: Any,
    ):
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
        encode_callable: Callable[..., Tuple[Tensor, Tensor]],
    ) -> Optional[FLAVATransformerOutput]:
        output = FLAVATransformerOutput()

        if data is not None and selected_head_encoder in encoder_options:
            output = encode_callable(data)

        return output

    def encode_mm(
        self,
        image_embedding: Tensor,
        text_embedding: Tensor,
    ):
        if image_embedding is None or text_embedding is None:
            # Since nothing is passed, it might be case without
            # masked data let's say.
            return FLAVATransformerOutput()

        image_embedding = self.image_to_mm_projection(image_embedding)
        text_embedding = self.text_to_mm_projection(text_embedding)
        fused_state = torch.cat([image_embedding, text_embedding], dim=1)
        return self.mm_encoder(fused_state)
