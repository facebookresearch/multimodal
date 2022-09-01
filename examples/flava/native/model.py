# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Tuple

import torch
from torch import nn
from torchmultimodal.models.flava.model import flava_model_for_pretraining
from transformers.optimization import get_cosine_schedule_with_warmup


def get_optimizer(
    model: torch.nn.Module,
    learning_rate: float = 0.0002,
    adam_eps: float = 1.0e-08,
    adam_weight_decay: float = 0.01,
    adam_betas: Tuple[int, int] = (0.9, 0.999),
    warmup_steps: int = 2000,
    max_steps: int = 450000,
):
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        betas=adam_betas,
        eps=adam_eps,
        weight_decay=adam_weight_decay,
    )
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=max_steps,
    )
    return optimizer, scheduler


class FLAVAPreTrainModule(nn.Module):
    def __init__(
        self,
        use_bf16: bool = True,
        **flava_pretraining_kwargs: Any,
    ):
        super().__init__()
        self.model = flava_model_for_pretraining(**flava_pretraining_kwargs)
        self.use_bf16 = use_bf16

    def forward(self, batch, action=None):
        # super hacky
        if action == "encode_text":
            return self.model.encode_text(batch)
        elif action == "encode_image":
            return self.model.encode_image(batch)

        if "image" in batch and ("text" in batch or "text_masked" in batch):
            required_embedding = "mm"
        elif "image" in batch:
            required_embedding = "image"
        elif "text" in batch or "text_masked" in batch:
            required_embedding = "text"
        else:
            raise RuntimeError("Batch needs to have either or both 'image' and 'text'.")

        output = self.model(
            image=batch.get("image"),
            image_for_codebook=batch.get("image_for_codebook"),
            image_patches_mask=batch.get("image_patches_mask"),
            text=batch.get("text"),
            text_masked=batch.get("text_masked"),
            mlm_labels=batch.get("mlm_labels"),
            itm_labels=batch.get("itm_labels"),
            required_embedding=required_embedding,
        )
        return output

    def encode_text(self, *args, **kwargs):
        return self.model.encode_text(*args, **kwargs)
