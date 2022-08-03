# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Tuple

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from torchmetrics import Accuracy
from torchmultimodal.models.flava.flava_model import (
    flava_model_for_classification,
    flava_model_for_pretraining,
)
from transformers.optimization import get_cosine_schedule_with_warmup
from flava.native.bfoptimizer import BFOptimizer


def get_optimizer(
    model: torch.nn.Module,
    learning_rate: float = 0.0002,
    adam_eps: float = 1.0e-08,
    adam_weight_decay: float = 0.01,
    adam_betas: Tuple[int, int] = (0.9, 0.999),
    warmup_steps: int = 2000,
    max_steps: int = 450000,
    use_bf16: bool = True,
):
    if use_bf16:
        print("using bf16")
        optimizer = BFOptimizer(
            model.parameters(),
            lr=learning_rate,
            betas=adam_betas,
            eps=adam_eps,
            weight_decay=adam_weight_decay,
        )
    else:
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

def get_batch_value(batch, key, use_bf16=True):
    val = batch.get(key, None)
    if val is None:
        return
    if val.dtype == torch.float32 and use_bf16:
        return val.to(dtype=torch.bfloat16)
    return val


class FLAVAPreTrainModule(nn.Module):
    def __init__(
        self,
        use_bf16: bool = True,
        **flava_pretraining_kwargs: Any,
    ):
        super().__init__()
        self.model = flava_model_for_pretraining(**flava_pretraining_kwargs)
        self.use_bf16 = use_bf16

    def forward(self, batch):
        if "image" in batch and ("text" in batch or "text_masked" in batch):
            required_embedding = "mm"
        elif "image" in batch:
            required_embedding = "image"
        elif "text" in batch or "text_masked" in batch:
            required_embedding = "text"
        else:
            raise RuntimeError("Batch needs to have either or both 'image' and 'text'.")

        output = self.model(
            image=get_batch_value(batch, "image",self.use_bf16),
            image_for_codebook=get_batch_value(batch, "image_for_codebook",self.use_bf16),
            image_patches_mask=get_batch_value(batch, "image_patches_mask",self.use_bf16),
            text=get_batch_value(batch, "text",self.use_bf16),
            text_masked=get_batch_value(batch, "text_masked",self.use_bf16),
            mlm_labels=get_batch_value(batch, "mlm_labels",self.use_bf16),
            itm_labels=get_batch_value(batch, "itm_labels",self.use_bf16),
            required_embedding=required_embedding,
        )
        return output


class FLAVAClassificationModel(nn.Module):
    def __init__(
        self,
        num_classes: int,
        **flava_classification_kwargs: Any,
    ):
        super().__init__()
        self.model = flava_model_for_classification(
            num_classes, **flava_classification_kwargs
        )
        self.metrics = Accuracy()

    def forward(self, batch, batch_idx):
        if "image" in batch and ("text" in batch or "text_masked" in batch):
            required_embedding = "mm"
        elif "image" in batch:
            required_embedding = "image"
        elif "text" in batch or "text_masked" in batch:
            required_embedding = "text"
        else:
            raise RuntimeError("Batch needs to have either or both 'image' and 'text'.")

        labels = batch["labels"]
        output = self.model(
            image=batch.get("image", None),
            text=batch.get("text", None),
            required_embedding=required_embedding,
            labels=labels,
        )

        accuracy = self.metrics(output.logits, labels)

        return output, accuracy
