# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from pytorch_lightning import LightningModule
from torchmultimodal.models.flava import flava_model_for_pretraining
from transformers.optimization import AdamW, get_cosine_schedule_with_warmup


class FLAVALightningModule(LightningModule):
    def __init__(
        self,
        learning_rate: float = 0.0002,
        adam_eps: float = 1.0e-08,
        adam_weight_decay: float = 0.01,
        warmup_steps: int = 2000,
        max_steps: int = 450000,
    ):
        super().__init__()
        self.model = flava_model_for_pretraining()
        self.learning_rate = learning_rate
        self.adam_eps = adam_eps
        self.adam_weight_decay = adam_weight_decay
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps

    # TODO: Setup validation loop
    def training_step(self, batch, batch_idx):
        if "image" in batch and ("text" in batch or "text_masked" in batch):
            required_embedding = "mm"
        elif "image" in batch:
            required_embedding = "image"
        elif "text" in batch or "text_masked" in batch:
            required_embedding = "text"
        else:
            raise RuntimeError("Batch needs to have either or both 'image' and 'text'.")

        output = self.model(
            image=batch.get("image", None),
            image_for_codebook=batch.get("image_for_codebook", None),
            image_patches_mask=batch.get("image_patches_mask", None),
            text=batch.get("text", None),
            text_masked=batch.get("text_masked", None),
            mlm_labels=batch.get("mlm_labels", None),
            itm_labels=batch.get("itm_labels", None),
            required_embedding=required_embedding,
        )
        loss = sum(value for value in output.values())
        for key in output:
            self.log(f"losses/{key}", output[key], prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        optimizer = AdamW(
            self.parameters(),
            lr=self.learning_rate,
            eps=self.adam_eps,
            weight_decay=self.adam_weight_decay,
        )
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.warmup_steps,
            num_training_steps=self.max_steps,
        )
        return [optimizer], [{"lr_scheduler": scheduler, "interval": "step"}]
