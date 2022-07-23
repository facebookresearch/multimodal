# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any

import torch

from examples.mugen.retrieval.video_clip import videoclip
from pytorch_lightning import LightningModule
from torchmultimodal.modules.losses.contrastive_loss_with_temperature import (
    ContrastiveLossWithTemperature,
)


class VideoCLIPLightningModule(LightningModule):
    """PyTorch Lightning module for evaluating VideoCLIP model.

    Args:
        logit_scale (float): Initial log-temperature value for contrastive loss funtion.
            Defaults to ``3.8282``, MUGEN's log-temperature value after training.
        logit_scale_max (float): Maximum log-temperature value for contrastive loss function.
            Defaults to ``100``, MUGEN's maximum log-temperature value.
        learning_rate (float): optimizer learning rate.
            Defaults to ``1e-3``, MUGEN's learning rate.
        weight_decay (float): optimizer weight decay.
            Defaults to ``1e-3``, MUGEN's weight decay.
        **videoclip_kwargs (Any): Keyword arguments for the videoCLIP model builder.
    """

    def __init__(
        self,
        logit_scale: float = 3.8282,
        logit_scale_max: float = 100,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-3,
        **videoclip_kwargs: Any,
    ):
        super().__init__()
        self.model = videoclip(**videoclip_kwargs)
        self.contrastive_loss = ContrastiveLossWithTemperature(
            logit_scale=logit_scale,
            logit_scale_min=None,
            logit_scale_max=logit_scale_max,
        )
        self.lr = learning_rate
        self.weight_decay = weight_decay

    def configure_optimizers(self):
        params = self.parameters()
        optimizer = torch.optim.AdamW(
            params, lr=self.lr, weight_decay=self.weight_decay
        )
        return optimizer

    def training_step(self, batch, batch_idx):
        text, video = batch.get("text"), batch.get("video")
        model_output = self.model(features_a=text, features_b=video)
        loss = self.contrastive_loss(
            model_output.embeddings_a, model_output.embeddings_b
        )
        self.log(
            "train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        return loss

    def validation_step(self, batch, batch_idx):
        text, video = batch.get("text"), batch.get("video")
        model_output = self.model(features_a=text, features_b=video)
        loss = self.contrastive_loss(
            model_output.embeddings_a, model_output.embeddings_b
        )
        self.log(
            "validation/loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        return loss

    def test_step(self, batch, batch_idx):
        text, video = batch.get("text"), batch.get("video")
        model_output = self.model(features_a=text, features_b=video)
        loss = self.contrastive_loss(
            model_output.embeddings_a, model_output.embeddings_b
        )
        return loss
