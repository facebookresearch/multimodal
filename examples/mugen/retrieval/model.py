# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any

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
        **videoclip_kwargs (Any): Keyword arguments for the videoCLIP model builder.
    """

    def __init__(
        self,
        logit_scale: float = 3.8282,
        logit_scale_max: float = 100,
        **videoclip_kwargs: Any,
    ):
        super().__init__()
        self.model = videoclip(**videoclip_kwargs)
        self.contrastive_loss = ContrastiveLossWithTemperature(
            logit_scale=logit_scale,
            logit_scale_min=None,
            logit_scale_max=logit_scale_max,
        )

    def test_step(self, batch, batch_idx):
        text, video = batch.get("text"), batch.get("video")
        model_output = self.model(features_a=text, features_b=video)
        loss = self.contrastive_loss(
            model_output.embeddings_a, model_output.embeddings_b
        )
        return loss
