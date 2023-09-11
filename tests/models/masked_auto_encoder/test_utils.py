# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

import pytest
import torch
from torch.optim import AdamW
from torchmultimodal.models.masked_auto_encoder.model import audio_mae
from torchmultimodal.models.masked_auto_encoder.utils import (
    CosineWithWarmupAndLRScaling,
    get_param_groups_with_layer_decay,
    get_param_groups_with_weight_decay,
)
from torchmultimodal.modules.encoders.vision_transformer import vision_transformer


class TestCosineWithWarmupAndLRScaling:
    @pytest.fixture
    def scheduler(self):
        optimizer = AdamW(
            [
                {"params": torch.ones(1), "lr_scale": 0.5},
                {"params": torch.ones(1), "lr_scale": 0.2},
            ],
            lr=0.01,
        )
        scheduler = CosineWithWarmupAndLRScaling(
            optimizer=optimizer, warmup_iters=4, max_iters=10
        )
        return scheduler

    def test_step(self, scheduler):
        optimizer = scheduler.optimizer
        expected_lrs = [
            (0.00125, 0.0005),
            (0.0025, 0.001),
            (0.00375, 0.0015),
            (0.005, 0.002),
            (0.00467, 0.00187),
            (0.00375, 0.0015),
            (0.0025, 0.001),
            (0.00125, 0.0005),
            (0.000335, 0.000134),
            (0.0, 0.0),
        ]
        for i in range(10):
            optimizer.step()
            scheduler.step()

            assert optimizer.param_groups[0]["lr"] == pytest.approx(
                expected_lrs[i][0], abs=0.0001
            )
            assert optimizer.param_groups[1]["lr"] == pytest.approx(
                expected_lrs[i][1], abs=0.0001
            )

    def test_default_lr_scale(self):
        optimizer = AdamW(
            [
                {"params": torch.ones(1)},
                {"params": torch.ones(1)},
            ],
            lr=0.01,
        )
        scheduler = CosineWithWarmupAndLRScaling(
            optimizer=optimizer, warmup_iters=4, max_iters=10
        )
        expected_lrs = [
            (0.0025, 0.0025),
            (0.005, 0.005),
            (0.0075, 0.0075),
            (0.01, 0.01),
            (0.00933, 0.00933),
            (0.0075, 0.0075),
            (0.005, 0.005),
            (0.0025, 0.0025),
            (0.00067, 0.00067),
            (0.0, 0.0),
        ]
        for i in range(10):
            optimizer.step()
            scheduler.step()
            assert optimizer.param_groups[0]["lr"] == pytest.approx(
                expected_lrs[i][0], abs=0.0001
            )
            assert optimizer.param_groups[1]["lr"] == pytest.approx(
                expected_lrs[i][1], abs=0.0001
            )


class TestGetParamGroupsWithLayerDecay:
    def test_param_group(self):
        model = vision_transformer(
            patch_size=2, hidden_dim=2, dim_feedforward=8, n_layer=2, n_head=1
        )
        model.embeddings.conv_projection.bias.requires_grad = False
        lr = 0.001
        weight_decay = 0.2
        lr_decay = 0.5
        pg = get_param_groups_with_layer_decay(
            model, lr=lr, weight_decay=weight_decay, layer_decay=lr_decay
        )
        # decay and no decay for first 3 layers + no decay for final ln
        assert len(pg) == 7

        embed_no_decay_params = pg["no_decay_0"]
        assert embed_no_decay_params["param_names"] == [
            "embeddings.cls_token",
            "embeddings.position_embeddings",
        ]
        assert embed_no_decay_params["lr"] == lr
        assert embed_no_decay_params["weight_decay"] == 0.0
        assert embed_no_decay_params["lr_scale"] == lr_decay**3

        embed_decay_params = pg["decay_0"]
        assert embed_decay_params["param_names"] == [
            "embeddings.conv_projection.weight",
        ]
        assert embed_decay_params["lr"] == lr
        assert embed_decay_params["weight_decay"] == weight_decay
        assert embed_decay_params["lr_scale"] == lr_decay**3

        layer_1_no_decay = pg["no_decay_1"]
        assert layer_1_no_decay["param_names"] == [
            "encoder.layer.0.attention.input_proj.bias",
            "encoder.layer.0.attention.output_proj.bias",
            "encoder.layer.0.feedforward.model.0.bias",
            "encoder.layer.0.feedforward.model.2.bias",
            "encoder.layer.0.attention_layernorm.weight",
            "encoder.layer.0.attention_layernorm.bias",
            "encoder.layer.0.feedforward_layernorm.weight",
            "encoder.layer.0.feedforward_layernorm.bias",
        ]
        assert layer_1_no_decay["lr"] == lr
        assert layer_1_no_decay["weight_decay"] == 0.0
        assert layer_1_no_decay["lr_scale"] == lr_decay**2

        layer_1_decay = pg["decay_1"]
        assert layer_1_decay["param_names"] == [
            "encoder.layer.0.attention.input_proj.weight",
            "encoder.layer.0.attention.output_proj.weight",
            "encoder.layer.0.feedforward.model.0.weight",
            "encoder.layer.0.feedforward.model.2.weight",
        ]
        assert layer_1_decay["lr"] == lr
        assert layer_1_decay["weight_decay"] == weight_decay
        assert layer_1_decay["lr_scale"] == lr_decay**2

        layer_2_no_decay = pg["no_decay_2"]
        assert layer_2_no_decay["param_names"] == [
            "encoder.layer.1.attention.input_proj.bias",
            "encoder.layer.1.attention.output_proj.bias",
            "encoder.layer.1.feedforward.model.0.bias",
            "encoder.layer.1.feedforward.model.2.bias",
            "encoder.layer.1.attention_layernorm.weight",
            "encoder.layer.1.attention_layernorm.bias",
            "encoder.layer.1.feedforward_layernorm.weight",
            "encoder.layer.1.feedforward_layernorm.bias",
        ]
        assert layer_2_no_decay["lr"] == lr
        assert layer_2_no_decay["weight_decay"] == 0.0
        assert layer_2_no_decay["lr_scale"] == lr_decay**1

        layer_2_decay = pg["decay_2"]
        assert layer_2_decay["param_names"] == [
            "encoder.layer.1.attention.input_proj.weight",
            "encoder.layer.1.attention.output_proj.weight",
            "encoder.layer.1.feedforward.model.0.weight",
            "encoder.layer.1.feedforward.model.2.weight",
        ]
        assert layer_2_decay["lr"] == lr
        assert layer_2_decay["weight_decay"] == weight_decay
        assert layer_2_decay["lr_scale"] == lr_decay**1

        layer_3_no_decay = pg["no_decay_3"]
        assert layer_3_no_decay["param_names"] == [
            "encoder.final_layer_norm.weight",
            "encoder.final_layer_norm.bias",
        ]
        assert layer_3_no_decay["lr"] == lr
        assert layer_3_no_decay["weight_decay"] == 0.0
        assert layer_3_no_decay["lr_scale"] == 1


class TestGetParamGroupsWithWeightDecay:
    def test_param_group(self):
        model = audio_mae(encoder_layers=1, decoder_layers=1)
        pg = get_param_groups_with_weight_decay(model, weight_decay=0.2)
        assert len(pg) == 2
        assert len(pg[0]["params"]) == 26
        assert pg[0]["weight_decay"] == 0.0
        assert len(pg[1]["params"]) == 15
        assert pg[1]["weight_decay"] == 0.2
