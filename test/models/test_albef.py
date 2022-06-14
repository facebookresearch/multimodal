# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
from test.test_utils import assert_expected, set_rng_seed
from torch import nn, Tensor
from torchmultimodal.models.albef import ALBEFModel


class TestALBEFModel:
    set_rng_seed(0)
    albef = ALBEFModel(
        nn.Linear(3, 2),
        nn.Linear(3, 2),
        nn.Linear(3, 2),
        nn.Linear(3, 2),
        nn.Linear(3, 2),
        embed_dim=2,
        queue_size=4,
    )

    def test_copy_params_momentum_models(self):
        self.albef.models_m = [nn.Linear(3, 2) for _ in range(5)]
        self.albef._copy_params_momentum_models()
        for model, model_m in zip(self.albef.models, self.albef.models_m):
            for param, param_m in zip(model.parameters(), model_m.parameters()):
                assert_expected(param, param_m)
                assert not param_m.requires_grad

    def test_dequeue_and_enqueue(self):
        image_feat_m = torch.randn(2, 2)
        text_feat_m = torch.randn(2, 2)
        self.albef._dequeue_and_enqueue(image_feat_m, text_feat_m)
        assert_expected(self.albef.image_queue[:, 0:2], image_feat_m.T)
        assert_expected(self.albef.text_queue[:, 0:2], text_feat_m.T)

    def test_momentum_update(self):
        init_weight = Tensor([[1, 2, 3], [4, 5, 6]])
        init_weight_m = Tensor([[6, 5, 4], [3, 2, 1]])
        self.albef.models[0].weight = nn.Parameter(init_weight)
        self.albef.models_m[0].weight = nn.Parameter(init_weight_m)
        self.albef._momentum_update()
        expected_weight_m = Tensor([[5.9750, 4.9850, 3.9950], [3.0050, 2.0150, 1.0250]])
        assert_expected(self.albef.models[0].weight, init_weight)
        assert_expected(self.albef.models_m[0].weight, expected_weight_m)
