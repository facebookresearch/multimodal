# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
from test.test_utils import assert_expected, set_rng_seed
from torch import nn, Tensor
from torchmultimodal.models.albef import ALBEFModel, ALBEFSimilarity


class TestALBEFModel:
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
                assert_expected(param, param_m, rtol=0, atol=1e-4)
                assert not param_m.requires_grad

    def test_dequeue_and_enqueue(self):
        image_feat_m = torch.randn(2, 2)
        text_feat_m = torch.randn(2, 2)
        self.albef._dequeue_and_enqueue(image_feat_m, text_feat_m)
        assert_expected(
            self.albef.image_queue[:, 0:2], image_feat_m.T, rtol=0, atol=1e-4
        )
        assert_expected(self.albef.text_queue[:, 0:2], text_feat_m.T, rtol=0, atol=1e-4)

    def test_momentum_update(self):
        init_weight = Tensor([[1, 2, 3], [4, 5, 6]])
        init_weight_m = Tensor([[6, 5, 4], [3, 2, 1]])
        self.albef.models[0].weight = nn.Parameter(init_weight)
        self.albef.models_m[0].weight = nn.Parameter(init_weight_m)
        self.albef._momentum_update()
        expected_weight_m = Tensor([[5.9750, 4.9850, 3.9950], [3.0050, 2.0150, 1.0250]])
        assert_expected(self.albef.models[0].weight, init_weight, rtol=0, atol=1e-4)
        assert_expected(
            self.albef.models_m[0].weight, expected_weight_m, rtol=0, atol=1e-4
        )

    def test_similarity(self):
        set_rng_seed(0)
        self.albef.image_queue = torch.randn(2, 4)
        self.albef.text_queue = torch.randn(2, 4)
        image_feat = torch.randn(2, 2)
        text_feat = torch.randn(2, 2)
        image_feat_m = torch.randn(2, 2)
        text_feat_m = torch.randn(2, 2)
        output = self.albef._similarity(
            image_feat, text_feat, image_feat_m, text_feat_m
        )
        expected_sim_i2t = Tensor(
            [
                [-3.660729, -11.191917, 1.252719, 9.129601, -0.882724, -0.818697],
                [15.755546, 21.687363, 27.634123, -18.587852, 30.696188, -4.964930],
            ]
        )
        expected_sim_t2i = Tensor(
            [
                [27.311251, -23.288742, -14.411922, -30.653456, -3.136197, 20.444725],
                [16.565296, -26.082125, 8.684321, -19.420963, -24.787359, 16.908016],
            ]
        )
        expected_sim_i2t_m = Tensor(
            [
                [-13.028821, -22.274969, -17.438065, 18.764980, -20.974815, 2.714235],
                [8.504787, 5.272466, 22.941040, -5.002890, 23.104834, -4.742508],
            ]
        )
        expected_sim_t2i_m = Tensor(
            [
                [-13.028821, 8.504787, 10.671893, 14.442708, -3.490067, -8.771053],
                [-22.274969, 5.272466, 31.752522, 24.050060, -23.705740, -11.501695],
            ]
        )
        assert_expected(output.sim_i2t, expected_sim_i2t, rtol=0, atol=1e-4)
        assert_expected(output.sim_t2i, expected_sim_t2i, rtol=0, atol=1e-4)
        assert_expected(output.sim_i2t_m, expected_sim_i2t_m, rtol=0, atol=1e-4)
        assert_expected(output.sim_t2i_m, expected_sim_t2i_m, rtol=0, atol=1e-4)

    def test_neg_embeddings(self):
        set_rng_seed(0)
        image_embeds = torch.randn(2, 1, 3)
        text_embeds = torch.randn(2, 1, 3)
        text_atts = torch.randn(2, 1)
        similarity = ALBEFSimilarity(
            sim_i2t=torch.randn(2, 5),
            sim_t2i=torch.randn(2, 5),
            sim_i2t_m=torch.randn(2, 5),
            sim_t2i_m=torch.randn(2, 5),
        )
        image_embeds_neg, text_embeds_neg, text_atts_neg = self.albef._neg_embeddings(
            image_embeds, text_embeds, text_atts, similarity
        )
        expected_image_embeds_neg = Tensor(
            [[0.568431, -1.084522, -1.398595], [1.540996, -0.293429, -2.178789]]
        ).unsqueeze(1)
        expected_text_embeds_neg = Tensor(
            [[-0.403344, -0.596635, 0.182036], [0.403347, 0.838026, -0.719258]]
        ).unsqueeze(1)
        expected_text_atts_neg = Tensor([1.100604, -0.856675]).unsqueeze(1)
        assert_expected(image_embeds_neg, expected_image_embeds_neg, rtol=0, atol=1e-4)
        assert_expected(text_embeds_neg, expected_text_embeds_neg, rtol=0, atol=1e-4)
        assert_expected(text_atts_neg, expected_text_atts_neg, rtol=0, atol=1e-4)
