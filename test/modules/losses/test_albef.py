# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import pytest
import torch
from test.test_utils import assert_expected, set_rng_seed
from torch import Tensor
from torchmultimodal.modules.losses.albef import (
    ImageTextContrastiveLoss,
    ImageTextMatchingLoss,
    MaskedLanguageModelingLoss,
)


class TestImageTextContrastiveLoss:
    @pytest.fixture(autouse=True)
    def setup(self):
        set_rng_seed(0)
        self.loss = ImageTextContrastiveLoss()

    def test_itc_loss_invalid_sim(self):
        sim_i2t = torch.randn(2, 4)  # all inputs should be the same size
        sim_t2i = torch.randn(2, 3)
        with pytest.raises(RuntimeError):
            self.loss(sim_i2t, sim_t2i)

    def test_itc_loss_missing_sim_m(self):
        # need momentum similarity inputs for ImageTextContrastiveLoss with nonzero alpha
        sim_i2t = torch.randn(2, 3)
        sim_t2i = torch.randn(2, 3)
        with pytest.raises(AssertionError):
            self.loss(sim_i2t, sim_t2i, alpha=0.4)

    def test_itc_loss_invalid_sim_m(self):
        sim_i2t = torch.randn(2, 3)
        sim_t2i = torch.randn(2, 3)
        sim_i2t_m = torch.randn(2, 4)  # all inputs should be the same size
        sim_t2i_m = torch.randn(2, 3)
        with pytest.raises(RuntimeError):
            self.loss(sim_i2t, sim_t2i, sim_i2t_m, sim_t2i_m, alpha=0.4)

    def test_itc_loss_invalid_sim_target(self):
        sim_i2t = torch.randn(2, 3)
        sim_t2i = torch.randn(2, 3)
        sim_targets = torch.randn(2, 4)  # all inputs should be the same size
        with pytest.raises(RuntimeError):
            self.loss(sim_i2t, sim_t2i, sim_targets=sim_targets)

    def test_itc_loss_without_distillation(self):
        sim_i2t = torch.randn(2, 3)
        sim_t2i = torch.randn(2, 3)
        output = self.loss(sim_i2t, sim_t2i).item()
        expected = 1.160506
        assert_expected(output, expected, rtol=0, atol=1e-4)

    def test_itc_loss_with_distillation(self):
        sim_i2t = torch.randn(2, 3)
        sim_t2i = torch.randn(2, 3)
        sim_i2t_m = torch.randn(2, 3)
        sim_t2i_m = torch.randn(2, 3)
        output = self.loss(sim_i2t, sim_t2i, sim_i2t_m, sim_t2i_m, alpha=0.4).item()
        expected = 1.341230
        assert_expected(output, expected, rtol=0, atol=1e-4)

    def test_itc_loss_with_sim_targets(self):
        sim_i2t = torch.randn(2, 3)
        sim_t2i = torch.randn(2, 3)
        sim_i2t_m = torch.randn(2, 3)
        sim_t2i_m = torch.randn(2, 3)
        sim_targets = torch.randn(2, 3)
        output = self.loss(
            sim_i2t, sim_t2i, sim_i2t_m, sim_t2i_m, sim_targets, alpha=0.4
        ).item()
        expected = -0.512445
        assert_expected(output, expected, rtol=0, atol=1e-4)


class TestImageTextMatchingLoss:
    @pytest.fixture(autouse=True)
    def setup(self):
        set_rng_seed(0)
        self.loss = ImageTextMatchingLoss(hidden_size=3)

    def test_itm_loss_invalid_input_hidden_size(self):
        # embeddings hidden size (dim 2) should match the hidden size of ImageTextMatchingLoss
        embeddings_pos = torch.randn(2, 4)
        embeddings_neg = torch.randn(4, 4)
        with pytest.raises(RuntimeError):
            self.loss(embeddings_pos, embeddings_neg)

    def test_itm_loss(self):
        embeddings_pos = torch.randn(2, 3)
        embeddings_neg = torch.randn(4, 3)
        output = self.loss(embeddings_pos, embeddings_neg).item()
        expected = 0.860578
        assert_expected(output, expected, rtol=0, atol=1e-4)


class TestMaskedLanguageModelingLoss:
    @pytest.fixture(autouse=True)
    def setup(self):
        set_rng_seed(0)
        self.loss = MaskedLanguageModelingLoss()

    def test_mlm_loss_invalid_labels(self):
        # labels dimensions should match the first two dimensions of prediction_scores
        labels = torch.randint(10, (2, 6))
        prediction_scores = torch.randn(2, 5, 20)
        with pytest.raises(ValueError):
            self.loss(labels, prediction_scores)

    def test_mlm_loss_missing_momentum_embeddings(self):
        # need prediction_scores_m input for MaskedLanguageModelingLoss with nonzero alpha
        labels = torch.randint(10, (2, 5))
        prediction_scores = torch.randn(2, 5, 20)
        alpha = 0.4
        with pytest.raises(AssertionError):
            self.loss(labels, prediction_scores, alpha=alpha)

    def test_mlm_loss(self):
        labels = torch.randint(10, (2, 5))
        prediction_scores = torch.randn(2, 5, 20)
        output = self.loss(labels, prediction_scores)
        expected = Tensor([14.552961, 14.930183])
        assert_expected(output, expected, rtol=0, atol=1e-4)

    def test_mlm_loss_with_distillation(self):
        labels = torch.randint(10, (2, 5))
        prediction_scores = torch.randn(2, 5, 20)
        prediction_scores_m = torch.randn(2, 5, 20)
        alpha = 0.4
        output = self.loss(labels, prediction_scores, prediction_scores_m, alpha)
        expected = Tensor([14.367424, 14.541029])
        assert_expected(output, expected, rtol=0, atol=1e-4)
