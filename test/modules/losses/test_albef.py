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
        sim_i2t_m = torch.randn(2, 3)
        sim_t2i_m = torch.randn(2, 3)
        with pytest.raises(RuntimeError):
            self.loss(sim_i2t, sim_t2i, sim_i2t_m, sim_t2i_m).item()

    def test_itc_loss_invalid_sim_m(self):
        sim_i2t = torch.randn(2, 3)
        sim_t2i = torch.randn(2, 3)
        sim_i2t_m = torch.randn(2, 4)  # all inputs should be the same size
        sim_t2i_m = torch.randn(2, 3)
        with pytest.raises(RuntimeError):
            self.loss(sim_i2t, sim_t2i, sim_i2t_m, sim_t2i_m).item()

    def test_itc_loss_invalid_sim_target(self):
        sim_i2t = torch.randn(2, 3)
        sim_t2i = torch.randn(2, 3)
        sim_i2t_m = torch.randn(2, 3)
        sim_t2i_m = torch.randn(2, 3)
        sim_targets = torch.randn(2, 4)  # all inputs should be the same size
        with pytest.raises(RuntimeError):
            self.loss(sim_i2t, sim_t2i, sim_i2t_m, sim_t2i_m, sim_targets).item()

    def test_itc_loss_without_sim_targets(self):
        sim_i2t = torch.randn(2, 3)
        sim_t2i = torch.randn(2, 3)
        sim_i2t_m = torch.randn(2, 3)
        sim_t2i_m = torch.randn(2, 3)
        output = self.loss(sim_i2t, sim_t2i, sim_i2t_m, sim_t2i_m).item()
        expected = 1.160506
        assert_expected(output, expected, rtol=0, atol=1e-4)

    def test_itc_loss_with_sim_targets(self):
        sim_i2t = torch.randn(2, 3)
        sim_t2i = torch.randn(2, 3)
        sim_i2t_m = torch.randn(2, 3)
        sim_t2i_m = torch.randn(2, 3)
        sim_targets = torch.randn(2, 3)
        output = self.loss(sim_i2t, sim_t2i, sim_i2t_m, sim_t2i_m, sim_targets).item()
        expected = -1.928954
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
        self.loss = MaskedLanguageModelingLoss(hidden_size=3)
        self.loss_with_distillation = MaskedLanguageModelingLoss(
            hidden_size=3, alpha=0.4
        )

    def test_mlm_loss_invalid_labels(self):
        # labels dimensions should match the first two dimensions of the embeddings
        labels = torch.randint(10, (2, 6))
        embeddings = torch.randn(2, 5, 3)
        with pytest.raises(ValueError):
            self.loss(labels, embeddings)

    def test_mlm_loss_invalid_embeddings(self):
        labels = torch.randint(10, (2, 5))
        # embeddings hidden size (dim 2) should match the hidden size of MaskedLanguageModelingLoss
        embeddings = torch.randn(2, 5, 4)
        with pytest.raises(RuntimeError):
            self.loss(labels, embeddings)

    def test_mlm_loss_missing_momentum_embeddings(self):
        # need momentum embeddings input for MaskedLanguageModelingLoss with nonzero alpha
        labels = torch.randint(10, (2, 5))
        embeddings = torch.randn(2, 5, 3)
        with pytest.raises(AssertionError):
            self.loss_with_distillation(labels, embeddings)

    def test_mlm_loss(self):
        labels = torch.randint(10, (2, 5))
        embeddings = torch.randn(2, 5, 3)
        output = self.loss(labels, embeddings)
        expected = Tensor([43.486340, 41.509407])
        assert_expected(output, expected, rtol=0, atol=1e-4)

    def test_mlm_loss_with_distillation(self):
        labels = torch.randint(10, (2, 5))
        embeddings = torch.randn(2, 5, 3)
        embeddings_m = torch.randn(2, 5, 3)
        output = self.loss_with_distillation(labels, embeddings, embeddings_m)
        expected = Tensor([43.015320, 41.552132])
        assert_expected(output, expected, rtol=0, atol=1e-4)
