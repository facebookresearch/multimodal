# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import pytest
import torch
from test.test_utils import assert_expected, set_rng_seed
from torchmultimodal.modules.losses.albef import (
    ImageTextContrastiveLoss,
    ImageTextMatchingLoss,
    MaskedLanguageModelingLoss,
)


@pytest.fixture(autouse=True)
def set_seed():
    set_rng_seed(0)


@pytest.fixture(autouse=True)
def itc_loss_module():
    return ImageTextContrastiveLoss()


@pytest.fixture(autouse=True)
def itm_loss_module():
    return ImageTextMatchingLoss(hidden_size=3)


@pytest.fixture(autouse=True)
def mlm_loss_module():
    return MaskedLanguageModelingLoss(hidden_size=3)


def test_itc_loss_invalid_sim(itc_loss_module):
    sim_i2t = torch.randn(2, 4)  # all inputs should be the same size
    sim_t2i = torch.randn(2, 3)
    sim_i2t_m = torch.randn(2, 3)
    sim_t2i_m = torch.randn(2, 3)
    with pytest.raises(RuntimeError):
        itc_loss_module(sim_i2t, sim_t2i, sim_i2t_m, sim_t2i_m).item()


def test_itc_loss_invalid_sim_m(itc_loss_module):
    sim_i2t = torch.randn(2, 3)
    sim_t2i = torch.randn(2, 3)
    sim_i2t_m = torch.randn(2, 4)  # all inputs should be the same size
    sim_t2i_m = torch.randn(2, 3)
    with pytest.raises(RuntimeError):
        itc_loss_module(sim_i2t, sim_t2i, sim_i2t_m, sim_t2i_m).item()


def test_itc_loss_invalid_sim_target(itc_loss_module):
    sim_i2t = torch.randn(2, 3)
    sim_t2i = torch.randn(2, 3)
    sim_i2t_m = torch.randn(2, 3)
    sim_t2i_m = torch.randn(2, 3)
    sim_targets = torch.randn(2, 4)  # all inputs should be the same size
    with pytest.raises(RuntimeError):
        itc_loss_module(sim_i2t, sim_t2i, sim_i2t_m, sim_t2i_m, sim_targets).item()


def test_itc_loss_without_sim_targets(itc_loss_module):
    sim_i2t = torch.randn(2, 3)
    sim_t2i = torch.randn(2, 3)
    sim_i2t_m = torch.randn(2, 3)
    sim_t2i_m = torch.randn(2, 3)
    output = itc_loss_module(sim_i2t, sim_t2i, sim_i2t_m, sim_t2i_m).item()
    expected = 1.160506
    assert_expected(output, expected, rtol=0, atol=1e-4)


def test_itc_loss_with_sim_targets(itc_loss_module):
    sim_i2t = torch.randn(2, 3)
    sim_t2i = torch.randn(2, 3)
    sim_i2t_m = torch.randn(2, 3)
    sim_t2i_m = torch.randn(2, 3)
    sim_targets = torch.randn(2, 3)
    output = itc_loss_module(sim_i2t, sim_t2i, sim_i2t_m, sim_t2i_m, sim_targets).item()
    expected = -1.928954
    assert_expected(output, expected, rtol=0, atol=1e-4)


def test_itm_loss_invalid_input_batch_size(itm_loss_module):
    # embeddings_neg's batch size (dim 0) should be double of embeddings_pos's batch size
    embeddings_pos = torch.randn(2, 5, 3)
    embeddings_neg = torch.randn(2, 5, 3)
    with pytest.raises(ValueError):
        itm_loss_module(embeddings_pos, embeddings_neg)


def test_itm_loss_invalid_input_hidden_size(itm_loss_module):
    # embeddings hidden size (dim 2) should match the hidden size of itm_loss_module
    embeddings_pos = torch.randn(2, 5, 4)
    embeddings_neg = torch.randn(4, 5, 4)
    with pytest.raises(RuntimeError):
        itm_loss_module(embeddings_pos, embeddings_neg)


def test_itm_loss(itm_loss_module):
    embeddings_pos = torch.randn(2, 5, 3)
    embeddings_neg = torch.randn(4, 5, 3)
    output = itm_loss_module(embeddings_pos, embeddings_neg).item()
    expected = 0.813487
    assert_expected(output, expected, rtol=0, atol=1e-4)


def test_mlm_loss_invalid_labels(mlm_loss_module):
    # labels dimensions should match the first two dimensions of the embeddings
    labels = torch.randint(10, (2, 6))
    embeddings = torch.randn(2, 5, 3)
    embeddings_m = torch.randn(2, 5, 3)
    weights = torch.randn(2)
    with pytest.raises(RuntimeError):
        mlm_loss_module(labels, embeddings, embeddings_m, weights)


def test_mlm_loss_invalid_embeddings(mlm_loss_module):
    labels = torch.randint(10, (2, 5))
    # embeddings hidden size (dim 2) should match the hidden size of mlm_loss_module
    embeddings = torch.randn(2, 5, 4)
    embeddings_m = torch.randn(2, 5, 4)
    weights = torch.randn(2)
    with pytest.raises(RuntimeError):
        mlm_loss_module(labels, embeddings, embeddings_m, weights)


def test_mlm_loss_invalid_weights(mlm_loss_module):
    labels = torch.randint(10, (2, 5))
    embeddings = torch.randn(2, 5, 3)
    embeddings_m = torch.randn(2, 5, 3)
    weights = torch.randn(3)  # all inputs should have the same batch size (dim 0)
    with pytest.raises(RuntimeError):
        mlm_loss_module(labels, embeddings, embeddings_m, weights)


def test_mlm_loss(mlm_loss_module):
    labels = torch.randint(10, (2, 5))
    embeddings = torch.randn(2, 5, 3)
    embeddings_m = torch.randn(2, 5, 3)
    weights = torch.randn(2)
    output = mlm_loss_module(labels, embeddings, embeddings_m, weights).item()
    expected = 13.533014
    assert_expected(output, expected, rtol=0, atol=1e-4)
