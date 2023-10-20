# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch

from examples.mugen.generation.video_vqvae import video_vqvae_mugen
from tests.test_utils import assert_expected, set_rng_seed


@pytest.fixture(autouse=True)
def random():
    set_rng_seed(4)


@pytest.fixture(scope="module")
def params():
    in_channel_dims = (2, 2)
    out_channel_dims = (2, 2)
    kernel_sizes = ((2, 2, 2), (2, 2, 2))
    strides = ((1, 1, 1), (1, 1, 1))
    return in_channel_dims, out_channel_dims, kernel_sizes, strides


@pytest.fixture(scope="module")
def input_tensor():
    return torch.ones(1, 2, 2, 2, 2)


class TestVideoVQVAEMUGEN:
    @pytest.fixture
    def vv(self):
        def create_model(model_key=None, freeze_model=False):
            model = video_vqvae_mugen(
                pretrained_model_key=model_key, freeze_model=freeze_model
            )
            model.eval()
            return model

        return create_model

    @pytest.fixture
    def input_data(self):
        def create_data(seq_len):
            return torch.randn(1, 3, seq_len, 256, 256)

        return create_data

    def test_invalid_pretrained_model_key(self, vv):
        with pytest.raises(KeyError):
            vv("invalid_key")

    def test_freeze_model(self, vv):
        model = vv("mugen_L32", freeze_model=True)
        for param in model.parameters():
            assert param.requires_grad is False

    @pytest.mark.parametrize(
        "seq_len,expected", [(8, -16005.1113), (16, 51112.2266), (32, 450732.6875)]
    )
    def test_forward(self, vv, input_data, seq_len, expected):
        x = input_data(seq_len)
        model = vv()
        output = model(x)
        actual = output.decoded
        # ensure embed is uninitialized
        assert model.codebook._is_embedding_init is False
        assert_expected(actual.shape, (1, 3, seq_len, 256, 256))
        assert_expected(actual.sum().item(), expected, rtol=1e-4, atol=1e-4)

    @pytest.mark.parametrize(
        "seq_len,expected", [(8, 132017.28125), (16, -109636.0), (32, 1193122.0)]
    )
    def test_checkpoint(self, vv, input_data, seq_len, expected):
        x = input_data(seq_len)
        model_key = f"mugen_L{seq_len}"
        model = vv(model_key)
        # ensure embed is initialized
        assert model.codebook._is_embedding_init
        output = model(x)
        actual_tensor = torch.sum(output.decoded)
        expected_tensor = torch.tensor(expected)
        assert_expected(actual_tensor, expected_tensor, rtol=1e-5, atol=1e-8)
