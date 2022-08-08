# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch

from test.test_utils import assert_expected, set_rng_seed
from torchmultimodal.models.mdetr.transformer import MDETRTransformer


class TestMDETRTransformer:
    @pytest.fixture(autouse=True)
    def rng(self):
        set_rng_seed(0)

    @pytest.fixture(autouse=True)
    def batch_size(self):
        return 2

    @pytest.fixture(autouse=True)
    def num_queries(self):
        return 4

    @pytest.fixture(autouse=True)
    def mm_dim(self):
        return 11

    @pytest.fixture(autouse=True)
    def embedding_dim(self):
        return 256

    @pytest.fixture(autouse=True)
    def num_decoder_layers(self):
        return 6

    @pytest.fixture()
    def src(self, mm_dim, batch_size, embedding_dim):
        return torch.randn(mm_dim, batch_size, embedding_dim)

    @pytest.fixture()
    def src_key_padding_mask(self, batch_size, mm_dim):
        return torch.randint(0, 2, (batch_size, mm_dim))

    @pytest.fixture()
    def pos(self, mm_dim, batch_size, embedding_dim):
        return torch.randn(mm_dim, batch_size, embedding_dim)

    @pytest.fixture()
    def tgt(self, num_queries, batch_size, embedding_dim):
        return torch.randn(num_queries, batch_size, embedding_dim)

    @pytest.fixture()
    def memory(self, mm_dim, batch_size, embedding_dim):
        return torch.randn(mm_dim, batch_size, embedding_dim)

    @pytest.fixture()
    def memory_key_padding_mask(self, batch_size, mm_dim):
        return torch.randint(0, 2, (batch_size, mm_dim))

    @pytest.fixture()
    def query_pos(self, num_queries, batch_size, embedding_dim):
        return torch.randn(num_queries, batch_size, embedding_dim)

    @pytest.fixture()
    def transformer(self, embedding_dim, num_decoder_layers):
        transformer = MDETRTransformer(
            d_model=embedding_dim, num_decoder_layers=num_decoder_layers
        )
        transformer.eval()
        return transformer

    def test_transformer_encoder(
        self,
        transformer,
        src,
        src_key_padding_mask,
        pos,
        mm_dim,
        batch_size,
        embedding_dim,
    ):
        actual = transformer.encoder(
            src=src, src_key_padding_mask=src_key_padding_mask, pos=pos
        )
        assert actual.size() == (mm_dim, batch_size, embedding_dim)
        expected = torch.Tensor([0.3924, 1.7622])
        assert_expected(actual[1, :, 1], expected, rtol=0, atol=1e-3)

    def test_transformer_decoder(
        self,
        transformer,
        tgt,
        memory,
        memory_key_padding_mask,
        pos,
        query_pos,
        num_decoder_layers,
        num_queries,
        batch_size,
        embedding_dim,
    ):
        actual = transformer.decoder(
            tgt=tgt,
            memory=memory,
            memory_key_padding_mask=memory_key_padding_mask,
            pos=pos,
            query_pos=query_pos,
        )
        assert actual.size() == (
            num_decoder_layers,
            num_queries,
            batch_size,
            embedding_dim,
        )
        expected = torch.Tensor(
            [[-1.6592, 0.3761], [-2.5531, 0.7192], [-1.2693, 0.3763], [-1.1510, 0.1224]]
        )
        assert_expected(actual[1, :, :, 1], expected, rtol=0, atol=1e-3)
