# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch

from test.test_utils import assert_expected, set_rng_seed
from torchmultimodal.models.mdetr import mdetr_resnet101, MDETRTransformer


class TestMDETR:
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
    def text_dim(self):
        return 5

    @pytest.fixture(autouse=True)
    def mm_dim(self):
        return 11

    @pytest.fixture(autouse=True)
    def embedding_dim(self):
        return 256

    @pytest.fixture(autouse=True)
    def num_decoder_layers(self):
        return 6

    @pytest.fixture(autouse=True)
    def num_queries_full(self):
        return 100

    @pytest.fixture(autouse=True)
    def num_classes_full(self):
        return 255

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
    def text_memory(self, text_dim, batch_size, embedding_dim):
        return torch.randn(text_dim, batch_size, embedding_dim)

    @pytest.fixture()
    def text_memory_key_padding_mask(self, batch_size, text_dim):
        return torch.randint(0, 2, (batch_size, text_dim))

    @pytest.fixture()
    def memory_key_padding_mask(self, batch_size, mm_dim):
        return torch.randint(0, 2, (batch_size, mm_dim))

    @pytest.fixture()
    def query_pos(self, num_queries, batch_size, embedding_dim):
        return torch.randn(num_queries, batch_size, embedding_dim)

    @pytest.fixture()
    def test_tensors(self):
        return torch.rand(2, 3, 64, 64).unbind(dim=0)

    @pytest.fixture()
    def input_ids(self):
        return torch.Tensor(
            [
                [0, 100, 64, 192, 5, 3778, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [
                    0,
                    1708,
                    190,
                    114,
                    38,
                    1395,
                    192,
                    5,
                    3778,
                    6,
                    38,
                    216,
                    14,
                    24,
                    8785,
                    2,
                ],
            ]
        ).to(dtype=torch.long)

    @pytest.fixture()
    def transformer(self, embedding_dim, num_decoder_layers):
        transformer = MDETRTransformer(
            d_model=embedding_dim, num_decoder_layers=num_decoder_layers
        )
        transformer.eval()
        return transformer

    @pytest.fixture()
    def mdetr(self, num_queries_full, num_classes_full):
        mdetr = mdetr_resnet101(
            num_queries=num_queries_full, num_classes=num_classes_full
        )
        mdetr.eval()
        return mdetr

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

    def test_full_mdetr_model(
        self,
        mdetr,
        test_tensors,
        input_ids,
        batch_size,
        num_queries_full,
        num_classes_full,
    ):
        out = mdetr(test_tensors, input_ids)
        logits_actual = out.pred_logits
        boxes_actual = out.pred_boxes
        logits_expected = torch.Tensor(
            [
                -0.8136,
                -0.8156,
                -0.8094,
                -0.8099,
                -0.8226,
                -0.8106,
                -0.8104,
                -0.8207,
                -0.8172,
                -0.8063,
            ]
        )
        boxes_expected = torch.Tensor(
            [
                0.5612,
                0.5623,
                0.5615,
                0.5617,
                0.5620,
                0.5614,
                0.5617,
                0.5611,
                0.5615,
                0.5620,
            ]
        )
        assert logits_actual.size() == (
            batch_size,
            num_queries_full,
            num_classes_full + 1,
        )
        assert boxes_actual.size() == (batch_size, num_queries_full, 4)
        assert_expected(logits_actual[1, :10, 1], logits_expected, rtol=0, atol=1e-3)
        assert_expected(boxes_actual[1, :10, 1], boxes_expected, rtol=0, atol=1e-3)
