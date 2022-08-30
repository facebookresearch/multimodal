# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch

from test.test_utils import assert_expected, set_rng_seed
from torchmultimodal.models.mdetr.model import (
    mdetr_for_phrase_grounding,
    mdetr_for_vqa,
    mdetr_resnet101,
)
from torchmultimodal.utils.common import remove_grad


class TestMDETR:
    @pytest.fixture(autouse=True)
    def rng(self):
        set_rng_seed(0)

    @pytest.fixture(autouse=True)
    def batch_size(self):
        return 2

    @pytest.fixture(autouse=True)
    def num_queries(self):
        return 100

    @pytest.fixture(autouse=True)
    def num_classes(self):
        return 255

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
    def mdetr(self, num_queries, num_classes):
        mdetr = mdetr_resnet101(num_queries=num_queries, num_classes=num_classes)
        mdetr.eval()
        return mdetr

    def test_mdetr_model(
        self,
        mdetr,
        test_tensors,
        input_ids,
        batch_size,
        num_queries,
        num_classes,
    ):
        out = mdetr(test_tensors, input_ids)
        logits_actual = out.pred_logits
        boxes_actual = out.pred_boxes
        logits_expected = torch.Tensor(
            [
                -0.0145,
                0.0121,
                0.0270,
                0.0310,
                0.0072,
                -0.0002,
                0.0100,
                0.0012,
                0.0290,
                0.0067,
            ]
        )
        boxes_expected = torch.Tensor(
            [
                0.4896,
                0.4898,
                0.4897,
                0.4900,
                0.4894,
                0.4895,
                0.4897,
                0.4908,
                0.4902,
                0.4899,
            ]
        )
        assert logits_actual.size() == (
            batch_size,
            num_queries,
            num_classes + 1,
        )
        assert boxes_actual.size() == (batch_size, num_queries, 4)
        assert_expected(logits_actual[1, :10, 1], logits_expected, rtol=0, atol=1e-3)
        assert_expected(boxes_actual[1, :10, 1], boxes_expected, rtol=0, atol=1e-3)

    @pytest.fixture(autouse=True)
    def contrastive_dim(self):
        return 64

    @pytest.fixture()
    def mdetr_model_for_phrase_grounding(self, contrastive_dim):
        return mdetr_for_phrase_grounding(contrastive_dim=contrastive_dim)

    def test_mdetr_model_for_phrase_grounding(
        self,
        mdetr_model_for_phrase_grounding,
        test_tensors,
        input_ids,
        batch_size,
        num_queries,
        contrastive_dim,
    ):
        out = mdetr_model_for_phrase_grounding(test_tensors, input_ids)
        logits_actual = out.model_output.pred_logits
        boxes_actual = out.model_output.pred_boxes

        logits_expected = torch.Tensor(
            [
                -0.1245,
                -0.5103,
                0.2710,
                -0.2171,
                -0.0561,
                0.2635,
                0.2804,
                -0.0415,
                0.2091,
                0.0110,
            ]
        )
        boxes_expected = torch.Tensor(
            [
                0.4789,
                0.4920,
                0.4898,
                0.4905,
                0.4765,
                0.4794,
                0.4932,
                0.4683,
                0.4845,
                0.4789,
            ]
        )

        assert_expected(logits_actual[1, :10, 1], logits_expected, rtol=0, atol=1e-3)
        assert_expected(boxes_actual[1, :10, 1], boxes_expected, rtol=0, atol=1e-3)

        query_embeddings_actual = out.contrastive_embeddings.query_embeddings
        token_embeddings_actual = out.contrastive_embeddings.token_embeddings
        query_embeddings_expected = torch.Tensor(
            [
                0.3083,
                0.3146,
                0.3221,
                0.2411,
                0.2673,
                0.3152,
                0.2798,
                0.2321,
                0.2433,
                0.2321,
            ]
        )
        token_embeddings_expected = torch.Tensor(
            [
                0.2002,
                0.1153,
                0.1196,
                0.2104,
                0.1716,
                0.1975,
                0.1587,
                0.1740,
                0.1350,
                0.1383,
            ]
        )

        assert query_embeddings_actual.size() == (
            batch_size,
            num_queries,
            contrastive_dim,
        )
        assert token_embeddings_actual.size() == (
            batch_size,
            input_ids.size()[1],
            contrastive_dim,
        )

        assert_expected(
            query_embeddings_actual[1, :10, 1],
            query_embeddings_expected,
            rtol=0,
            atol=1e-3,
        )
        assert_expected(
            token_embeddings_actual[1, :10, 1],
            token_embeddings_expected,
            rtol=0,
            atol=1e-3,
        )

    @pytest.fixture()
    def mdetr_model_for_vqa(self):
        model = mdetr_for_vqa()
        model.eval()
        remove_grad(model)
        return model

    def test_mdetr_model_for_vqa(
        self,
        mdetr_model_for_vqa,
        test_tensors,
        input_ids,
        batch_size,
        num_queries,
    ):
        out = mdetr_model_for_vqa(test_tensors, input_ids)
        # logits_actual = out.model_output.pred_logits
        # boxes_actual = out.model_output.pred_boxes

        # logits_expected = torch.Tensor(

        # )
        # boxes_expected = torch.Tensor(

        # )

        # assert_expected(logits_actual[1, :10, 1], logits_expected, rtol=0, atol=1e-3)
        # assert_expected(boxes_actual[1, :10, 1], boxes_expected, rtol=0, atol=1e-3)

        # query_embeddings_actual = out.contrastive_embeddings.query_embeddings
        # token_embeddings_actual = out.contrastive_embeddings.token_embeddings
        # query_embeddings_expected = torch.Tensor(

        # )
        # token_embeddings_expected = torch.Tensor(

        # )

        # assert_expected(
        #     query_embeddings_actual[1, :10, 1],
        #     query_embeddings_expected,
        #     rtol=0,
        #     atol=1e-3,
        # )
        # assert_expected(
        #     token_embeddings_actual[1, :10, 1],
        #     token_embeddings_expected,
        #     rtol=0,
        #     atol=1e-3,
        # )

        # Finally, check the vqa heads
        answer_type_actual = out.vqa_preds["answer_type"]
        answer_obj_actual = out.vqa_preds["answer_obj"]
        answer_rel_actual = out.vqa_preds["answer_rel"]
        answer_attr_actual = out.vqa_preds["answer_attr"]
        answer_cat_actual = out.vqa_preds["answer_cat"]
        answer_global_actual = out.vqa_preds["answer_global"]

        answer_type_expected = torch.Tensor(
            [-0.5345, -0.5067, -0.4841, -0.7165, -0.1300]
        )
        answer_obj_expected = torch.Tensor([0.7397, -0.7879, -1.1478])
        answer_rel_expected = torch.Tensor(
            [
                0.1138,
                -0.0270,
                0.5583,
                -0.0701,
                0.5942,
                -0.5173,
                0.2762,
                -0.7870,
                -0.3243,
                0.1833,
            ]
        )
        answer_attr_expected = torch.Tensor(
            [
                0.1531,
                -0.5439,
                0.9922,
                1.1380,
                -0.3717,
                -0.1283,
                -0.3008,
                0.7216,
                -0.2271,
                -0.0351,
            ]
        )
        answer_cat_expected = torch.Tensor(
            [
                0.1644,
                -0.1254,
                -0.2684,
                0.3595,
                -0.0538,
                0.3129,
                -1.0193,
                -0.4543,
                0.4901,
                -1.1946,
            ]
        )
        answer_global_expected = torch.Tensor(
            [
                0.8541,
                -0.0703,
                -0.2690,
                0.5855,
                1.7249,
                0.3501,
                -0.3301,
                -0.5751,
                -0.1189,
                0.8937,
            ]
        )

        assert_expected(
            answer_type_actual[1],
            answer_type_expected,
            rtol=0,
            atol=1e-3,
        )
        assert_expected(
            answer_obj_actual[1],
            answer_obj_expected,
            rtol=0,
            atol=1e-3,
        )
        assert_expected(
            answer_rel_actual[1, :10],
            answer_rel_expected,
            rtol=0,
            atol=1e-3,
        )
        assert_expected(
            answer_attr_actual[1, :10],
            answer_attr_expected,
            rtol=0,
            atol=1e-3,
        )
        assert_expected(
            answer_cat_actual[1, :10],
            answer_cat_expected,
            rtol=0,
            atol=1e-3,
        )
        assert_expected(
            answer_global_actual[1, :10],
            answer_global_expected,
            rtol=0,
            atol=1e-3,
        )
