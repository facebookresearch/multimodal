# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from copy import deepcopy

import pytest
import torch
from examples.mdetr.data.postprocessors import PostProcessFlickr
from examples.mdetr.utils.box_ops import box_cxcywh_to_xyxy
from test.test_utils import assert_expected, set_rng_seed


@pytest.fixture(scope="class")
def random():
    set_rng_seed(0)


class TestBoxTransformUtil:
    @pytest.fixture(scope="class")
    def coords(self):
        return torch.Tensor([1.0, 2.0, 3.0, 4.0])

    def test_invalid_inputs(self, coords):
        invalid_coords = torch.concat([coords, torch.Tensor([1])])
        with pytest.raises(ValueError):
            box_cxcywh_to_xyxy(invalid_coords)

    def test_valid_inputs(self, coords):
        actual = box_cxcywh_to_xyxy(coords)
        expected = torch.Tensor([-0.5, 0.0, 2.5, 4])
        assert_expected(actual, expected)


class TestFlickrPostProcessor:
    @pytest.fixture(scope="class")
    def batch_size(self):
        return 2

    @pytest.fixture(scope="class")
    def n_queries(self):
        return 5

    @pytest.fixture(scope="class")
    def n_classes(self):
        return 7

    @pytest.fixture(scope="class")
    def max_seq_len(self):
        return 8

    @pytest.fixture(scope="class")
    def pred_logits(self, random, batch_size, n_queries, n_classes):
        return torch.randn((batch_size, n_queries, n_classes + 1))

    @pytest.fixture(scope="class")
    def pred_boxes(self, random, batch_size, n_queries):
        return torch.rand((batch_size, n_queries, 4))

    @pytest.fixture(scope="class")
    def target_sizes(self):
        return torch.Tensor([[100, 100], [50, 50]])

    @pytest.fixture(scope="class")
    def n_tokens(self):
        return [[2, 3, 4], [2, 2, 2, 3]]

    @pytest.fixture(scope="class")
    def items_per_batch_element(self, n_tokens):
        return [len(x) for x in n_tokens]

    @pytest.fixture(scope="class")
    def starting_indices(self, random, n_classes, n_tokens):
        return [
            torch.randint(0, n_classes + 1 - max(tok), (len(tok),)) for tok in n_tokens
        ]

    @pytest.fixture(scope="class")
    def pos_map(self, n_tokens, starting_indices, n_classes):
        def _construct_test_pos_map_for_sample(n_toks, starting_indices, max_length):
            assert len(n_toks) == len(
                starting_indices
            ), "n_toks and starting_indices must have same length"
            out = torch.zeros((len(n_toks), max_length))
            idx_list = []
            for i, (n_tok, starting_idx) in enumerate(zip(n_toks, starting_indices)):
                r = torch.arange(starting_idx, starting_idx + n_tok).unsqueeze(-1)
                idx_list.append(torch.cat([i * torch.ones_like(r), r], dim=-1))
            indices = torch.cat(idx_list)
            out.index_put_(tuple(indices.t()), torch.Tensor([1]))
            out = out / out.sum(axis=1).unsqueeze(-1)
            return out

        assert len(n_tokens) == len(
            starting_indices
        ), "n_toks and starting_indices must have same length"
        bs = len(n_tokens)
        pos_map = [
            _construct_test_pos_map_for_sample(
                n_tokens[i], starting_indices[i], n_classes + 1
            )
            for i in range(bs)
        ]
        return pos_map

    @pytest.fixture(scope="class")
    def batched_pos_map(self, n_tokens, n_classes, pos_map):
        n_boxes = sum([len(x) for x in n_tokens])
        batched_pos_map = torch.zeros((n_boxes, n_classes + 1), dtype=torch.bool)
        cur_count = 0
        for sample in pos_map:
            batched_pos_map[
                cur_count : cur_count + len(sample), : sample.shape[1]
            ] = sample
            cur_count += len(sample)
        assert cur_count == len(batched_pos_map)

        return batched_pos_map

    @pytest.fixture(scope="class")
    def transform(self):
        transform = PostProcessFlickr()
        return transform

    def test_invalid_inputs(
        self,
        transform,
        pred_logits,
        pred_boxes,
        target_sizes,
        pos_map,
        batched_pos_map,
        items_per_batch_element,
    ):
        with pytest.raises(TypeError):
            _ = transform(
                outputs={"pred_logits": pred_logits, "pred_boxes": pred_boxes},
                target_sizes=target_sizes,
                positive_map=pos_map,
                items_per_batch_element=items_per_batch_element,
            )
        with pytest.raises(AssertionError):
            incorrect_items_per_batch_element = deepcopy(items_per_batch_element)
            incorrect_items_per_batch_element[-1] -= 1
            _ = transform(
                outputs={"pred_logits": pred_logits, "pred_boxes": pred_boxes},
                target_sizes=target_sizes,
                positive_map=batched_pos_map,
                items_per_batch_element=incorrect_items_per_batch_element,
            )

    def test_valid_inputs(
        self,
        transform,
        pred_logits,
        pred_boxes,
        target_sizes,
        batched_pos_map,
        items_per_batch_element,
        batch_size,
        n_queries,
    ):
        actual = transform(
            outputs={"pred_logits": pred_logits, "pred_boxes": pred_boxes},
            target_sizes=target_sizes,
            positive_map=batched_pos_map,
            items_per_batch_element=items_per_batch_element,
        )
        assert len(actual) == batch_size
        assert len(actual[0]) == items_per_batch_element[0]
        assert len(actual[1]) == items_per_batch_element[1]
        assert len(actual[0][0]) == n_queries
        assert len(actual[0][0][0]) == 4
        # Corresponds to out[0][1][1]
        expected_first_sample_val = torch.Tensor([85.535, 59.915, 86.045, 77.485])
        # Corresponds to out[1][2][1]
        expected_second_sample_val = torch.Tensor([41.47, 32.49, 55.57, 51.2])

        assert_expected(
            torch.Tensor(actual[0][1][1]),
            expected_first_sample_val,
            rtol=0.0,
            atol=1e-2,
        )
        assert_expected(
            torch.Tensor(actual[1][2][1]),
            expected_second_sample_val,
            rtol=0.0,
            atol=1e-2,
        )


@pytest.fixture(scope="class")
def pos_map(construct_test_pos_map):
    return construct_test_pos_map()
