# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest

import torch
from test.test_utils import assert_expected, set_rng_seed
from torchmultimodal.architectures.clip import CLIPArchitecture


class TestCLIPArchitecture:
    @pytest.fixture
    def start(self):
        set_rng_seed(1234)

        query_encoder = torch.nn.Linear(5, 3)
        retrieval_encoder = torch.nn.Linear(4, 3)
        encoders = torch.nn.ModuleDict(
            {"query": query_encoder, "retrieval": retrieval_encoder}
        )
        clip = CLIPArchitecture(encoders=encoders)

        input_query = torch.randint(1, 8, (2, 5), dtype=torch.float)
        input_retrieval = torch.randint(1, 8, (2, 4), dtype=torch.float)

        return clip, input_query, input_retrieval

    def test_forward(self, start):
        clip, input_query, input_retrieval = start
        assert isinstance(clip, torch.nn.Module)

        out = clip(modalities={"query": input_query, "retrieval": input_retrieval})
        assert (
            hasattr(out, "query_embeddings")
            and hasattr(out, "retrieval_embeddings")
            and len(out.__dict__) == 2
        )

        actual_q_embedding = out.query_embeddings
        actual_r_embedding = out.retrieval_embeddings
        expected_q_embedding = torch.Tensor(
            [[-0.8066, -0.1749, 0.5647], [-0.7709, -0.1118, 0.6271]]
        )
        expected_r_embedding = torch.Tensor(
            [[-0.1719, 0.7932, 0.5842], [-0.2805, 0.8761, -0.3921]]
        )
        assert_expected(
            actual=actual_q_embedding, expected=expected_q_embedding, rtol=0, atol=1e-4
        )
        assert_expected(
            actual=actual_r_embedding, expected=expected_r_embedding, rtol=0, atol=1e-4
        )

    def test_forward_missing_input(self, start):
        clip, input_query, _ = start
        assert isinstance(clip, torch.nn.Module)

        with pytest.raises(AssertionError):
            clip(modalities={"query": input_query})

    def test_forward_extra_input(self, start):
        clip, input_query, input_retrieval = start
        assert isinstance(clip, torch.nn.Module)

        with pytest.warns(UserWarning):
            out = clip(
                modalities={
                    "query": input_query,
                    "retrieval": input_retrieval,
                    "extra": torch.Tensor([1]).to(dtype=float),
                }
            )

        assert (
            hasattr(out, "query_embeddings")
            and hasattr(out, "retrieval_embeddings")
            and len(out.__dict__) == 2
        )
