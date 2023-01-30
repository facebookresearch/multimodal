# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch
from tests.test_utils import assert_expected, set_rng_seed
from torchmultimodal.models.clip import model as pretrained


class TestCLIPCheckpoint:
    @pytest.fixture(scope="class")
    def data(self):
        set_rng_seed(0)
        image224 = torch.randn(1, 3, 224, 224)
        image288 = torch.randn(1, 3, 288, 288)
        image384 = torch.randn(1, 3, 384, 384)
        image448 = torch.randn(1, 3, 448, 448)
        text = torch.randint(0, 49408, (1, 77))
        return text, image224, image288, image384, image448

    def test_clip_vit_b16(self, data):
        text, image224, *_ = data
        model = pretrained.clip_vit_b16(True)
        model.eval()
        with torch.no_grad():
            actual_a_embedding, actual_b_embedding = model(image224, text)

        assert_expected(
            actual=actual_a_embedding.mean(),
            expected=torch.tensor(0.0030),
            rtol=0,
            atol=1e-4,
        )

        assert_expected(
            actual=actual_b_embedding.mean(),
            expected=torch.tensor(0.0023),
            rtol=0,
            atol=1e-4,
        )

        assert_expected(
            actual=torch.tensor(actual_a_embedding.size()),
            expected=torch.tensor((1, 512)),
        )

        assert_expected(
            actual=torch.tensor(actual_b_embedding.size()),
            expected=torch.tensor((1, 512)),
        )

    def test_clip_vit_b32(self, data):
        text, image224, *_ = data
        model = pretrained.clip_vit_b32(True)
        model.eval()
        with torch.no_grad():
            actual_a_embedding, actual_b_embedding = model(image224, text)

        assert_expected(
            actual=actual_a_embedding.mean(),
            expected=torch.tensor(-0.0014),
            rtol=0,
            atol=1e-4,
        )

        assert_expected(
            actual=actual_b_embedding.mean(),
            expected=torch.tensor(-0.0041),
            rtol=0,
            atol=1e-4,
        )

        assert_expected(
            actual=torch.tensor(actual_a_embedding.size()),
            expected=torch.tensor((1, 512)),
        )

        assert_expected(
            actual=torch.tensor(actual_b_embedding.size()),
            expected=torch.tensor((1, 512)),
        )

    def test_clip_vit_l14(self, data):
        text, image224, *_ = data
        model = pretrained.clip_vit_l14(True)
        model.eval()
        with torch.no_grad():
            actual_a_embedding, actual_b_embedding = model(image224, text)

        assert_expected(
            actual=actual_a_embedding.mean(),
            expected=torch.tensor(0.0006),
            rtol=0,
            atol=1e-4,
        )

        assert_expected(
            actual=actual_b_embedding.mean(),
            expected=torch.tensor(-0.0022),
            rtol=0,
            atol=1e-4,
        )

        assert_expected(
            actual=torch.tensor(actual_a_embedding.size()),
            expected=torch.tensor((1, 768)),
        )

        assert_expected(
            actual=torch.tensor(actual_b_embedding.size()),
            expected=torch.tensor((1, 768)),
        )

    def test_clip_rn50(self, data):
        text, image224, *_ = data
        model = pretrained.clip_rn50(True)
        model.eval()
        with torch.no_grad():
            actual_a_embedding, actual_b_embedding = model(image224, text)

        assert_expected(
            actual=actual_a_embedding.mean(),
            expected=torch.tensor(-0.0012),
            rtol=0,
            atol=1e-4,
        )

        assert_expected(
            actual=actual_b_embedding.mean(),
            expected=torch.tensor(-0.0001),
            rtol=0,
            atol=1e-4,
        )

        assert_expected(
            actual=torch.tensor(actual_a_embedding.size()),
            expected=torch.tensor((1, 1024)),
        )

        assert_expected(
            actual=torch.tensor(actual_b_embedding.size()),
            expected=torch.tensor((1, 1024)),
        )

    def test_clip_rn101(self, data):
        text, image224, *_ = data
        model = pretrained.clip_rn101(True)
        model.eval()
        with torch.no_grad():
            actual_a_embedding, actual_b_embedding = model(image224, text)

        assert_expected(
            actual=actual_a_embedding.mean(),
            expected=torch.tensor(-0.0012),
            rtol=0,
            atol=1e-4,
        )

        assert_expected(
            actual=actual_b_embedding.mean(),
            expected=torch.tensor(-0.0017),
            rtol=0,
            atol=1e-4,
        )

        assert_expected(
            actual=torch.tensor(actual_a_embedding.size()),
            expected=torch.tensor((1, 512)),
        )

        assert_expected(
            actual=torch.tensor(actual_b_embedding.size()),
            expected=torch.tensor((1, 512)),
        )

    def test_clip_rn50x4(self, data):
        text, _, image288, *_ = data
        model = pretrained.clip_rn50x4(True)
        model.eval()
        with torch.no_grad():
            actual_a_embedding, actual_b_embedding = model(image288, text)

        assert_expected(
            actual=actual_a_embedding.mean(),
            expected=torch.tensor(0.0006),
            rtol=0,
            atol=1e-4,
        )

        assert_expected(
            actual=actual_b_embedding.mean(),
            expected=torch.tensor(0.0002),
            rtol=0,
            atol=1e-4,
        )

        assert_expected(
            actual=torch.tensor(actual_a_embedding.size()),
            expected=torch.tensor((1, 640)),
        )

        assert_expected(
            actual=torch.tensor(actual_b_embedding.size()),
            expected=torch.tensor((1, 640)),
        )

    def test_clip_rn50x16(self, data):
        text, *_, image384, _ = data
        model = pretrained.clip_rn50x16(True)
        model.eval()
        with torch.no_grad():
            actual_a_embedding, actual_b_embedding = model(image384, text)

        assert_expected(
            actual=actual_a_embedding.mean(),
            expected=torch.tensor(0.0017),
            rtol=0,
            atol=1e-4,
        )

        assert_expected(
            actual=actual_b_embedding.mean(),
            expected=torch.tensor(0.0012),
            rtol=0,
            atol=1e-4,
        )

        assert_expected(
            actual=torch.tensor(actual_a_embedding.size()),
            expected=torch.tensor((1, 768)),
        )

        assert_expected(
            actual=torch.tensor(actual_b_embedding.size()),
            expected=torch.tensor((1, 768)),
        )

    def test_clip_rn50x64(self, data):
        text, *_, image448 = data
        model = pretrained.clip_rn50x64(True)
        model.eval()
        with torch.no_grad():
            actual_a_embedding, actual_b_embedding = model(image448, text)

        assert_expected(
            actual=actual_a_embedding.mean(),
            expected=torch.tensor(0.0004),
            rtol=0,
            atol=1e-4,
        )

        assert_expected(
            actual=actual_b_embedding.mean(),
            expected=torch.tensor(-0.0004),
            rtol=0,
            atol=1e-4,
        )

        assert_expected(
            actual=torch.tensor(actual_a_embedding.size()),
            expected=torch.tensor((1, 1024)),
        )

        assert_expected(
            actual=torch.tensor(actual_b_embedding.size()),
            expected=torch.tensor((1, 1024)),
        )
