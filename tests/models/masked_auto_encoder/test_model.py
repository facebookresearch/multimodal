# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch
from tests.test_utils import assert_expected, init_weights_with_constant, set_rng_seed
from torchmultimodal.models.masked_auto_encoder.model import (
    audio_mae,
    image_mae,
    vit_b_16_audio_mae,
    vit_l_16_audio_mae,
    vit_s_16_audio_mae,
)


class TestImageMaskedAutoEncoder:
    @pytest.fixture(autouse=True)
    def random(self):
        set_rng_seed(0)

    @pytest.fixture
    def inputs(self):
        return torch.ones(2, 3, 224, 224)

    @pytest.fixture
    def image_mae_model(self, masking_ratio=0.75):
        return image_mae(
            encoder_layers=1, decoder_layers=1, masking_ratio=masking_ratio
        )

    def test_image_mae_eval(self, inputs):
        model = image_mae(encoder_layers=1, decoder_layers=1)
        init_weights_with_constant(model)
        actual = model(inputs).encoder_output.last_hidden_state

        # default masking = 0.75, so num patches in encoder output = (224/16)^2 * 0.25 + 1
        assert_expected(actual.size(), (2, 50, 768))
        assert_expected(actual.mean().item(), 1.0)

    def test_image_mae_train_no_masking(self, inputs):
        model = image_mae(encoder_layers=1, decoder_layers=1, masking_ratio=0.0)
        init_weights_with_constant(model)
        actual = model(inputs)

        encoder_out = actual.encoder_output.last_hidden_state
        assert_expected(encoder_out.size(), (2, 197, 768))
        assert_expected(encoder_out.mean().item(), 1.0)

        assert_expected(actual.mask, torch.zeros(2, 196))

        pred = actual.decoder_pred
        assert_expected(pred.size(), (2, 196, 768))
        assert_expected(pred.mean().item(), 513.0)

        labels = actual.label_patches
        assert_expected(labels, torch.ones(2, 196, 16 * 16 * 3))

    def test_image_mae_train_masking(self, inputs):
        model = image_mae(encoder_layers=1, decoder_layers=1, masking_ratio=0.5)
        init_weights_with_constant(model)
        actual = model(inputs)

        encoder_out = actual.encoder_output.last_hidden_state
        assert_expected(encoder_out.size(), (2, 99, 768))
        assert_expected(encoder_out.mean().item(), 1.0)

        assert_expected(actual.mask.size(), (2, 196))
        assert_expected(actual.mask.sum(dim=-1), torch.Tensor([98, 98]))

        pred = actual.decoder_pred
        assert_expected(pred.size(), (2, 196, 768))
        assert_expected(pred.mean().item(), 513.0)

        labels = actual.label_patches
        assert_expected(labels, torch.ones(2, 196, 16 * 16 * 3))

    def test_label_patchification(self):
        model = image_mae(
            encoder_layers=1, decoder_layers=1, image_size=4, patch_size=2
        )
        inputs = torch.Tensor(
            [
                [
                    [[1, 2, 3, 4], [5, 6, 7, 8], [8, 7, 6, 5], [4, 3, 2, 1]],
                    [[2, 4, 6, 8], [1, 3, 5, 7], [8, 6, 4, 2], [7, 5, 1, 3]],
                    [[1, 2, 1, 2], [3, 2, 1, 2], [1, 2, 1, 2], [1, 2, 1, 2]],
                ]
            ]
        )
        actual = model._patchify_input(inputs)
        assert_expected(
            actual,
            torch.Tensor(
                [
                    [
                        [1, 2, 1, 2, 4, 2, 5, 1, 3, 6, 3, 2],
                        [3, 6, 1, 4, 8, 2, 7, 5, 1, 8, 7, 2],
                        [8, 8, 1, 7, 6, 2, 4, 7, 1, 3, 5, 2],
                        [6, 4, 1, 5, 2, 2, 2, 1, 1, 1, 3, 2],
                    ]
                ]
            ),
        )

    def test_image_mae_init(self):
        model = image_mae(
            encoder_layers=1,
            decoder_layers=1,
            masking_ratio=0.0,
            encoder_hidden_dim=4,
            decoder_hidden_dim=8,
            image_size=(2, 2),
            patch_size=1,
        )
        for p in model.embeddings.position_embeddings:
            assert p.requires_grad is False
        for p in model.decoder_embed.position_embeddings:
            assert p.requires_grad is False
        assert_expected(
            model.embeddings.position_embeddings,
            torch.Tensor(
                [
                    [
                        [0.0000, 0.0000, 0.0000, 0.0000],
                        [0.0000, 1.0000, 0.0000, 1.0000],
                        [0.8415, 0.5403, 0.0000, 1.0000],
                        [0.0000, 1.0000, 0.8415, 0.5403],
                        [0.8415, 0.5403, 0.8415, 0.5403],
                    ]
                ]
            ),
            atol=1e-4,
            rtol=1e-4,
        )

        assert_expected(
            model.decoder_embed.position_embeddings,
            torch.Tensor(
                [
                    [
                        [
                            0.0000,
                            0.0000,
                            0.0000,
                            0.0000,
                            0.0000,
                            0.0000,
                            0.0000,
                            0.0000,
                        ],
                        [
                            0.0000,
                            0.0000,
                            1.0000,
                            1.0000,
                            0.0000,
                            0.0000,
                            1.0000,
                            1.0000,
                        ],
                        [
                            0.8415,
                            0.0100,
                            0.5403,
                            0.9999,
                            0.0000,
                            0.0000,
                            1.0000,
                            1.0000,
                        ],
                        [
                            0.0000,
                            0.0000,
                            1.0000,
                            1.0000,
                            0.8415,
                            0.0100,
                            0.5403,
                            0.9999,
                        ],
                        [
                            0.8415,
                            0.0100,
                            0.5403,
                            0.9999,
                            0.8415,
                            0.0100,
                            0.5403,
                            0.9999,
                        ],
                    ]
                ]
            ),
            atol=1e-4,
            rtol=1e-4,
        )


class TestAudioMaskedAutoEncoder:
    @pytest.fixture(autouse=True)
    def random(self):
        set_rng_seed(0)

    @pytest.fixture
    def inputs(self):
        return torch.ones(2, 1, 1024, 128)

    @pytest.fixture
    def audio_mae_model(self, masking_ratio=0.8):
        return audio_mae(
            encoder_layers=1, decoder_layers=1, masking_ratio=masking_ratio
        )

    def test_audio_mae_eval(self, inputs):
        model = audio_mae(encoder_layers=1, decoder_layers=1)
        init_weights_with_constant(model)
        actual = model(inputs).encoder_output.last_hidden_state
        assert_expected(actual.size(), (2, 103, 768))
        assert_expected(actual.mean().item(), 1.0)

    def test_audio_mae_train_no_masking(self, inputs):
        model = audio_mae(encoder_layers=1, decoder_layers=1, masking_ratio=0.0)
        init_weights_with_constant(model)
        actual = model(inputs)

        encoder_out = actual.encoder_output.last_hidden_state
        assert_expected(encoder_out.size(), (2, 513, 768))
        assert_expected(encoder_out.mean().item(), 1.0)

        assert_expected(actual.mask, torch.zeros(2, 512))

        pred = actual.decoder_pred
        assert_expected(pred.size(), (2, 512, 16 * 16 * 1))
        assert_expected(pred.mean().item(), 513.0, rtol=1e-4, atol=1e-4)

        labels = actual.label_patches
        assert_expected(labels, torch.ones(2, 512, 16 * 16 * 1))

    def test_audio_mae_train_masking(self, inputs):
        model = audio_mae(encoder_layers=1, decoder_layers=1, masking_ratio=0.5)
        init_weights_with_constant(model)
        actual = model(inputs)

        encoder_out = actual.encoder_output.last_hidden_state
        assert_expected(encoder_out.size(), (2, 257, 768))
        assert_expected(encoder_out.mean().item(), 1.0)

        assert_expected(actual.mask.size(), (2, 512))
        assert_expected(actual.mask.sum(dim=-1), torch.Tensor([256, 256]))

        pred = actual.decoder_pred
        assert_expected(pred.size(), (2, 512, 16 * 16 * 1))
        assert_expected(pred.mean().item(), 513.0, rtol=1e-4, atol=1e-4)

        labels = actual.label_patches
        assert_expected(labels, torch.ones(2, 512, 16 * 16 * 1))

    @pytest.mark.parametrize(
        "model, output_size",
        [
            (vit_s_16_audio_mae, (2, int(64 * 8 * 0.2) + 1, 384)),
            (vit_b_16_audio_mae, (2, int(64 * 8 * 0.2) + 1, 768)),
            (vit_l_16_audio_mae, (2, int(64 * 8 * 0.2) + 1, 1024)),
        ],
    )
    def test_standard_audio_mae(self, model, output_size, inputs):
        model = model()
        init_weights_with_constant(model)
        actual = model(inputs).encoder_output.last_hidden_state
        assert_expected(actual.size(), output_size)
        assert_expected(actual.mean().item(), 1.0)

    def test_label_patchification(self):
        model = audio_mae(
            encoder_layers=1, decoder_layers=1, input_size=(2, 4), patch_size=2
        )
        inputs = torch.Tensor(
            [
                [
                    [[1, 2, 3, 4], [5, 6, 7, 8]],
                ]
            ]
        )
        actual = model._patchify_input(inputs)
        assert_expected(
            actual,
            torch.Tensor(
                [
                    [
                        [1, 2, 5, 6],
                        [3, 4, 7, 8],
                    ]
                ]
            ),
        )

    def test_audio_mae_init(self):
        model = audio_mae(
            encoder_layers=1,
            decoder_layers=1,
            masking_ratio=0.0,
            encoder_hidden_dim=4,
            decoder_hidden_dim=8,
            input_size=(2, 3),
            patch_size=1,
        )
        for p in model.embeddings.position_embeddings:
            assert p.requires_grad is False
        for p in model.decoder_embed.position_embeddings:
            assert p.requires_grad is False
        assert_expected(
            model.embeddings.position_embeddings,
            torch.Tensor(
                [
                    [
                        [0.0000, 0.0000, 0.0000, 0.0000],
                        [0.0000, 1.0000, 0.0000, 1.0000],
                        [0.8415, 0.5403, 0.0000, 1.0000],
                        [0.0000, 1.0000, 0.8415, 0.5403],
                        [0.8415, 0.5403, 0.8415, 0.5403],
                        [0.0000, 1.0000, 0.9093, -0.4161],
                        [0.8415, 0.5403, 0.9093, -0.4161],
                    ]
                ]
            ),
            atol=1e-4,
            rtol=1e-4,
        )

        assert_expected(
            model.decoder_embed.position_embeddings,
            torch.Tensor(
                [
                    [
                        [
                            0.0000,
                            0.0000,
                            0.0000,
                            0.0000,
                            0.0000,
                            0.0000,
                            0.0000,
                            0.0000,
                        ],
                        [
                            0.0000,
                            0.0000,
                            1.0000,
                            1.0000,
                            0.0000,
                            0.0000,
                            1.0000,
                            1.0000,
                        ],
                        [
                            0.8415,
                            0.0100,
                            0.5403,
                            0.9999,
                            0.0000,
                            0.0000,
                            1.0000,
                            1.0000,
                        ],
                        [
                            0.0000,
                            0.0000,
                            1.0000,
                            1.0000,
                            0.8415,
                            0.0100,
                            0.5403,
                            0.9999,
                        ],
                        [
                            0.8415,
                            0.0100,
                            0.5403,
                            0.9999,
                            0.8415,
                            0.0100,
                            0.5403,
                            0.9999,
                        ],
                        [
                            0.0000,
                            0.0000,
                            1.0000,
                            1.0000,
                            0.9093,
                            0.0200,
                            -0.4161,
                            0.9998,
                        ],
                        [
                            0.8415,
                            0.0100,
                            0.5403,
                            0.9999,
                            0.9093,
                            0.0200,
                            -0.4161,
                            0.9998,
                        ],
                    ]
                ]
            ),
            atol=1e-4,
            rtol=1e-4,
        )
