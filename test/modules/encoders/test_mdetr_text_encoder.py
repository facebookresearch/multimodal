# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from copy import deepcopy

import torch
from test.test_utils import assert_expected, set_rng_seed
from torchmultimodal.modules.encoders.mdetr_text_encoder import (
    MDETRTextEmbeddings,
    MDETRTextEncoder,
    WrappedTransformerEncoder,
)
from torchmultimodal.utils.common import filter_dict
from torchtext.models.roberta.bundler import ROBERTA_BASE_ENCODER


class TestMDETRTextEncoder(unittest.TestCase):
    def setUp(self):
        self.max_position_embeddings = 514
        self.hidden_size = 768
        self.embeddings = MDETRTextEmbeddings(
            hidden_size=self.hidden_size,
            vocab_size=50265,
            pad_token_id=1,
            type_vocab_size=1,
            max_position_embeddings=self.max_position_embeddings,
            layer_norm_eps=1e-05,
            hidden_dropout_prob=0.1,
        )
        set_rng_seed(0)
        self._populate_embedding_weights()

        self.roberta_encoder = ROBERTA_BASE_ENCODER.get_model()
        # Remove extra args due to TorchText RoBERTa encoder's forward taking in tokens instead of embeddings
        wrapped_args = filter_dict(
            lambda x: x not in ["vocab_size", "padding_idx", "max_seq_len", "scaling"],
            ROBERTA_BASE_ENCODER.encoderConf.__dict__,
        )
        self.wrapped_transformer_encoder = WrappedTransformerEncoder(**wrapped_args)
        self._populate_wrapped_transformer_weights()

        self.text_encoder = MDETRTextEncoder(
            embeddings=self.embeddings, encoder=self.wrapped_transformer_encoder
        )
        self.text_encoder.eval()

        self.input_ids = torch.tensor(
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
            ],
            dtype=torch.int,
        )
        self.attention_mask = torch.tensor(
            [
                [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            ],
            dtype=torch.int,
        )
        self.batch_size, self.input_length = self.input_ids.size()

    def test_mdetr_text_embeddings(self):
        expected = torch.Tensor(
            [
                0.62926,
                0.71966,
                0.59531,
                0.54416,
                0.61227,
                0.67181,
                0.61707,
                0.47298,
                0.72058,
                0.68760,
                0.75449,
                0.52463,
                0.49111,
                0.57840,
                0.44969,
                0.71950,
            ]
        )
        out = self.embeddings(self.input_ids, self.attention_mask)
        actual = out[1, :, 1]
        self.assertEqual(
            out.size(), (self.batch_size, self.input_length, self.hidden_size)
        )
        assert_expected(actual, expected, rtol=0.0, atol=1e-4)

    def test_mdetr_wrapped_transformer(self):
        set_rng_seed(0)
        inp = torch.rand((2, 16, 768))
        expected = torch.Tensor(
            [
                0.10437,
                0.10304,
                0.10998,
                0.10926,
                0.10760,
                0.10326,
                0.10478,
                0.10586,
                0.10500,
                0.11583,
                0.10339,
                0.10585,
                0.10405,
                0.10533,
                0.11012,
                0.11071,
            ]
        )
        out = self.wrapped_transformer_encoder(inp, self.attention_mask)
        actual = out[1, :, 1]
        self.assertEqual(
            out.size(), (self.batch_size, self.input_length, self.hidden_size)
        )
        assert_expected(actual, expected, rtol=0.0, atol=1e-4)

    def test_mdetr_text_encoder(self):
        expected = torch.Tensor(
            [
                0.04690,
                0.06246,
                0.05839,
                0.05727,
                0.06413,
                0.05640,
                0.06599,
                0.04888,
                0.07137,
                0.05381,
                0.06141,
                0.06225,
                0.04788,
                0.06256,
                0.05058,
                0.06433,
            ]
        )
        out = self.text_encoder(self.input_ids, self.attention_mask)
        actual = out[1, :, 1]
        self.assertEqual(
            out.size(), (self.batch_size, self.input_length, self.hidden_size)
        )
        assert_expected(actual, expected, rtol=0.0, atol=1e-4)

    # Unlike with the encoder where all weights are already present in the TorchText checkpoint,
    # for the embedding we need to construct our own weights since word_embeddings are not present
    # in the TorchText model bundle.
    def _populate_embedding_weights(self):
        embeddings_state_dict = {}
        for k, v in self.embeddings.state_dict().items():
            # These have already been set on init and shouldn't be changed
            if k == "position_ids":
                embeddings_state_dict[k] = v
            # Set these to match the HF RoBERTa weights
            elif k == "token_type_embeddings.weight":
                embeddings_state_dict[k] = torch.zeros(v.size())
            # Set all other weights randomly
            else:
                embeddings_state_dict[k] = torch.rand(v.size())
        self.embeddings.load_state_dict(embeddings_state_dict)

    def _populate_wrapped_transformer_weights(self):
        transformer_state_dict = deepcopy(
            self.roberta_encoder.encoder.transformer.state_dict()
        )
        # Remove embedding weights, but keep final layer norm weights
        wrapped_state_dict = filter_dict(
            lambda x: "embedding" not in x or "layer_norm" in x, transformer_state_dict
        )
        self.wrapped_transformer_encoder.load_state_dict(wrapped_state_dict)
