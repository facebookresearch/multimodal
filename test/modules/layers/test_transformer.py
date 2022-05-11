# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

# import torch
from test.test_utils import set_rng_seed  # , assert_expected

# from torch import nn
from torchmultimodal.modules.layers.transformer import TransformerEncoder


class TestTransformer(unittest.TestCase):
    """
    Test the Transformer classes
    """

    def setUp(self):
        set_rng_seed(4)
        self.encoder = TransformerEncoder()

    def test_self_attention(self):
        pass
