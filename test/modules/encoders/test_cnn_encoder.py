# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
from torchmultimodal.modules.encoders.cnn_encoder import CNNEncoder

class TestCnnEncoder(unittest.TestCase):
    def test_invalid_dim_lengths(self):
        input_dims = [1,2]
        output_dims = [3,4,5]
        kernel_dims = [6,7,8]
        self.assertRaises(
            AssertionError, CNNEncoder, input_dims, output_dims, kernel_dims
        )
