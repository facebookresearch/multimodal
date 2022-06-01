# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
from torchmultimodal.modules.encoders.cnn_encoder import CNNEncoder

class TestCnnEncoder(unittest.TestCase):
    def test_invalid_arg_lengths(self):
        input_dims = [1,2]
        output_dims = [3,4,5]
        kernel_sizes = [6,7,8]
        self.assertRaises(
            AssertionError, CNNEncoder, input_dims, output_dims, kernel_sizes
        )

    def test_invalid_output_dim(self):
        input_dims = [0,1,2,3]
        output_dims = [1,2,4,5]
        kernel_sizes = [8,9,10,11]
        self.assertRaises(
            AssertionError, CNNEncoder, input_dims, output_dims, kernel_sizes
        )
