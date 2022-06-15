# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
from test.test_utils import assert_expected, set_rng_seed
from torch import Tensor
from torchmultimodal.modules.encoders.albef_text_encoder import ALBEFTextEmbeddings


class TestALBEFVisionEncoder:
    def test_conv_proj(self):
        # test the embeddings of the ALBEFTextEncoder
        set_rng_seed(0)
        text_embeddings = ALBEFTextEmbeddings(hidden_size=3)
        input = torch.randint(10, (2, 2))
        output = text_embeddings(input)
        expected = Tensor(
            [
                [[0.306898, -1.349008, 1.042110], [-1.241071, 0.033332, 1.207738]],
                [[1.097125, -1.321374, 0.224249], [-0.619877, 1.410763, -0.790886]],
            ]
        )
        assert_expected(output, expected, rtol=0, atol=1e-4)
