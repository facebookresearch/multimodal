# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Callable, List, Optional

import torch
from torch import nn
from torchmultimodal.architectures.late_fusion import LateFusionArchitecture
from torchmultimodal.modules.encoders.cnn_encoder import CNNEncoder
from torchmultimodal.modules.encoders.lstm_encoder import LSTMEncoder
from torchmultimodal.modules.fusions.concat_fusion import ConcatFusionModule
from torchmultimodal.modules.layers.mlp import MLP


class CNNLSTM(LateFusionArchitecture):
    """A simple baseline model for vision and language tasks.

    CNNLSTM passes images through a CNN and text through an LSTM,
    followed by a concat fusion module before a final classifier.
    The classifier can be passed explicitly, but the default is an MLP.

    Args:
        text_vocab_size (int): The vocab size for text data.
        text_embedding_dim (int): The size of each text embedding vector.
        cnn_input_dims (List[int]): Input dimensions for CNN layers.
        cnn_output_dims (List[int]): Output dimensions for CNN layers.
            Should match input dimensions offset by one.
        cnn_kernel_sizes (List[int]): Kernel sizes for CNN convolutions.
            Should match the sizes of cnn_input_dims and cnn_output_dims.
        lstm_input_size (int): Number of expected features in LSTM input.
        lstm_hidden_dim (int): Number of features in the LSTM hidden state.
        lstm_bidirectional (bool): Whether to use a bidirectional LSTM.
        lstm_batch_first (bool): Whether to provide LSTM batches as
            (batch, seq, feature) or (seq, batch, feature).
        classifier (Optional[Callable[..., Tensor]]): An optional classifier
            head. If not passed, an MLP will be used with classifier_in_dim
            input dimension.
        classifier_in_dim (Optional[int]): Input dimension for classifier.
            Should equal output_dim for CNN + output_dim for LSTM (flattened).
        num_classes (int): Number of classes predicted by classifier.

    Inputs: image (Tensor): Tensor containing image features.
            text (Tensor): Tensor containing text features.
    """

    def __init__(
        self,
        text_vocab_size: int,
        text_embedding_dim: int = 20,
        cnn_input_dims: Optional[List[int]] = None,
        cnn_output_dims: Optional[List[int]] = None,
        cnn_kernel_sizes: Optional[List[int]] = None,
        lstm_input_size: int = 20,
        lstm_hidden_dim: int = 50,
        lstm_bidirectional: bool = True,
        lstm_batch_first: bool = True,
        classifier: Optional[Callable[..., nn.Module]] = None,
        classifier_in_dim: Optional[int] = 450,
        num_classes: Optional[int] = 2,
    ):
        check_all = [cnn_input_dims, cnn_output_dims, cnn_kernel_sizes]
        none_count = check_all.count(None)
        if none_count > 0:
            assert none_count == 3, (
                "Please either pass all CNN parameters or "
                + "none of them. If you don't pass any, expected image input "
                + "is of size 224 x 224."
            )
            cnn_input_dims = (3, 64, 128, 128, 64, 64)
            cnn_output_dims = cnn_input_dims[1:] + (10,)
            cnn_kernel_sizes = [7, 5, 5, 5, 5, 1]

        assert classifier is not None or classifier_in_dim is not None, (
            "Please pass either classifier callable or classifier_in_dim for "
            + "MLP classifier."
        )
        assert classifier is not None or num_classes is not None, (
            "Please pass either classifier callable or num_classes for "
            + "MLP classifier."
        )

        cnn_encoder = CNNEncoder(
            input_dims=cnn_input_dims,
            output_dims=cnn_output_dims,
            kernel_sizes=cnn_kernel_sizes,
        )

        text_encoder = LSTMEncoder(
            vocab_size=text_vocab_size,
            embedding_dim=text_embedding_dim,
            input_size=lstm_input_size,
            hidden_size=lstm_hidden_dim,
            bidirectional=lstm_bidirectional,
            batch_first=lstm_batch_first,
        )

        fusion_module = ConcatFusionModule()
        if classifier is None:
            classifier = MLP(
                classifier_in_dim,
                num_classes,
                activation=nn.ReLU,
                normalization=nn.BatchNorm1d,
            )
        else:
            classifier = classifier()

        super().__init__(
            {"image": cnn_encoder, "text": text_encoder}, fusion_module, classifier
        )

    def forward(self, image: torch.Tensor, text: torch.Tensor) -> torch.Tensor:
        return super().forward({"image": image, "text": text})


def simple_cnn_lstm(**kwargs) -> CNNLSTM:
    return CNNLSTM(**kwargs)
