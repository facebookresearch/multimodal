# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import List

from examples.cnn_lstm.cnn_encoder import CNNEncoder
from examples.cnn_lstm.lstm_encoder import LSTMEncoder
from torch import nn
from torchmultimodal.models.late_fusion import LateFusion
from torchmultimodal.modules.fusions.concat_fusion import ConcatFusionModule
from torchmultimodal.modules.layers.mlp import MLP


DEFAULT_CNN_INPUT_DIMS = [3, 64, 128, 128, 64, 64]
DEFAULT_CNN_OUTPUT_DIMS = DEFAULT_CNN_INPUT_DIMS[1:] + [10]
DEFAULT_CNN_KERNEL_SIZES = [7, 5, 5, 5, 5, 1]


def cnn_lstm_classifier(
    # Parameters for encoding the text
    text_vocab_size: int,
    text_embedding_dim: int = 20,
    lstm_input_size: int = 20,
    lstm_hidden_dim: int = 50,
    lstm_bidirectional: bool = True,
    lstm_batch_first: bool = True,
    # parameters for encoding the image
    cnn_input_dims: List[int] = DEFAULT_CNN_INPUT_DIMS,
    cnn_output_dims: List[int] = DEFAULT_CNN_OUTPUT_DIMS,
    cnn_kernel_sizes: List[int] = DEFAULT_CNN_KERNEL_SIZES,
    # parameters for the classifier
    classifier_in_dim: int = 450,
    num_classes: int = 2,
) -> LateFusion:
    """
    A simple example to show the composability in TorchMultimodal, and how to
    make use of builder functions to build a given model from an
    architecture. A builder_function takes in all of the parameters needed for
    building the individual layers and simplifies the interface for the
    architecture. In this example, we are explicitly working with the "text"
    and "image" modalities. This is reflected in the ModuleDict passed to the
    LateFusion's init function. Note that these keys should match up
    with the input of the forward function, which will raise an error in case there's
    a mismatch.

    We use the LateFusion to build a multimodal classifier
    which uses a CNN to encode images, an LSTM to encode text and a
    simple MLP as a classifier. The output is raw scores.

    Args:
        text_vocab_size (int): The vocab size for text data.
        text_embedding_dim (int): The size of each text embedding vector.
        lstm_input_size (int): Number of expected features in LSTM input.
        lstm_hidden_dim (int): Number of features in the LSTM hidden state.
        lstm_bidirectional (bool): Whether to use a bidirectional LSTM.
        lstm_batch_first (bool): Whether to provide LSTM batches as
            (batch, seq, feature) or (seq, batch, feature).

        cnn_input_dims (List[int]): Input dimensions for CNN layers.
        cnn_output_dims (List[int]): Output dimensions for CNN layers.
            Should match input dimensions offset by one.
        cnn_kernel_sizes (List[int]): Kernel sizes for CNN convolutions.
            Should match the sizes of cnn_input_dims and cnn_output_dims.

        classifier_in_dim (Optional[int]): Input dimension for classifier.
            Should equal output_dim for CNN + output_dim for LSTM (flattened).
        num_classes (int): Number of classes predicted by classifier.
    """

    image_encoder = CNNEncoder(
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

    # Notice the output of the classifier is raw scores
    classifier = MLP(
        classifier_in_dim,
        num_classes,
        activation=nn.ReLU,
        normalization=nn.BatchNorm1d,
    )

    # The use of builder functions allows us to keep the architecture
    # interfaces clean and intuitive
    return LateFusion(
        encoders=nn.ModuleDict({"image": image_encoder, "text": text_encoder}),
        fusion_module=fusion_module,
        head_module=classifier,
    )
