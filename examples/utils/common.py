# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import copy
import hashlib
from typing import Optional

import torch
from torch import nn


def store_model_weights(
    model: nn.Module,
    checkpoint_path: str,
    checkpoint_key: Optional[str] = None,
    strict=True,
):
    """
    This method can be used to prepare weights files for new models. It receives as
    input a model architecture and a checkpoint from the training script and produces
    a file with the weights ready for release.

    Code reference:
        https://github.com/pytorch/vision/blob/main/references/classification/utils.py

    Args:
        model (nn.Module): The model on which the weights will be loaded for validation purposes.
        checkpoint_path (str): The path of the checkpoint we will load.
        checkpoint_key (str, optional): The key of the checkpoint where the model weights are stored.
            Default: "model".
        strict (bool): whether to strictly enforce that the keys
            in :attr:`state_dict` match the keys returned by this module's
            :meth:`~torch.nn.Module.state_dict` function. Default: ``True``
    Returns:
        output_path (str): The location where the weights are saved.
    """
    # Store the new model next to the checkpoint_path
    checkpoint_path = os.path.abspath(checkpoint_path)
    output_dir = os.path.dirname(checkpoint_path)

    # Deep copy to avoid side-effects on the model object.
    model = copy.deepcopy(model)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    # Load the weights to the model to validate that everything works
    # and remove unnecessary weights (such as auxiliaries, etc)
    # If no checkpoint_key is provided, the checkpoint_path is treated
    # as the model weights file
    if checkpoint_key is None:
        model.load_state_dict(checkpoint, strict=strict)
    else:
        if checkpoint_key == "model_ema":
            del checkpoint[checkpoint_key]["n_averaged"]
            nn.modules.utils.consume_prefix_in_state_dict_if_present(
                checkpoint[checkpoint_key], "module."
            )
        model.load_state_dict(checkpoint[checkpoint_key], strict=strict)

    tmp_path = os.path.join(output_dir, str(model.__hash__()))
    torch.save(model.state_dict(), tmp_path)

    sha256_hash = hashlib.sha256()
    with open(tmp_path, "rb") as f:
        # Read and update hash string value in blocks of 4K
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
        hh = sha256_hash.hexdigest()

    output_path = os.path.join(output_dir, "weights-" + str(hh[:8]) + ".pth")
    os.replace(tmp_path, output_path)

    return output_path
