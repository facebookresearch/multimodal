# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# This file is a temporary way to perform the state dict mapping from MDETR -> TorchMultimodal classes
# until we have a pretrained TorchMultimodal checkpoint available
from typing import Dict, List, Optional

import torch
from torch import Tensor

# Define a couple helper functions
def filter_dict(key_condition, d):
    return {k: v for k, v in d.items() if key_condition(k)}


def get_params_for_layer(state_dict, i):
    return [x for x in state_dict.keys() if f"layer.{i}." in x or f"layers.{i}" in x]


# Mapping from TorchText layers to Hugging Face ones
# TorchText's input projection should equal the concatenation of
# Hugging Face's Q,K,V matrics
param_mapping = {
    "self_attn.in_proj_weight": [
        "attention.self.query",
        "attention.self.key",
        "attention.self.value",
    ],
    "self_attn.in_proj_bias": [
        "attention.self.query",
        "attention.self.key",
        "attention.self.value",
    ],
    "self_attn.out_proj": "attention.output.dense",
    "norm1": "attention.output.LayerNorm",
    "linear1": "intermediate.dense",
    "linear2": "output.dense",
    "norm2": "output.LayerNorm",
}

# These are the prefixes of the text encoder layers as they occur in Hugging Face and TorchText
hf_layer_prefix = "transformer.text_encoder.encoder.layer"
tt_layer_prefix = "text_encoder.encoder.layers.layers"

postfixes = ["weight", "bias"]

# Create a state dict for ith layer of TorchText RoBERTa encoder
# for storing weights from ith layer of Hugging Face's encoder
def map_layer(hf_state_dict, tt_state_dict, i):
    mapped_state_dict = {}
    hf_layer = get_params_for_layer(hf_state_dict, i)
    tt_layer = get_params_for_layer(tt_state_dict, i)
    for tt_key_short, hf_key_short in param_mapping.items():
        tt_key_short = ".".join([tt_layer_prefix, str(i), tt_key_short])
        # For Q,K,V matrices we need to concat the weights
        if isinstance(hf_key_short, List):
            hf_keys_short = list(
                map(lambda x: ".".join([hf_layer_prefix, str(i), x]), hf_key_short)
            )
            # for postfix in postfixes:
            postfix = tt_key_short.split("_")[-1]
            hf_keys = [".".join([x, postfix]) for x in hf_keys_short]
            if not any([x in tt_key_short for x in postfixes]):
                tt_key = ".".join([tt_key_short, postfix])
            else:
                tt_key = tt_key_short
            qkv_combined = torch.concat([hf_state_dict[hf_key] for hf_key in hf_keys])
            mapped_state_dict[tt_key] = qkv_combined
        else:
            hf_key_short = ".".join([hf_layer_prefix, str(i), hf_key_short])
            for postfix in postfixes:
                tt_key = ".".join([tt_key_short, postfix])
                hf_key = ".".join([hf_key_short, postfix])
                mapped_state_dict[tt_key] = hf_state_dict[hf_key]

    return mapped_state_dict


# Just a for loop around the text encoder layer mapping
def map_text_encoders(
    hf_state_dict: Dict[str, Tensor],
    tt_state_dict: Dict[str, Tensor],
    n_layers: int = 12,
):
    mapped_state_dict = {}
    for i in range(n_layers):
        mapped_state_dict.update(map_layer(hf_state_dict, tt_state_dict, i))
    return mapped_state_dict


# The main function used to map from the MDETR state dict to the TorchMultimodal one
# TODO: refactor to remove the explicit dependency on n_layers
def map_mdetr_state_dict(
    mdetr_state_dict,
    mm_state_dict,
    n_layers: int = 12,
    prefix: Optional[str] = "model",
    include_contrastive: bool = True,
):
    # Perform the text encoder mapping
    mapped_state_dict = map_text_encoders(
        mdetr_state_dict, mm_state_dict, n_layers=n_layers
    )

    # Miscellaneous renaming (this can probably be cleaned up)
    mapped_state_dict = {
        k.replace("transformer.text_encoder", "text_encoder"): v
        for k, v in mapped_state_dict.items()
        if "embeddings" not in k
    }
    for k, v in mdetr_state_dict.items():
        if (
            not k.startswith("transformer.text_encoder")
            and not k.startswith("transformer.resizer")
            and "input_proj" not in k
        ):
            mapped_state_dict[k.replace("backbone.0", "image_backbone")] = v
        if "embeddings" in k:
            mapped_state_dict[k.replace("transformer.", "")] = v
        if "input_proj" in k:
            mapped_state_dict[k.replace("input_proj", "image_projection")] = v
        if "resizer" in k:
            mapped_state_dict[
                k.replace("transformer.", "").replace("resizer", "text_projection")
            ] = v
        if "embeddings.LayerNorm" in k:
            new_k = k.replace("transformer.", "")
            mapped_state_dict[new_k.replace("LayerNorm", "layer_norm")] = v
            del mapped_state_dict[new_k]
        if "bbox_embed" in k:
            parsed = k.split(".")
            i = int(parsed[parsed.index("layers") + 1])
            mapped_state_dict[
                k.replace("layers", "model").replace(str(i), str(2 * i))
            ] = v
            del mapped_state_dict[k]
        if all([x in k for x in ["transformer", "layers", "linear"]]):
            k_split = k.split(".")
            i = int(k_split[-2][-1])
            k_new = ".".join(
                k_split[:-2] + ["mlp", "model", str(3 * (i - 1)), k_split[-1]]
            )
            mapped_state_dict[k_new] = v
            del mapped_state_dict[k]

        if "contrastive" in k:
            if include_contrastive:
                k_new = (
                    k.replace("align", "alignment")
                    .replace("projection_image", "image_projection")
                    .replace("projection_text", "text_projection")
                )
                mapped_state_dict[k_new] = v
            del mapped_state_dict[k]
        if "qa" in k or "answer" in k:
            k_new = k.replace("qa", "vqa").replace("_head", "")
            if "head" in k:
                k_new = ".".join(["vqa_heads", "heads", k_new])
            mapped_state_dict[k_new] = v
            del mapped_state_dict[k]

    if prefix is not None:
        mapped_state_dict = {
            ".".join([prefix, k])
            .replace("model.contrastive", "contrastive")
            .replace("model.vqa", "vqa")
            .replace("model.answer", "answer"): v
            for k, v in mapped_state_dict.items()
        }
    return mapped_state_dict
