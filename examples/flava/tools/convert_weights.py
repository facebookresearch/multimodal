# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse

import torch
from torchmultimodal.models.flava.flava_model import flava_model_for_pretraining

KEY_REPLACEMENTS = {
    "image_encoder.module": "image_encoder",
    "text_encoder.module": "text_encoder",
    "mm_encoder.module": "mm_encoder",
    "mm_encoder.encoder.cls_token": "mm_encoder.cls_token",
    "mm_image_projection": "image_to_mm_projection",
    "mm_text_projection": "text_to_mm_projection",
    "model.heads.cmd.mim_head": "loss.mmm_loss.mim",
    "model.heads.cmd.mlm_head": "loss.mmm_loss.mlm",
    "model.heads.fairseq_mlm": "loss.mlm_loss",
    "model.heads.imagenet.mim_head": "loss.mim_loss",
    "cls.predictions.transform": "cls",
    "cls.predictions": "cls",
    "cls.LayerNorm": "cls.layer_norm",
    "model.text_projection": "loss.contrastive_loss.text_projection",
    "model.image_projection": "loss.contrastive_loss.image_projection",
    "model.heads.cmd.clip_head.logit_scale": "loss.contrastive_loss.logit_scale",
    "model.heads.cmd.itm_head": "loss.itm_loss",
    "intermediate.dense": "intermediate",
    "output.dense": "output",
}


def convert_weights(args):
    ckpt = torch.load(args.ckpt_file, map_location="cpu")
    flava = flava_model_for_pretraining()
    model = ckpt["model"]
    import pdb

    pdb.set_trace()
    for key in list(model.keys()):
        original = key
        for option, replacement in KEY_REPLACEMENTS.items():
            key = key.replace(option, replacement)
        model[key] = model.pop(original)

    if args.add_codebook:
        # Since codebook is anyways not trained in FLAVA pretraining
        # we can use the pretrained one that we get from FLAVA initialized
        # model
        model.update(
            {
                f"image_codebook.{key}": value
                for key, value in flava.image_codebook.state_dict().items()
            }
        )
    flava.load_state_dict(model)

    # Let's save the model now.
    torch.save(flava.state_dict(), args.save_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert weights")
    parser.add_argument("ckpt_file", type=str)
    parser.add_argument("save_file", type=str)
    parser.add_argument("--add_codebook", action="store_true")

    args = parser.parse_args()

    convert_weights(args)
