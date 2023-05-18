# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torchmultimodal.models.dalle2.adm.adm import dalle2_adm
from torchmultimodal.modules.diffusion.cfguidance import CFGuidance
from torchmultimodal.modules.diffusion.ddpm import DDPModule
from torchmultimodal.modules.diffusion.predictors import NoisePredictor
from torchmultimodal.modules.diffusion.schedules import (
    cosine_beta_schedule,
    DiffusionSchedule,
)


def dalle2_decoder(
    *,
    # DiffusionSchedule args
    timesteps: int = 1000,
    # ADM args
    time_embed_dim: int = 512,
    cond_embed_dim: int = 2048,
    clip_embed_dim: int = 768,
    clip_embed_name: str = "clip_image",
    predict_variance_value: bool = True,
    image_channels: int = 4,
    depth: int = 512,
    num_resize: int = 3,
    num_res_per_layer: int = 3,
    # CFGuidance args
    use_cf_guidance: bool = True,
    clip_image_guidance_dropout: float = 0.1,
    guidance_strength: float = 7.0,
    learn_null_emb: bool = True,
) -> DDPModule:
    """Constructs primary DALLE-2 diffusion decoder without upsampling.

    Consists of an ADM UNet diffusion model conditioned on CLIP image embeddings. Uses DDPM to generate
    images.

    Ref: https://arxiv.org/abs/2204.06125

    Follows parameters in this config: https://fburl.com/code/jyi0nhdt

    Args:
        timesteps (int): number of timesteps in the diffusion process
        time_embed_dim (int): desired dimensionality of timestep embedding
        cond_embed_dim (int): desired dimensionality of conditional input embeddings
        clip_embed_dim (int): expected dimensionality of CLIP image embeddings
        clip_embed_name (str): name of CLIP embedding conditional input
        predict_variance_value (bool): if True, will double UNet's output channel dim to predict variance values of
            diffusion process
        image_channels (int): channel dim of input images
        depth (int): channel dim of UNet convolutional modules. Expands everytime resolution is downscaled.
            Must be divisible by number of groups used in GroupNorm layers. ADMResBlock
            and ADMAttentionBlock use 32 as a default.
        num_resize (int): number of times resolution will be scaled
        num_res_per_layer (int): number of residual blocks per resolution
        use_cf_guidance (bool): if True, use classifier-free guidance with a conditional input (CLIP embeddings).
            If False, do not condition the model, and ignore clip_image_guidance_dropout, guidance_strength,
            learn_null_emb, clip_embed_dim parameters.
        clip_image_guidance_dropout (float): probability of dropping CLIP image embedding conditional input
            and using the unconditional embedding.
        guidance_strength (float): probability values control the ratio of conditional
            and unconditional generation during inference. Higher values give
            better sample quality at the cost of lesser diversity. A value of -1
            ignores conditional variables, while a value of 0 ignores unconditional
            variables. Default is 2.0, from code ref: https://fburl.com/code/04wxq7ln
        learn_null_emb (bool): If False, then unconditional embeddings are set to zero and are not trainable
            If True, then unconditional embeddings are set to random and are trainable. Defaults to True.
    """
    #
    # Construct UNet
    #
    diffusion_model = dalle2_adm(
        time_embed_dim=time_embed_dim,
        cond_embed_dim=cond_embed_dim,
        clip_embed_dim=clip_embed_dim,
        clip_embed_name=clip_embed_name,
        predict_variance_value=predict_variance_value,
        image_channels=image_channels,
        depth=depth,
        num_resize=num_resize,
        num_res_per_layer=num_res_per_layer,
    )

    #
    # Construct CFGuidance wrapper around ADM model
    #
    if use_cf_guidance:
        diffusion_model = CFGuidance(
            model=diffusion_model,
            dim_cond={clip_embed_name: clip_embed_dim},
            p=clip_image_guidance_dropout,
            guidance=guidance_strength,
            learn_null_emb=learn_null_emb,
        )

    #
    # Construct DDPM decoder
    #
    eval_steps = torch.linspace(0, timesteps - 1, timesteps // 4, dtype=torch.int)
    schedule = DiffusionSchedule(cosine_beta_schedule(timesteps))
    predictor = NoisePredictor(schedule, lambda x: x.clamp(-1, 1))
    model = DDPModule(
        model=diffusion_model,
        schedule=schedule,
        predictor=predictor,
        eval_steps=eval_steps,
    )
    return model
