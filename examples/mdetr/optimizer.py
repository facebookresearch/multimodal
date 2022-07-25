# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Copyright (c) Aishwarya Kamath & Nicolas Carion. Licensed under the Apache License 2.0. All Rights Reserved
"""Collections of utilities related to optimization."""
from bisect import bisect_right

import torch


def update_ema(model, model_ema, decay):
    """Apply exponential moving average update.

    The  weights are updated in-place as follow:
    w_ema = w_ema * decay + (1 - decay) * w
    Args:
        model: active model that is being optimized
        model_ema: running average model
        decay: exponential decay parameter
    """
    with torch.no_grad():
        if hasattr(model, "module"):
            # unwrapping DDP
            model = model.module
        msd = model.state_dict()
        for k, ema_v in model_ema.state_dict().items():
            model_v = msd[k].detach()
            ema_v.copy_(ema_v * decay + (1.0 - decay) * model_v)


def adjust_learning_rate(
    optimizer,
    epoch: int,
    curr_step: int,
    num_training_steps: int,
    args,
):
    """Adjust the lr according to the schedule.

    Args:
        Optimizer: torch optimizer to update.
        epoch(int): number of the current epoch.
        curr_step(int): number of optimization step taken so far.
        num_training_step(int): total number of optimization steps.
        args: additional training dependent args:
              - lr_drop(int): number of epochs before dropping the learning rate.
              - fraction_warmup_steps(float) fraction of steps over which the lr will be increased to its peak.
              - lr(float): base learning rate
              - lr_backbone(float): learning rate of the backbone
              - text_encoder_backbone(float): learning rate of the text encoder
              - schedule(str): the requested learning rate schedule:
                   "step": all lrs divided by 10 after lr_drop epochs
                   "multistep": divided by 2 after lr_drop epochs, then by 2 after every 50 epochs
                   "linear_with_warmup": same as "step" for backbone + transformer, but for the text encoder, linearly
                                         increase for a fraction of the training, then linearly decrease back to 0.
                   "all_linear_with_warmup": same as "linear_with_warmup" for all learning rates involved.

    """
    num_warmup_steps: int = round(args.fraction_warmup_steps * num_training_steps)
    if args.schedule == "step":
        gamma = 0.1 ** (epoch // args.lr_drop)
        text_encoder_gamma = gamma
    elif args.schedule == "multistep":
        milestones = list(range(args.lr_drop, args.epochs, 50))
        gamma = 0.5 ** bisect_right(milestones, epoch)
        text_encoder_gamma = gamma
    elif args.schedule == "linear_with_warmup":
        gamma = 0.1 ** (epoch // args.lr_drop)
        if curr_step < num_warmup_steps:
            text_encoder_gamma = float(curr_step) / float(max(1, num_warmup_steps))
        else:
            text_encoder_gamma = max(
                0.0,
                float(num_training_steps - curr_step)
                / float(max(1, num_training_steps - num_warmup_steps)),
            )
    elif args.schedule == "all_linear_with_warmup":
        if curr_step < num_warmup_steps:
            text_encoder_gamma = float(curr_step) / float(max(1, num_warmup_steps))
        else:
            text_encoder_gamma = max(
                0.0,
                float(num_training_steps - curr_step)
                / float(max(1, num_training_steps - num_warmup_steps)),
            )
        gamma = text_encoder_gamma
    else:
        raise NotImplementedError

    base_lrs = [args.lr, args.lr_backbone, args.text_encoder_lr]
    gammas = [gamma, gamma, text_encoder_gamma]
    assert len(optimizer.param_groups) == len(base_lrs)
    for param_group, lr, gamma_group in zip(optimizer.param_groups, base_lrs, gammas):
        param_group["lr"] = lr * gamma_group


def build_optimizer(model, args):
    param_dicts = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if "backbone" not in n and "text_encoder" not in n and p.requires_grad
            ]
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if "backbone" in n and p.requires_grad
            ],
            "lr": args.lr_backbone,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if "text_encoder" in n and p.requires_grad
            ],
            "lr": args.text_encoder_lr,
        },
    ]
    optimizer = torch.optim.AdamW(
        param_dicts, lr=args.lr, weight_decay=args.weight_decay
    )
    return optimizer
