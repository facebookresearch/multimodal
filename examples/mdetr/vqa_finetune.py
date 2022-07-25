# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import datetime
import json
import math
import random
import sys
import time
from copy import deepcopy
from pathlib import Path
from typing import Dict, Iterable, Optional

import examples.mdetr.utils.dist as dist
import numpy as np
import torch
from examples.mdetr.data.datamodule import GQADataModule
from examples.mdetr.loss import build_weight_dict
from examples.mdetr.model import mdetr_for_vqa
from examples.mdetr.optimizer import adjust_learning_rate, build_optimizer, update_ema
from examples.mdetr.utils.args_parse import get_args_parser
from examples.mdetr.utils.metrics import MetricLogger, SmoothedValue
from examples.mdetr.utils.misc import targets_to
from examples.mdetr.vqa_eval import evaluate


def train_one_epoch(
    model: torch.nn.Module,
    data_loader: Iterable,
    weight_dict: Dict[str, float],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    args,
    max_norm: float = 0,
    model_ema: Optional[torch.nn.Module] = None,
):
    model.train()
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", SmoothedValue(window_size=1, fmt="{value:.6f}"))
    metric_logger.add_meter(
        "lr_backbone", SmoothedValue(window_size=1, fmt="{value:.6f}")
    )
    metric_logger.add_meter(
        "lr_text_encoder", SmoothedValue(window_size=1, fmt="{value:.6f}")
    )
    header = "Epoch: [{}]".format(epoch)
    print_freq = 10

    num_training_steps = int(len(data_loader) * args.epochs)
    for i, batch_dict in enumerate(
        metric_logger.log_every(data_loader, print_freq, header)
    ):
        curr_step = epoch * len(data_loader) + i
        samples = [x.to(device) for x in batch_dict["samples"]]
        targets = batch_dict["targets"]
        text = [t["tokenized"].to(device) for t in targets]
        targets = targets_to(targets, device)
        answers = {k: v.to(device) for k, v in batch_dict["answers"].items()}
        answer_types = {
            k: v.to(device) for k, v in batch_dict["answer_type_mask"].items()
        }
        positive_map = (
            batch_dict["positive_map"].to(device)
            if "positive_map" in batch_dict
            else None
        )
        outputs = model(
            samples,
            text,
            targets,
            positive_map,
            answers,
            answer_types,
            batch_dict["batch_encoding"],
            weight_dict,
        )

        loss_dict = outputs.loss
        losses = sum(
            loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict
        )

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = dist.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {
            f"{k}_unscaled": v for k, v in loss_dict_reduced.items()
        }
        loss_dict_reduced_scaled = {
            k: v * weight_dict[k]
            for k, v in loss_dict_reduced.items()
            if k in weight_dict
        }
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())
        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()

        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()

        adjust_learning_rate(
            optimizer,
            epoch,
            curr_step,
            num_training_steps=num_training_steps,
            args=args,
        )
        if model_ema is not None:
            update_ema(model, model_ema, args.ema_decay)

        metric_logger.update(
            loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled
        )
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(lr_backbone=optimizer.param_groups[1]["lr"])
        metric_logger.update(lr_text_encoder=optimizer.param_groups[2]["lr"])

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def main(args):
    # Init distributed mode
    dist.init_distributed_mode(args)

    # Update dataset specific configs
    if args.dataset_config is not None:
        # https://stackoverflow.com/a/16878364
        d = vars(args)
        with open(args.dataset_config, "r") as f:
            cfg = json.load(f)
        d.update(cfg)

    device = torch.device(args.device)

    if torch.distributed.is_available() and torch.distributed.is_initialized():
        rank = torch.distributed.get_rank()
    else:
        rank = 0

    # fix the seed for reproducibility
    seed = args.seed + rank
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.use_deterministic_algorithms(True)

    # Set up datamodule
    datamodule = GQADataModule(args)
    datamodule.setup("train")
    datamodule.setup("val")
    train_loaders = datamodule.train_dataloader()
    val_loader = datamodule.val_dataloader()

    # Build the model
    model = mdetr_for_vqa(contrastive_dim=64, temperature=0.07)
    model.to(device)

    # Loss weights
    weight_dict = build_weight_dict(args, model.vqa_heads.heads.keys())
    model_ema = deepcopy(model) if args.ema else None
    model_without_ddp = model

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu], find_unused_parameters=False
        )
        model_without_ddp = model.module

    optimizer = build_optimizer(model_without_ddp, args)

    if args.load:
        print("loading from", args.load)
        checkpoint = torch.load(args.load, map_location="cpu")

        model_without_ddp.load_state_dict(checkpoint["model"], strict=False)

    if args.ema:
        model_ema = deepcopy(model_without_ddp)

    print("Start training")
    start_time = time.time()
    best_metric = 0.0
    for epoch in range(args.start_epoch, args.epochs):
        if args.epoch_chunks > 0:
            train_loader = train_loaders[epoch % len(train_loaders)]
            sampler_train = datamodule.samplers_train[epoch % len(train_loaders)]
            print(
                f"Starting epoch {epoch // len(train_loader)}, sub_epoch {epoch % len(train_loaders)}"
            )
        else:
            train_loader = train_loaders
            print(f"Starting epoch {epoch}")
        if args.distributed:
            sampler_train.set_epoch(epoch)
        train_stats = train_one_epoch(
            model=model,
            data_loader=train_loader,
            weight_dict=weight_dict,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            args=args,
            max_norm=args.clip_max_norm,
            model_ema=model_ema,
        )

        if args.output_dir:
            is_main_process = (
                not torch.distributed.is_initialized()
            ) or torch.distributed.get_rank() == 0
            output_dir = Path(args.output_dir)
            checkpoint_paths = [output_dir / "checkpoint.pth"]
            # extra checkpoint before LR drop and every 2 epochs
            if (epoch + 1) % args.lr_drop == 0 or (epoch + 1) % 2 == 0:
                checkpoint_paths.append(output_dir / f"checkpoint{epoch:04}.pth")
            for checkpoint_path in checkpoint_paths:
                if is_main_process:
                    torch.save(
                        {
                            "model": model_without_ddp.state_dict(),
                            "model_ema": model_ema.state_dict() if args.ema else None,
                            "optimizer": optimizer.state_dict(),
                            "epoch": epoch,
                            "args": args,
                        },
                        checkpoint_path,
                    )

        if epoch % args.eval_skip == 0:
            test_stats = {}
            test_model = model_ema if model_ema is not None else model

            curr_test_stats = evaluate(
                model=test_model,
                data_loader=val_loader,
                device=device,
                weight_dict=weight_dict,
                include_contrastive_loss=True,
            )
            test_stats.update(curr_test_stats)
        else:
            test_stats = {}

        log_stats = {
            **{f"train_{k}": v for k, v in train_stats.items()},
            **{f"test_{k}": v for k, v in test_stats.items()},
            "epoch": epoch,
        }

        if args.output_dir and is_main_process:
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

        if epoch % args.eval_skip == 0:
            metric = test_stats["answer_total_accuracy_unscaled"]

            if args.output_dir and metric > best_metric:
                best_metric = metric
                checkpoint_paths = [output_dir / "BEST_checkpoint.pth"]
                # extra checkpoint before LR drop and every 100 epochs
                for checkpoint_path in checkpoint_paths:
                    if is_main_process:
                        torch.save(
                            {
                                "model": model_without_ddp.state_dict(),
                                "optimizer": optimizer.state_dict(),
                                "epoch": epoch,
                                "args": args,
                            },
                            checkpoint_path,
                        )

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("Training time {}".format(total_time_str))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "DETR training and evaluation script", parents=[get_args_parser()]
    )
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
