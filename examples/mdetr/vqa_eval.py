# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import json
import random
from copy import deepcopy
from pathlib import Path

import examples.mdetr.utils.dist as dist
import numpy as np
import torch
from examples.mdetr.data.datamodule import GQADataModule
from examples.mdetr.loss import build_weight_dict
from examples.mdetr.model import mdetr_for_vqa
from examples.mdetr.utils.args_parse import get_args_parser
from examples.mdetr.utils.metrics import MetricLogger
from examples.mdetr.utils.misc import targets_to


@torch.no_grad()
def evaluate(
    model, data_loader, device, weight_dict, include_contrastive_loss: bool = False
):

    model.eval()
    metric_logger = MetricLogger(delimiter="  ")
    header = "Test:"
    for batch_dict in metric_logger.log_every(data_loader, 10, header):
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
            include_contrastive_loss,
        )

        loss_dict = outputs.loss
        loss_dict_reduced = dist.reduce_dict(loss_dict)
        metric_logger.update(**loss_dict_reduced)

        loss_dict_reduced_scaled = {
            k: v * weight_dict[k]
            for k, v in loss_dict_reduced.items()
            if k in weight_dict
        }
        loss_dict_reduced_unscaled = {
            f"{k}_unscaled": v for k, v in loss_dict_reduced.items()
        }
        metric_logger.update(
            loss=sum(loss_dict_reduced_scaled.values()),
            **loss_dict_reduced_scaled,
            **loss_dict_reduced_unscaled,
        )

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}

    return stats


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
    datamodule.setup("val")
    val_loader = datamodule.val_dataloader()

    # Build the model
    model = mdetr_for_vqa()
    model.to(device)

    # Loss weights
    weight_dict = build_weight_dict(
        args, model.vqa_heads.heads.keys(), include_contrastive_loss=False
    )

    model_ema = deepcopy(model) if args.ema else None
    model_without_ddp = model

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu], find_unused_parameters=True
        )
        model_without_ddp = model.module

    if args.resume.startswith("https"):
        checkpoint = torch.hub.load_state_dict_from_url(
            args.resume, map_location="cpu", check_hash=True
        )
    else:
        checkpoint = torch.load(args.resume, map_location="cpu")

    model_without_ddp.load_state_dict(checkpoint["model"])
    # Load EMA model
    if "model_ema" not in checkpoint:
        print("WARNING: ema model not found in checkpoint, resetting to current model")
        model_ema = deepcopy(model_without_ddp)
    else:
        model_ema.load_state_dict(checkpoint["model_ema"])

    test_model = model_ema if model_ema is not None else model

    test_stats = evaluate(
        model=test_model,
        data_loader=val_loader,
        device=device,
        weight_dict=weight_dict,
    )

    print(test_stats)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "DETR training and evaluation script", parents=[get_args_parser()]
    )
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
