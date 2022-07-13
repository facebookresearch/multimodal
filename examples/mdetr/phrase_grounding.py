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
from examples.mdetr.data.datamodule import FlickrDataModule
from examples.mdetr.data.flickr_eval import FlickrEvaluator
from examples.mdetr.data.postprocessors import PostProcessFlickr
from examples.mdetr.utils.args_parse import get_args_parser
from examples.mdetr.utils.checkpoint import map_mdetr_state_dict
from examples.mdetr.utils.metrics import MetricLogger
from examples.mdetr.utils.misc import targets_to
from torchmultimodal.models.mdetr import mdetr_resnet101


@torch.no_grad()
def evaluate(
    model,
    postprocessor,
    data_loader,
    evaluator,
    device,
):

    model.eval()

    metric_logger = MetricLogger(delimiter="  ")
    header = "Test:"

    for batch_dict in metric_logger.log_every(data_loader, 10, header):
        samples = [x.to(device) for x in batch_dict["samples"]]
        targets = batch_dict["targets"]
        text = [t["tokenized"].to(device) for t in targets]
        targets = targets_to(targets, device)
        outputs = model(images=samples, text=text)

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        flickr_res = []
        image_ids = [t["original_img_id"] for t in targets]
        sentence_ids = [t["sentence_id"] for t in targets]
        phrases_per_sample = [t["nb_eval"] for t in targets]
        positive_map_eval = batch_dict["positive_map_eval"].to(device)
        flickr_results = postprocessor(
            outputs,
            orig_target_sizes,
            positive_map_eval,
            phrases_per_sample,
        )
        assert len(flickr_results) == len(image_ids) == len(sentence_ids)
        for im_id, sent_id, output in zip(image_ids, sentence_ids, flickr_results):
            flickr_res.append(
                {"image_id": im_id, "sentence_id": sent_id, "boxes": output}
            )

        evaluator.update(flickr_res)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    evaluator.synchronize_between_processes()
    flickr_res = evaluator.summarize()

    return flickr_res


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
    output_dir = Path(args.output_dir)

    # fix the seed for reproducibility
    seed = args.seed + dist.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.use_deterministic_algorithms(True)

    # Set up datamodule
    datamodule = FlickrDataModule(args)
    datamodule.setup("val")
    val_loader = datamodule.val_dataloader()

    # Build the model
    model = mdetr_resnet101(args.num_queries, args.num_classes)
    model.to(device)

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

    mapped_state_dict = map_mdetr_state_dict(checkpoint["model"], model.state_dict())
    model_without_ddp.load_state_dict(mapped_state_dict)
    # Load EMA model
    if "model_ema" not in checkpoint:
        print("WARNING: ema model not found in checkpoint, resetting to current model")
        model_ema = deepcopy(model_without_ddp)
    else:
        ema_mapped_state_dict = map_mdetr_state_dict(
            checkpoint["model"], model.state_dict()
        )
        model_ema.load_state_dict(ema_mapped_state_dict)

    test_model = model_ema if model_ema is not None else model

    # Construct evaluator
    evaluator = FlickrEvaluator(
        args.flickr_dataset_path,
        subset="test" if args.test else "val",
        merge_boxes=args.GT_type == "merged",
    )

    postprocessor = PostProcessFlickr()
    test_stats = evaluate(
        model=test_model,
        postprocessor=postprocessor,
        data_loader=val_loader,
        evaluator=evaluator,
        device=device,
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
