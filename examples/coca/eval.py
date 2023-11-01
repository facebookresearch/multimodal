# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os
import random
from pathlib import Path

import numpy as np
import torch
from clip_benchmark.datasets.builder import build_dataset
from data import imagenet_classnames, openai_imagenet_template
from torch.utils.data import DataLoader
from torchmultimodal import _PATH_MANAGER
from torchmultimodal.models.coca.coca_model import coca_vit_l_14_open_clip
from torchmultimodal.transforms.clip_transform import (
    CLIPImageTransform,
    CLIPTextTransform,
)
from tqdm import tqdm


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__

    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop("force", False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def init_distributed_mode(args):
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ["WORLD_SIZE"])
        args.gpu = int(os.environ["LOCAL_RANK"])
    elif "SLURM_PROCID" in os.environ:
        args.rank = int(os.environ["SLURM_PROCID"])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print("Not using distributed mode")
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = "nccl"
    print(
        "| distributed init (rank {}): {}".format(args.rank, args.dist_url),
        flush=True,
    )
    torch.distributed.init_process_group(
        backend=args.dist_backend,
        init_method=args.dist_url,
        world_size=args.world_size,
        rank=args.rank,
    )
    torch.distributed.barrier()
    setup_for_distributed(args.rank == 0)


def get_args_parser():
    parser = argparse.ArgumentParser("CoCa eval", add_help=False)
    parser.add_argument("--device", default=0, type=int, help="GPU id to use")
    parser.add_argument(
        "--world-size", default=1, type=int, help="number of distributed processes"
    )
    parser.add_argument(
        "--dist-url", default="env://", help="init url for distributed training"
    )
    parser.add_argument("--seed", default=42, type=int, help="random seed")
    parser.add_argument(
        "--pretrained", default="", help="path to pretrained checkpoint"
    )
    parser.add_argument("--output_dir", default=".", help="path to save outputs")
    return parser


@torch.no_grad
def _zero_shot_classifier(model, device, text_transform, *args, **kwargs):
    zeroshot_weights = []
    for classname in tqdm(imagenet_classnames):
        texts = text_transform(
            [template(classname) for template in openai_imagenet_template]
        )
        texts = texts.to(device)
        class_embeddings = model.encode_text(texts)
        class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
        class_embedding = class_embeddings.mean(dim=0)
        class_embedding /= class_embedding.norm()
        zeroshot_weights.append(class_embedding)

    zeroshot_weights = torch.stack(zeroshot_weights, dim=1).to(device)
    return zeroshot_weights


def _accuracy(output, target, topk=(1,)):
    pred = output.topk(max(topk), 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    return [
        float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy())
        for k in topk
    ]


@torch.no_grad
def run_imagenet_zero_shot(model, dataloader, device, text_transform, *args, **kwargs):
    print("Starting ImageNet Zero-Shot Eval")
    print("Building classifier")
    classifier = _zero_shot_classifier(model, device, text_transform)
    print("Classifier built")
    top1, top5, n = 0.0, 0.0, 0.0
    i = 0
    for sample in tqdm(dataloader):
        i = i + 1
        images, target = sample
        images = images.to(device)
        target = target.to(device)

        image_features = model.encode_image(images)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        logits = 100.0 * image_features @ classifier

        # measure accuracy
        acc1, acc5 = _accuracy(logits, target, topk=(1, 5))
        top1 += acc1
        top5 += acc5
        n += images.size(0)

    top1 = top1 / n
    top5 = top5 / n
    results = {}
    results["imagenet-zeroshot-test-top1"] = top1
    results["imagenet-zeroshot-test-top5"] = top5
    return results


def accuracy(output, target, topk=(1,)):
    pred = output.topk(max(topk), 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    return [
        float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy())
        for k in topk
    ]


def run_open_clip_zero_shot():
    baseline_command = """\
    clip_benchmark eval --pretrained_model \"/data/users/ebs/models.txt\" \
    --dataset \"/data/users/ebs/webdatasets.txt\" \
    --dataset_root \"https://huggingface.co/datasets/clip-benchmark/wds_{dataset_cleaned}/tree/main\" \
    --output "benchmark_{dataset}_{pretrained}_{model}_{language}_{task}.json" \
    """
    print("Running open_clip zero-shot on imagenet")
    os.system(baseline_command)


def run_torchmultimodal_zero_shot(device):
    print("defining transform")
    transform = CLIPImageTransform(is_train=False)
    print("building dataset")
    tmm_dataset = build_dataset(
        dataset_name="wds/imagenet1k",
        root="https://huggingface.co/datasets/clip-benchmark/wds_imagenet1k/tree/main",
        transform=transform,
        split="test",
        annotation_file="",
        download=True,
        language="en",
        task="zeroshot_classification",
        custom_template_file=None,
        custom_classname_file=None,
        wds_cache_dir=None,
    )
    dataloader = DataLoader(tmm_dataset, batch_size=64)

    # Build the model
    print("building model")
    model = coca_vit_l_14_open_clip()
    model.to(device)
    if args.pretrained:
        print("loading checkpoint")
        with _PATH_MANAGER.open(args.pretrained, "rb") as f:
            weights = torch.load(f)
        model.load_state_dict(weights)
    model.eval()

    text_transform = CLIPTextTransform()
    tmm_out = run_imagenet_zero_shot(model, dataloader, device, text_transform)
    return tmm_out


def main(args):
    # Init distributed mode
    init_distributed_mode(args)

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
    torch.use_deterministic_algorithms(True, warn_only=True)

    tmm_out = run_torchmultimodal_zero_shot(device)
    print(f"TorchMultimodal zero-shot accuracy: {tmm_out}")
    run_open_clip_zero_shot()


if __name__ == "__main__":
    parser = argparse.ArgumentParser("CoCa zero-shot eval", parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
