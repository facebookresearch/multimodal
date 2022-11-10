# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
import random
from typing import Any

import torch
from flava.data.imagenet_zeroshot_data import (
    imagenet_classnames,
    openai_imagenet_template,
)
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch import distributed as dist
from tqdm import tqdm

# optional syntax-highlighting for console output
try:
    from rich.console import Console

    c = Console(force_terminal=True)
    print = c.log
except ImportError:
    pass


def build_config() -> DictConfig:
    cli_conf = OmegaConf.from_cli()
    yaml_conf = OmegaConf.load(cli_conf.config)
    conf = instantiate(yaml_conf)
    conf = OmegaConf.merge(conf, cli_conf)
    return conf


# TODO replace with tlc.copy_data_to_device
def move_to_device(obj: Any, device: torch.device) -> Any:
    if isinstance(obj, dict):
        d = {}
        for k, v in obj.items():
            d[k] = move_to_device(v, device)
        return d
    if isinstance(obj, list):
        l = []
        for v in obj:
            l.append(move_to_device(v, device))
        return l

    return obj.to(device)


def get_model_size_gb(model: torch.nn.Module) -> int:
    return sum(p.numel() * p.element_size() for p in model.parameters()) / (1024**3)


def get_model_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    random.seed(seed)


def setup_distributed_device() -> torch.device:
    if not torch.cuda.is_available() or not dist.is_available():
        return torch.device("cpu")

    dist.init_process_group("nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    print("local rank", local_rank)
    torch.cuda.set_device(local_rank)
    return torch.device(f"cuda:{local_rank}")


def print0(*args, **kwargs) -> None:
    if not dist.is_initialized() or dist.get_rank() == 0:
        print(*args, **kwargs)


def enable_tf32() -> None:
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True


def rank0_only(func):
    def wrapper(*args, **kwargs):
        if not dist.is_initialized() or dist.get_rank() == 0:
            return func(*args, **kwargs)

    return wrapper


# zero shot classifier functions


def _zero_shot_classifier(model, device, text_transform, *args, **kwargs):
    zeroshot_weights = []
    for classname in tqdm(imagenet_classnames):
        texts = text_transform(
            [template(classname) for template in openai_imagenet_template]
        )["input_ids"]
        texts = texts.to(device)
        class_embeddings = model(texts, action="encode_text")
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


def run_imagenet_zero_shot(model, dataloader, device, text_transform, *args, **kwargs):
    print0("Starting ImageNet Zero-Shot Eval")
    print0("Building classifier")
    classifier = _zero_shot_classifier(model, device, text_transform)
    print0("Classifier built")
    top1, top5, n = 0.0, 0.0, 0.0
    for i, sample in tqdm(enumerate(dataloader)):
        images = sample["image"]
        target = sample["label"]
        images = images.to(device)
        target = target.to(device)

        # predict
        # if hasattr(model, "module"):
        #     image_features = model.module.encode_image({"image": images})
        # else:
        image_features = model(images, action="encode_image")
        image_features /= image_features.norm(dim=-1, keepdim=True)
        logits = 100.0 * image_features @ classifier

        # measure accuracy
        acc1, acc5 = _accuracy(logits, target, topk=(1, 5))
        top1 += acc1
        top5 += acc5
        n += images.size(0)
        if i == 5:
            break

    top1 = top1 / n
    top5 = top5 / n
    results = {}
    results["imagenet-zeroshot-val-top1"] = top1
    results["imagenet-zeroshot-val-top5"] = top5
    print0("results: ", results)
    return results
