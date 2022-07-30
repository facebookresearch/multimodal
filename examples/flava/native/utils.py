import os
import random
from typing import Any

import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf
from torch import distributed as dist

# optional syntax-highlighting for console output
try:
    from rich.console import Console

    c = Console(force_terminal=True)
    print = c.log
except ImportError:
    pass


def build_config():
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
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    random.seed(seed)


def setup_distributed_device() -> None:
    dist.init_process_group("nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    print("local rank", local_rank)
    torch.cuda.set_device(local_rank)
    return torch.device(local_rank)  # TODO work with CPU


def print0(*args, **kwargs):
    if not dist.is_initialized() or dist.get_rank() == 0:
        print(*args, **kwargs)
