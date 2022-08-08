# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import json
import os

import torch
import torch.distributed as dist


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


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def init_distributed_mode(args):
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        args["rank"] = int(os.environ["RANK"])
        args["world_size"] = int(os.environ["WORLD_SIZE"])
        args["gpu"] = int(os.environ["LOCAL_RANK"])
    elif "SLURM_PROCID" in os.environ:
        args["rank"] = int(os.environ["SLURM_PROCID"])
        args["gpu"] = args["rank"] % torch.cuda.device_count()
    else:
        print("Not using distributed mode")
        args["distributed"] = False
        return

    args["distributed"] = True

    torch.cuda.set_device(args["gpu"])
    args["dist_backend"] = "nccl"
    print(
        "| distributed init (rank {}): {}".format(args["rank"], args["dist_url"]),
        flush=True,
    )
    torch.distributed.init_process_group(
        backend=args["dist_backend"],
        init_method=args["dist_url"],
        world_size=args["world_size"],
        rank=args["rank"],
    )
    torch.distributed.barrier()
    setup_for_distributed(args["rank"] == 0)


def save_result(result, directory, file_name):
    rank_path = os.path.join(directory, "{}_rank_{}.json".format(file_name, get_rank()))
    main_path = os.path.join(directory, "{}.json".format(file_name))
    json.dump(result, open(rank_path, "w"))

    if is_dist_avail_and_initialized():
        dist.barrier()

    if is_main_process():
        result = []
        for rank in range(get_world_size()):
            rank_path = os.path.join(
                directory, "{}_rank_{}.json".format(file_name, rank)
            )
            rank_res = json.load(open(rank_path, "r"))
            result += rank_res
        json.dump(result, open(main_path, "w"))

    if is_dist_avail_and_initialized():
        dist.barrier()
