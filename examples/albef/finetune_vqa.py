# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import datetime
import os
import random
import time

import ruamel.yaml as yaml
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from data.vqa_datamodules import VQADataModule
from model import albef_model_for_vqa
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from utils import (
    add_weight_decay,
    get_rank,
    get_world_size,
    init_distributed_mode,
    is_dist_avail_and_initialized,
    is_main_process,
)


def train(model, datamodule, args, device):
    model_without_ddp = model.module if is_dist_avail_and_initialized() else model
    model.train()

    optimizer_params = add_weight_decay(model, args["weight_decay"])
    optimizer = AdamW(optimizer_params, lr=args["lr"])
    scheduler = CosineAnnealingWarmRestarts(
        optimizer, T_0=args["max_epochs"], eta_min=args["min_lr"]
    )

    step_size = args["step_size"]
    warmup_steps = args["warmup_steps"]
    warmup_iterations = warmup_steps * step_size

    data_loader = datamodule.train_dataloader(
        is_distributed=is_dist_avail_and_initialized(),
        num_tasks=get_world_size(),
        global_rank=get_rank(),
    )

    start_time = time.time()

    for epoch in range(args["max_epochs"]):
        if is_dist_avail_and_initialized():
            data_loader.sampler.set_epoch(epoch)

        if epoch > 0:
            scheduler.step(epoch + warmup_steps)

        for batch, (
            images,
            questions,
            questions_atts,
            answers,
            answers_atts,
            ans_weights,
            ans_lengths,
        ) in enumerate(data_loader):
            if epoch > 0:
                alpha = args["alpha"]
            else:
                alpha = args["alpha"] * min(1, batch / len(data_loader))

            images = images.to(device, non_blocking=True)
            questions = questions.to(device)
            questions_atts = questions_atts.to(device)
            answers = answers.to(device)
            answers_atts = answers_atts.to(device)
            ans_weights = ans_weights.to(device)

            loss = model(
                images,
                questions,
                questions_atts,
                answers,
                answers_atts,
                ans_weights=ans_weights,
                ans_lengths=ans_lengths,
                alpha=alpha,
                is_train=True,
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if epoch == 0 and batch % step_size == 0 and batch <= warmup_iterations:
                scheduler.step(batch // step_size)

            if batch % args["log_every_n_steps"] == 0:
                total_time = time.time() - start_time
                time_str = "time {},".format(
                    datetime.timedelta(seconds=int(total_time))
                )
                epoch_str = "epoch {}/{},".format(epoch, args["max_epochs"])
                batch_str = "batch {}/{},".format(batch, len(data_loader))
                loss_str = "loss {}".format(loss.item())
                print(time_str, epoch_str, batch_str, loss_str)

        if is_main_process():
            save_obj = {
                "model": model_without_ddp.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "epoch": epoch,
            }
            torch.save(
                save_obj,
                os.path.join(args["checkpoint_root"], "vqa_checkpoint_%02d.pt" % epoch),
            )

        if is_dist_avail_and_initialized():
            dist.barrier()


@torch.no_grad()
def evaluation(model, datamodule, args, device):
    model.eval()

    result = []

    answer_list = datamodule.test_dataset.answer_list
    answer_input_ids = datamodule.test_dataset.answer_input_ids.to(device)
    answer_atts = datamodule.test_dataset.answer_attention_mask.to(device)
    data_loader = datamodule.test_dataloader(
        is_distributed=is_dist_avail_and_initialized(),
        num_tasks=get_world_size(),
        global_rank=get_rank(),
    )

    start_time = time.time()

    for batch, (img, ques, ques_atts, ques_ids) in enumerate(data_loader):
        img = img.to(device, non_blocking=True)
        ques = ques.to(device)
        ques_atts = ques_atts.to(device)

        topk_ids, topk_probs = model(
            img,
            ques,
            ques_atts,
            answer_input_ids,
            answer_atts,
            k=args["k_test"],
            is_train=False,
        )

        for ques_id, topk_id, topk_prob in zip(ques_ids, topk_ids, topk_probs):
            _, pred = topk_prob.max(dim=0)
            result.append(
                {"question_id": ques_id, "answer": answer_list[topk_id[pred]]}
            )

        if batch % args["log_every_n_steps"] == 0:
            total_time = time.time() - start_time
            total_time_str = str(datetime.timedelta(seconds=int(total_time)))
            print(
                "time {}, batch {}/{}".format(total_time_str, batch, len(data_loader))
            )

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="./examples/albef/configs/vqa.yaml")
    args = parser.parse_args()
    config = yaml.load(open(args.config, "r"), Loader=yaml.Loader)

    init_distributed_mode(config)
    device = torch.device(config["device"])

    seed = config["seed"] + get_rank()
    torch.manual_seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    datamodule = VQADataModule(**config["datamodule_args"])
    model = albef_model_for_vqa(config, pretrained=True)
    model = model.to(device)
    if is_dist_avail_and_initialized():
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[config["gpu"]]
        )

    train(model, datamodule, config["training_args"], device)
    result = evaluation(model, datamodule, config["eval_args"], device)
    save_result(result, config["output_root"], "vqa_output")


if __name__ == "__main__":
    main()
