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
from data.retrieval_datamodule import RetrievalDataModule
from model import albef_model_for_retrieval
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
    model.train()

    model_without_ddp = model.module if is_dist_avail_and_initialized() else model

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
        if epoch > 0:
            scheduler.step(epoch + warmup_steps)

        for batch, (image, text, text_atts, idx) in enumerate(data_loader):
            if epoch > 0:
                alpha = args["alpha"]
            else:
                alpha = args["alpha"] * min(1, batch / len(data_loader))

            image = image.to(device, non_blocking=True)
            text = text.to(device)
            text_atts = text_atts.to(device)
            idx = idx.to(device, non_blocking=True)
            loss = model(image, text, text_atts, idx, alpha, is_train=True)

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
                "lr_scheduler": scheduler.state_dict(),
                "epoch": epoch,
            }
            torch.save(
                save_obj,
                os.path.join(
                    args["checkpoint_root"], "retrieval_checkpoint_%02d.pt" % epoch
                ),
            )

        if is_dist_avail_and_initialized():
            dist.barrier()
            torch.cuda.empty_cache()


@torch.no_grad()
def encode_text(model, text_dataloader, device):
    text_embeds = []
    text_feats = []
    text_atts = []
    for text, text_att in text_dataloader:
        text = text.to(device)
        text_att = text_att.to(device)
        text_embed, text_feat = model(
            text=text, text_atts=text_att, input_type="text", is_train=False
        )
        text_embeds.append(text_embed)
        text_feats.append(text_feat)
        text_atts.append(text_att)
    text_embeds = torch.cat(text_embeds, dim=0)
    text_feats = torch.cat(text_feats, dim=0)
    text_atts = torch.cat(text_atts, dim=0)
    return text_embeds, text_feats, text_atts


@torch.no_grad()
def encode_image(model, image_dataloader, device):
    image_embeds = []
    image_feats = []
    for image in image_dataloader:
        image = image.to(device)
        image_embed, image_feat = model(image=image, input_type="image", is_train=False)
        image_embeds.append(image_embed)
        image_feats.append(image_feat)
    image_embeds = torch.cat(image_embeds, dim=0)
    image_feats = torch.cat(image_feats, dim=0)
    return image_embeds, image_feats


@torch.no_grad()
def image_to_text(
    model,
    image_embeds,
    text_embeds,
    text_atts,
    sims_matrix,
    num_images,
    num_text,
    device,
    args,
):
    start_time = time.time()
    world_size = get_world_size()
    rank = get_rank()
    step = sims_matrix.size(0) // world_size + 1
    start = rank * step
    end = min(sims_matrix.size(0), start + step)
    k = args["k_test"]

    image_to_text_scores = torch.full((num_images, num_text), -100.0).to(device)
    for i, sims in enumerate(sims_matrix[start:end]):
        _, topk_idx = sims.topk(k, dim=0)

        score = model(
            image=image_embeds[start + i].repeat(k, 1, 1),
            text=text_embeds[topk_idx],
            text_atts=text_atts[topk_idx],
            input_type="multimodal",
            is_train=False,
        )
        image_to_text_scores[start + i, topk_idx] = score

        if i % args["log_every_n_steps"] == 0:
            total_time = time.time() - start_time
            time_str = "time {},".format(datetime.timedelta(seconds=int(total_time)))
            batch_str = "batch {}/{},".format(i, len(sims_matrix[start:end]))
            print("image to text retrieval", time_str, batch_str)
    return image_to_text_scores


@torch.no_grad()
def text_to_image(
    model,
    image_embeds,
    text_embeds,
    text_atts,
    sims_matrix,
    num_images,
    num_text,
    device,
    args,
):
    start_time = time.time()
    world_size = get_world_size()
    rank = get_rank()
    step = sims_matrix.size(0) // world_size + 1
    start = rank * step
    end = min(sims_matrix.size(0), start + step)
    k = args["k_test"]

    text_to_image_scores = torch.full((num_text, num_images), -100.0).to(device)
    for i, sims in enumerate(sims_matrix[start:end]):
        _, topk_idx = sims.topk(k, dim=0)
        score = model(
            image=image_embeds[topk_idx],
            text=text_embeds[start + i].repeat(k, 1, 1),
            text_atts=text_atts[start + i].repeat(k, 1, 1),
            input_type="multimodal",
            is_train=False,
        )
        text_to_image_scores[start + i, topk_idx] = score

        if i % args["log_every_n_steps"] == 0:
            total_time = time.time() - start_time
            time_str = "time {},".format(datetime.timedelta(seconds=int(total_time)))
            batch_str = "batch {}/{},".format(i, len(sims_matrix[start:end]))
            print("text to image retrieval", time_str, batch_str)
    return text_to_image_scores


@torch.no_grad()
def evaluation(model, datamodule, args, device):
    model.eval()

    text_loader = datamodule.text_dataloader()
    image_loader = datamodule.image_dataloader()
    num_images = len(datamodule.image_dataset)
    num_text = len(datamodule.text_dataset)

    text_embeds, text_feats, text_atts = encode_text(model, text_loader, device)
    image_embeds, image_feats = encode_image(model, image_loader, device)

    sims_matrix = image_feats @ text_feats.t()
    image_to_text_scores = image_to_text(
        model,
        image_embeds,
        text_embeds,
        text_atts,
        sims_matrix,
        num_images,
        num_text,
        device,
        args,
    )

    sims_matrix = sims_matrix.t()
    text_to_image_scores = text_to_image(
        model,
        image_embeds,
        text_embeds,
        text_atts,
        sims_matrix,
        num_images,
        num_text,
        device,
        args,
    )

    if is_dist_avail_and_initialized():
        dist.barrier()
        torch.distributed.all_reduce(
            image_to_text_scores, op=torch.distributed.ReduceOp.SUM
        )
        torch.distributed.all_reduce(
            text_to_image_scores, op=torch.distributed.ReduceOp.SUM
        )

    return image_to_text_scores.cpu(), text_to_image_scores.cpu()


@torch.no_grad()
def itm_eval(
    image_to_text_scores,
    text_to_image_scores,
    image_to_text_mapping,
    text_to_image_mapping,
):
    # Images to Text
    ranks = torch.zeros(image_to_text_scores.size(0))
    for index, score in enumerate(image_to_text_scores):
        inds = torch.flip(torch.argsort(score), dims=[0])
        rank = 1e10
        # each image has multiple text mappings
        # check retrieved inds with each ground truth mappping i
        for i in image_to_text_mapping[index]:
            tmp = torch.where(inds == i)[0][0]
            if tmp < rank:
                rank = tmp
        ranks[index] = rank

    # Compute metrics
    tr1 = 100.0 * len(torch.where(ranks < 1)[0]) / len(ranks)
    tr5 = 100.0 * len(torch.where(ranks < 5)[0]) / len(ranks)
    tr10 = 100.0 * len(torch.where(ranks < 10)[0]) / len(ranks)

    # Text to Images
    ranks = torch.zeros(text_to_image_scores.size(0))
    for index, score in enumerate(text_to_image_scores):
        inds = torch.flip(torch.argsort(score), dims=[0])
        ranks[index] = torch.where(inds == text_to_image_mapping[index])[0][0]

    # Compute metrics
    ir1 = 100.0 * len(torch.where(ranks < 1)[0]) / len(ranks)
    ir5 = 100.0 * len(torch.where(ranks < 5)[0]) / len(ranks)
    ir10 = 100.0 * len(torch.where(ranks < 10)[0]) / len(ranks)

    tr_mean = (tr1 + tr5 + tr10) / 3
    ir_mean = (ir1 + ir5 + ir10) / 3
    r_mean = (tr_mean + ir_mean) / 2

    eval_result = {
        "txt_r1": tr1,
        "txt_r5": tr5,
        "txt_r10": tr10,
        "txt_r_mean": tr_mean,
        "img_r1": ir1,
        "img_r5": ir5,
        "img_r10": ir10,
        "img_r_mean": ir_mean,
        "r_mean": r_mean,
    }
    return eval_result


@torch.no_grad()
def format_output(
    image_to_text_scores,
    text_to_image_scores,
    image_dataset,
    text_dataset,
):
    image_to_text_output = {}
    for index, score in enumerate(image_to_text_scores):
        image = image_dataset.images[index]
        top10_ids = torch.flip(torch.argsort(score), dims=[0])[:10]
        top10_text = [text_dataset.text[i] for i in top10_ids]
        image_to_text_output[index] = {
            "image": image,
            "output": top10_text,
        }
    text_to_image_output = {}
    for index, score in enumerate(text_to_image_scores):
        text = text_dataset.text[index]
        top10_ids = torch.flip(torch.argsort(score), dims=[0])[:10]
        top10_images = [image_dataset.images[i] for i in top10_ids]
        text_to_image_output[index] = {
            "text": text,
            "output": top10_images,
        }
    return image_to_text_output, text_to_image_output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="./examples/albef/configs/retrieval.yaml")
    args = parser.parse_args()
    config = yaml.load(open(args.config, "r"), Loader=yaml.Loader)

    init_distributed_mode(config)
    device = torch.device(config["device"])

    seed = config["seed"] + get_rank()
    torch.manual_seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    datamodule = RetrievalDataModule(**config["datamodule_args"])
    model = albef_model_for_retrieval(config, pretrained=True)
    model = model.to(device)
    if is_dist_avail_and_initialized():
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[config["gpu"]]
        )

    train(model, datamodule, config["training_args"], device)
    image_to_text_scores, text_to_image_scores = evaluation(
        model, datamodule, config["eval_args"], device
    )
    val_result = itm_eval(
        image_to_text_scores,
        text_to_image_scores,
        datamodule.image_dataset.image_to_text,
        datamodule.text_dataset.text_to_image,
    )
    image_to_text_output, text_to_image_output = format_output(
        image_to_text_scores,
        text_to_image_scores,
        datamodule.image_dataset,
        datamodule.text_dataset,
    )
    result = {
        "image_to_text_output": image_to_text_output,
        "text_to_image_output": text_to_image_output,
        **val_result,
    }
    torch.save(result, config["output_path"])


if __name__ == "__main__":
    main()
