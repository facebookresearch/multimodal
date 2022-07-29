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

import examples.utils.distributed as dist_utils
import ruamel.yaml as yaml
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from examples.albef.data.retrieval_datamodule import RetrievalDataModule
from examples.albef.model import albef_model_for_retrieval
from examples.albef.utils import add_weight_decay
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts


def train(model, datamodule, args, checkpoint_root, device):
    model.train()

    model_without_ddp = (
        model.module if dist_utils.is_dist_avail_and_initialized() else model
    )

    optimizer_params = add_weight_decay(model, args["weight_decay"])
    optimizer = AdamW(optimizer_params, lr=args["lr"])
    scheduler = CosineAnnealingWarmRestarts(
        optimizer, T_0=args["max_epochs"], eta_min=args["min_lr"]
    )

    step_size = args["step_size"]
    warmup_steps = args["warmup_steps"]
    warmup_iterations = warmup_steps * step_size

    data_loader = datamodule.train_dataloader(
        is_distributed=dist_utils.is_dist_avail_and_initialized(),
        num_tasks=dist_utils.get_world_size(),
        global_rank=dist_utils.get_rank(),
    )

    dataset_length = len(datamodule.train_dataset)
    start_time = time.time()
    log_every_n_steps = 100

    for epoch in range(args["max_epochs"]):
        if epoch > 0:
            scheduler.step(epoch + warmup_steps)

        for batch, (image, text, text_atts, idx) in enumerate(data_loader):
            if epoch > 0:
                alpha = args["alpha"]
            else:
                alpha = args["alpha"] * min(1, batch / dataset_length)

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

            if batch % log_every_n_steps == 0:
                total_time = time.time() - start_time
                time_str = "time {},".format(
                    datetime.timedelta(seconds=int(total_time))
                )
                epoch_str = "epoch {}/{},".format(epoch, args["max_epochs"])
                batch_str = "batch {}/{},".format(batch, len(data_loader))
                loss_str = "loss {}".format(loss.item())
                print(time_str, epoch_str, batch_str, loss_str)

        save_obj = {
            "model": model_without_ddp.state_dict(),
            "optimizer": optimizer.state_dict(),
            "lr_scheduler": scheduler.state_dict(),
            "epoch": epoch,
        }
        torch.save(
            save_obj,
            os.path.join(checkpoint_root, "retrieval_checkpoint_%02d.pth" % epoch),
        )


@torch.no_grad()
def evaluation(model, datamodule, k, device):
    model.eval()

    text_loader = datamodule.text_dataloader()
    image_loader = datamodule.image_dataloader()
    num_images = len(datamodule.image_dataset)
    num_text = len(datamodule.text_dataset)

    start_time = time.time()
    log_every_n_steps = 100

    text_embeds = []
    text_feats = []
    text_atts = []
    for text, text_att in text_loader:
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

    image_embeds = []
    image_feats = []
    for image in image_loader:
        image = image.to(device)
        image_embed, image_feat = model(image=image, input_type="image", is_train=False)
        image_embeds.append(image_embed)
        image_feats.append(image_feat)
    image_embeds = torch.cat(image_embeds, dim=0)
    image_feats = torch.cat(image_feats, dim=0)

    sims_matrix = image_feats @ text_feats.t()
    image_to_text_scores = torch.full((num_images, num_text), -100.0).to(device)

    num_tasks = dist_utils.get_world_size()
    rank = dist_utils.get_rank()
    step = sims_matrix.size(0) // num_tasks + 1
    start = rank * step
    end = min(sims_matrix.size(0), start + step)

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

        if i % log_every_n_steps == 0:
            total_time = time.time() - start_time
            time_str = "time {},".format(datetime.timedelta(seconds=int(total_time)))
            batch_str = "batch {}/{},".format(i, len(sims_matrix[start:end]))
            print("image to text retrieval", time_str, batch_str)

    sims_matrix = sims_matrix.t()
    text_to_image_scores = torch.full((num_text, num_images), -100.0).to(device)

    step = sims_matrix.size(0) // num_tasks + 1
    start = rank * step
    end = min(sims_matrix.size(0), start + step)

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

        if i % log_every_n_steps == 0:
            total_time = time.time() - start_time
            time_str = "time {},".format(datetime.timedelta(seconds=int(total_time)))
            batch_str = "batch {}/{},".format(i, len(sims_matrix[start:end]))
            print("text to image retrieval", time_str, batch_str)

    if dist_utils.is_dist_avail_and_initialized():
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="./examples/albef/configs/retrieval.yaml")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--world_size", default=1, type=int, help="number of distributed processes"
    )
    parser.add_argument(
        "--dist_url", default="env://", help="url used to set up distributed training"
    )
    args = parser.parse_args()
    config = yaml.load(open(args.config, "r"), Loader=yaml.Loader)

    dist_utils.init_distributed_mode(args)
    device = torch.device(args.device)

    seed = 42 + dist_utils.get_rank()
    torch.manual_seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    datamodule = RetrievalDataModule(**config["datamodule_args"])
    model = albef_model_for_retrieval(config)

    pretrained_checkpoint_path = os.path.join(
        config["checkpoint_root"], config["pretrained_checkpoint"]
    )
    model.load_state_dict(torch.load(pretrained_checkpoint_path, map_location="cpu"))

    model = model.to(device)
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])

    train(model, datamodule, config["training_args"], config["checkpoint_root"], device)
    image_to_text_scores, text_to_image_scores = evaluation(
        model, datamodule, config["k_test"], device
    )
    val_result = itm_eval(
        image_to_text_scores,
        text_to_image_scores,
        datamodule.image_dataset.image_to_text,
        datamodule.text_dataset.text_to_image,
    )
    result = {
        "image_to_text_scores": image_to_text_scores,
        "text_to_image_scores": text_to_image_scores,
        **val_result,
    }
    torch.save(result, config["output_path"])


if __name__ == "__main__":
    main()
