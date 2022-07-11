# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Based on https://github.com/pytorch/vision/blob/main/references/classification/train.py

import datetime
import os
import time

import examples.omnivore.data.data_builder as data_builder
import examples.omnivore.utils as utils

import torch
import torch.utils.data
import torchmultimodal.models.omnivore as omnivore
from torch import nn


def _chunk_forward_backward(
    model,
    image,
    target,
    input_type,
    chunk_start,
    chunk_end,
    realized_accum_iter,
    criterion,
    optimizer,
    device,
    args,
    scaler=None,
):

    chunk_image, chunk_target = image[chunk_start:chunk_end, ...].to(device), target[
        chunk_start:chunk_end, ...
    ].to(device)

    with torch.cuda.amp.autocast(enabled=scaler is not None):
        chunk_output = model(chunk_image, input_type)
        loss = criterion(chunk_output, chunk_target)

    # Normalize the loss
    loss /= realized_accum_iter

    if scaler is not None:
        scaler.scale(loss).backward()
        if args.clip_grad_norm is not None:
            # we should unscale the gradients of optimizer's assigned params if do gradient clipping
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
    else:
        loss.backward()
        if args.clip_grad_norm is not None:
            nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)

    return loss, chunk_output


def train_one_epoch(
    model,
    criterion,
    optimizer,
    data_loader,
    device,
    epoch,
    args,
    model_ema=None,
    scaler=None,
):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value}"))
    metric_logger.add_meter("img/s", utils.SmoothedValue(window_size=10, fmt="{value}"))

    data_loader.init_indices(epoch=epoch, shuffle=True)

    header = f"Epoch: [{epoch}]"
    for i, ((image, target), input_type) in enumerate(
        metric_logger.log_every(data_loader, args.print_freq, header)
    ):
        # If input_type is video, we will do "gradient accumulation" to reduce gpu memory usage
        # Each forward-backward call will be done on smaller chunk_size where chunk_size
        # is roughly batch_size divided by number of accumulation iteration
        accum_iter = 1
        if input_type == "video":
            accum_iter = args.video_grad_accum_iter
        start_time = time.time()
        b, c, t, h, w = image.shape

        chunk_start = 0
        # We rounding up chunk_size and realized_accum_iter in case the batch size
        # is not divisible by accum_iter
        chunk_size = (b + accum_iter - 1) // accum_iter
        realized_accum_iter = (b + chunk_size - 1) // chunk_size
        all_chunk_outputs = []
        accum_loss = 0
        for chunk_num in range(realized_accum_iter):
            chunk_end = chunk_start + chunk_size
            if args.distributed and chunk_num < realized_accum_iter - 1:
                # We dont synchronized unless it is the last chunk in DDP mode
                with model.no_sync():
                    loss, chunk_output = _chunk_forward_backward(
                        model,
                        image,
                        target,
                        input_type,
                        chunk_start,
                        chunk_end,
                        realized_accum_iter,
                        criterion,
                        optimizer,
                        device,
                        args,
                        scaler,
                    )
            else:
                loss, chunk_output = _chunk_forward_backward(
                    model,
                    image,
                    target,
                    input_type,
                    chunk_start,
                    chunk_end,
                    realized_accum_iter,
                    criterion,
                    optimizer,
                    device,
                    args,
                    scaler,
                )

            all_chunk_outputs.append(chunk_output)
            accum_loss += loss.item()
            chunk_start = chunk_end

        # Weight update
        if scaler is not None:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()

        optimizer.zero_grad()

        if model_ema and i % args.model_ema_steps == 0:
            model_ema.update_parameters(model)
            if epoch < args.lr_warmup_epochs:
                # Reset ema buffer to keep copying weights during warmup period
                model_ema.n_averaged.fill_(0)

        output = torch.cat(all_chunk_outputs, dim=0)
        target = target.to(device)
        acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))
        batch_size = image.shape[0]
        metric_logger.update(loss=accum_loss, lr=optimizer.param_groups[0]["lr"])
        metric_logger.meters[f"{input_type}_acc1"].update(acc1.item(), n=batch_size)
        metric_logger.meters[f"{input_type}_acc5"].update(acc5.item(), n=batch_size)
        metric_logger.meters["img/s"].update(batch_size / (time.time() - start_time))


def evaluate(
    model, criterion, data_loader, device, args, print_freq=100, log_suffix=""
):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = f"Test: {log_suffix}"

    data_loader.init_indices(epoch=0, shuffle=False)

    num_processed_samples = 0
    with torch.inference_mode():
        for (image, target), input_type in metric_logger.log_every(
            data_loader, print_freq, header
        ):
            # We do the evaluation in chunks to reduce memory usage for video
            accum_iter = 1
            if input_type == "video":
                accum_iter = args.video_grad_accum_iter
            b, c, t, h, w = image.shape

            chunk_start = 0
            chunk_size = (b + accum_iter - 1) // accum_iter
            realized_accum_iter = (b + chunk_size - 1) // chunk_size
            accum_loss = 0
            all_chunk_outputs = []
            for chunk_num in range(realized_accum_iter):
                chunk_end = chunk_start + chunk_size

                chunk_image = image[chunk_start:chunk_end, ...].to(
                    device, non_blocking=True
                )
                chunk_target = target[chunk_start:chunk_end, ...].to(
                    device, non_blocking=True
                )
                chunk_output = model(chunk_image, input_type)
                loss = criterion(chunk_output, chunk_target)

                accum_loss += loss.item()
                all_chunk_outputs.append(chunk_output)
                chunk_start = chunk_end

            output = torch.cat(all_chunk_outputs, dim=0)
            target = target.to(device, non_blocking=True)
            acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))
            # FIXME need to take into account that the datasets
            # could have been padded in distributed setup
            batch_size = image.shape[0]
            metric_logger.update(loss=accum_loss)
            metric_logger.meters[f"{input_type}_acc1"].update(acc1.item(), n=batch_size)
            metric_logger.meters[f"{input_type}_acc5"].update(acc5.item(), n=batch_size)
            num_processed_samples += batch_size

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    for modality in args.modalities:
        try:
            acc1 = getattr(metric_logger, f"{modality}_acc1").global_avg
            acc5 = getattr(metric_logger, f"{modality}_acc5").global_avg
            print(f"{header} {modality} Acc@1 {acc1:.3f} Image Acc@5 {acc5:.3f}")
        except Exception:
            pass


def main(args):
    if args.output_dir:
        utils.mkdir(args.output_dir)

    utils.init_distributed_mode(args)
    print(args)

    device = torch.device(args.device)

    if args.use_deterministic_algorithms:
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
    else:
        torch.backends.cudnn.benchmark = True

    data_loader, data_loader_test = data_builder.get_omnivore_data_loader(args)

    print(f"Creating model: {args.model}")
    model = getattr(omnivore, args.model)()
    model.to(device)

    if args.distributed and args.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    parameters = utils.set_weight_decay(
        model,
        args.weight_decay,
        norm_weight_decay=args.norm_weight_decay,
    )

    opt_name = args.opt.lower()
    if opt_name.startswith("sgd"):
        optimizer = torch.optim.SGD(
            parameters,
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            nesterov="nesterov" in opt_name,
        )
    elif opt_name == "rmsprop":
        optimizer = torch.optim.RMSprop(
            parameters,
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            eps=0.0316,
            alpha=0.9,
        )
    elif opt_name == "adamw":
        optimizer = torch.optim.AdamW(
            parameters, lr=args.lr, weight_decay=args.weight_decay
        )
    else:
        raise RuntimeError(
            f"Invalid optimizer {args.opt}. Only SGD, RMSprop and AdamW are supported."
        )

    scaler = torch.cuda.amp.GradScaler() if args.amp else None

    args.lr_scheduler = args.lr_scheduler.lower()
    if args.lr_scheduler == "steplr":
        main_lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma
        )
    elif args.lr_scheduler == "cosineannealinglr":
        main_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs - args.lr_warmup_epochs, eta_min=args.lr_min
        )
    elif args.lr_scheduler == "exponentiallr":
        main_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer, gamma=args.lr_gamma
        )
    else:
        raise RuntimeError(
            f"Invalid lr scheduler '{args.lr_scheduler}'. Only StepLR, CosineAnnealingLR and ExponentialLR "
            "are supported."
        )

    if args.lr_warmup_epochs > 0:
        if args.lr_warmup_method == "linear":
            warmup_lr_scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=args.lr_warmup_decay,
                total_iters=args.lr_warmup_epochs,
            )
        elif args.lr_warmup_method == "constant":
            warmup_lr_scheduler = torch.optim.lr_scheduler.ConstantLR(
                optimizer,
                factor=args.lr_warmup_decay,
                total_iters=args.lr_warmup_epochs,
            )
        else:
            raise RuntimeError(
                f"Invalid warmup lr method '{args.lr_warmup_method}'. Only linear and constant are supported."
            )
        lr_scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_lr_scheduler, main_lr_scheduler],
            milestones=[args.lr_warmup_epochs],
        )
    else:
        lr_scheduler = main_lr_scheduler

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu], find_unused_parameters=True
        )
        model_without_ddp = model.module

    model_ema = None
    if args.model_ema:
        # Decay adjustment that aims to keep the decay independent from other hyper-parameters originally proposed at:
        # https://github.com/facebookresearch/pycls/blob/f8cd9627/pycls/core/net.py#L123
        #
        # total_ema_updates = (Dataset_size / n_GPUs) * epochs / (batch_size_per_gpu * EMA_steps)
        # We consider constant = Dataset_size for a given dataset/setup and ommit it. Thus:
        # adjust = 1 / total_ema_updates ~= n_GPUs * batch_size_per_gpu * EMA_steps / epochs
        adjust = args.world_size * args.batch_size * args.model_ema_steps / args.epochs
        alpha = 1.0 - args.model_ema_decay
        alpha = min(1.0, alpha * adjust)
        model_ema = utils.ExponentialMovingAverage(
            model_without_ddp, device=device, decay=1.0 - alpha
        )

    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu")
        model_without_ddp.load_state_dict(checkpoint["model"])
        if not args.test_only:
            optimizer.load_state_dict(checkpoint["optimizer"])
            lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
        args.start_epoch = checkpoint["epoch"] + 1
        if model_ema:
            model_ema.load_state_dict(checkpoint["model_ema"])
        if scaler:
            scaler.load_state_dict(checkpoint["scaler"])

    if args.test_only:
        # We disable the cudnn benchmarking because it can noticeably affect the accuracy
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        if model_ema:
            evaluate(
                model_ema,
                criterion,
                data_loader_test,
                device=device,
                args=args,
                log_suffix="EMA",
            )
        else:
            evaluate(model, criterion, data_loader_test, device=device, args=args)
        return

    print("Start training")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        train_one_epoch(
            model,
            criterion,
            optimizer,
            data_loader,
            device,
            epoch,
            args,
            model_ema,
            scaler,
        )
        lr_scheduler.step()
        if epoch % args.num_epoch_per_eval == args.num_epoch_per_eval - 1:
            evaluate(model, criterion, data_loader_test, device=device, args=args)
            if model_ema:
                evaluate(
                    model_ema,
                    criterion,
                    data_loader_test,
                    device=device,
                    args=args,
                    log_suffix="EMA",
                )
        if args.output_dir:
            checkpoint = {
                "model": model_without_ddp.state_dict(),
                "optimizer": optimizer.state_dict(),
                "lr_scheduler": lr_scheduler.state_dict(),
                "epoch": epoch,
                "args": args,
            }
            if model_ema:
                checkpoint["model_ema"] = model_ema.state_dict()
            if scaler:
                checkpoint["scaler"] = scaler.state_dict()
            utils.save_on_master(
                checkpoint, os.path.join(args.output_dir, f"model_{epoch}.pth")
            )
            utils.save_on_master(
                checkpoint, os.path.join(args.output_dir, "checkpoint.pth")
            )

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f"Training time {total_time_str}")


def get_args_parser(add_help=True):
    import argparse

    parser = argparse.ArgumentParser(
        description="Torchmultimodal Omnivore Training", add_help=add_help
    )
    parser.add_argument(
        "--device",
        default="cuda",
        type=str,
        help="device (Use cuda or cpu Default: cuda)",
    )
    parser.add_argument(
        "--model",
        default="omnivore_swin_t",
        type=str,
        help="Model name. Default: 'omnivore_swin_t'",
    )
    parser.add_argument(
        "-b",
        "--batch-size",
        default=32,
        type=int,
        help="images per gpu, the total batch size is $NGPU x batch_size",
    )
    parser.add_argument(
        "--epochs",
        default=90,
        type=int,
        metavar="N",
        help="number of total epochs to run",
    )
    parser.add_argument(
        "-j",
        "--workers",
        default=16,
        type=int,
        metavar="N",
        help="number of data loading workers (default: 16)",
    )
    parser.add_argument("--opt", default="sgd", type=str, help="optimizer")
    parser.add_argument("--lr", default=0.1, type=float, help="initial learning rate")
    parser.add_argument(
        "--momentum", default=0.9, type=float, metavar="M", help="momentum"
    )
    parser.add_argument(
        "--wd",
        "--weight-decay",
        default=1e-4,
        type=float,
        metavar="W",
        help="weight decay (default: 1e-4)",
        dest="weight_decay",
    )
    parser.add_argument(
        "--norm-weight-decay",
        default=None,
        type=float,
        help="weight decay for Normalization layers (default: None, same value as --wd)",
    )
    parser.add_argument(
        "--label-smoothing",
        default=0.0,
        type=float,
        help="label smoothing (default: 0.0)",
        dest="label_smoothing",
    )
    parser.add_argument(
        "--mixup-alpha", default=0.0, type=float, help="mixup alpha (default: 0.0)"
    )
    parser.add_argument(
        "--cutmix-alpha", default=0.0, type=float, help="cutmix alpha (default: 0.0)"
    )
    parser.add_argument(
        "--lr-scheduler",
        default="steplr",
        type=str,
        help="the lr scheduler (default: steplr)",
    )
    parser.add_argument(
        "--lr-warmup-epochs",
        default=0,
        type=int,
        help="the number of epochs to warmup (default: 0)",
    )
    parser.add_argument(
        "--lr-warmup-method",
        default="constant",
        type=str,
        help="the warmup method (default: constant)",
    )
    parser.add_argument(
        "--lr-warmup-decay", default=0.01, type=float, help="the decay for lr"
    )
    parser.add_argument(
        "--lr-step-size",
        default=30,
        type=int,
        help="decrease lr every step-size epochs",
    )
    parser.add_argument(
        "--lr-gamma",
        default=0.1,
        type=float,
        help="decrease lr by a factor of lr-gamma",
    )
    parser.add_argument(
        "--lr-min",
        default=0.0,
        type=float,
        help="minimum lr of lr schedule (default: 0.0)",
    )
    parser.add_argument("--print-freq", default=10, type=int, help="print frequency")
    parser.add_argument(
        "--output-dir", default=".", type=str, help="path to save outputs"
    )
    parser.add_argument("--resume", default="", type=str, help="path of checkpoint")
    parser.add_argument(
        "--start-epoch", default=0, type=int, metavar="N", help="start epoch"
    )
    parser.add_argument(
        "--cache-dataset",
        dest="cache_dataset",
        help="Cache the datasets for quicker initialization. It also serializes the transforms",
        action="store_true",
    )
    parser.add_argument(
        "--sync-bn",
        dest="sync_bn",
        help="Use sync batch norm",
        action="store_true",
    )
    parser.add_argument(
        "--test-only",
        dest="test_only",
        help="Only test the model",
        action="store_true",
    )
    parser.add_argument(
        "--auto-augment",
        default=None,
        type=str,
        help="auto augment policy (default: None)",
    )
    parser.add_argument(
        "--random-erase",
        default=0.0,
        type=float,
        help="random erasing probability (default: 0.0)",
    )

    # Mixed precision training parameters
    parser.add_argument(
        "--amp",
        action="store_true",
        help="Use torch.cuda.amp for mixed precision training",
    )

    # distributed training parameters
    parser.add_argument(
        "--world-size", default=1, type=int, help="number of distributed processes"
    )
    parser.add_argument(
        "--dist-url",
        default="env://",
        type=str,
        help="url used to set up distributed training",
    )
    parser.add_argument(
        "--model-ema",
        action="store_true",
        help="enable tracking Exponential Moving Average of model parameters",
    )
    parser.add_argument(
        "--model-ema-steps",
        type=int,
        default=32,
        help="the number of iterations that controls how often to update the EMA model (default: 32)",
    )
    parser.add_argument(
        "--model-ema-decay",
        type=float,
        default=0.99998,
        help="decay factor for Exponential Moving Average of model parameters (default: 0.99998)",
    )
    parser.add_argument(
        "--use-deterministic-algorithms",
        action="store_true",
        help="Forces the use of deterministic algorithms only.",
    )
    parser.add_argument(
        "--val-resize-size",
        default=256,
        type=int,
        help="the resize size used for validation (default: 256)",
    )
    parser.add_argument(
        "--val-crop-size",
        default=224,
        type=int,
        help="the central crop size used for validation (default: 224)",
    )
    parser.add_argument(
        "--train-crop-size",
        default=224,
        type=int,
        help="the random crop size used for training (default: 224)",
    )
    parser.add_argument(
        "--clip-grad-norm",
        default=None,
        type=float,
        help="the maximum gradient norm (default None)",
    )
    parser.add_argument(
        "--ra-sampler",
        action="store_true",
        help="whether to use Repeated Augmentation in training",
    )
    parser.add_argument(
        "--ra-reps",
        default=3,
        type=int,
        help="number of repetitions for Repeated Augmentation (default: 3)",
    )
    parser.add_argument(
        "--weights", default=None, type=str, help="the weights enum name to load"
    )
    parser.add_argument(
        "--train-resize-size",
        default=256,
        type=int,
        help="the resize size used for training (default: 256)",
    )
    parser.add_argument(
        "--imagenet-data-path", type=str, help="Root directory path of imagenet dataset"
    )
    parser.add_argument(
        "--kinetics-data-path", type=str, help="Root directory path of kinetics dataset"
    )
    parser.add_argument(
        "--sunrgbd-data-path", type=str, help="Root directory path of sunrgbd dataset"
    )
    parser.add_argument(
        "--cache-video-dataset",
        dest="cache_video_dataset",
        help="Cache the video datasets for quicker initialization. It also serializes the transforms",
        action="store_true",
    )
    parser.add_argument(
        "--train-clips-per-video",
        default=1,
        type=int,
        help="maximum number of clips per video to consider during training",
    )
    parser.add_argument(
        "--val-clips-per-video",
        default=4,
        type=int,
        help="maximum number of clips per video to consider during validation",
    )
    parser.add_argument(
        "--kinetics-dataset-workers",
        default=4,
        type=int,
        help="number of kinetics dataset reader workers (default=4)",
    )
    parser.add_argument(
        "--extra-kinetics-dataloader-workers",
        default=8,
        type=int,
        help="number of kinetics data loader workers (default=8)",
    )
    parser.add_argument(
        "--num-epoch-per-eval",
        default=5,
        type=int,
        help="Number of epoch between each evaluation on validation dataset",
    )
    parser.add_argument(
        "--modalities",
        default=["image", "video", "rgbd"],
        type=str,
        nargs="+",
        help="Modalities that will be used in training",
    )
    parser.add_argument(
        "--val-data-sampling-factor",
        default=[1.0, 1.0, 1.0],
        type=float,
        nargs="+",
        help="Sampling factor for validation data for each modality",
    )
    parser.add_argument(
        "--train-data-sampling-factor",
        default=[1.0, 1.0, 10.0],
        type=float,
        nargs="+",
        help="Samping factor for training data for each modality",
    )
    parser.add_argument(
        "--loader-pin-memory",
        help="Pin_memory parameter in data_loader",
        action="store_true",
    )
    parser.add_argument(
        "--color-jitter-factor",
        nargs=4,
        type=float,
        help="Color jitter factor in brightness, contrast, saturation, and hue",
    )
    parser.add_argument(
        "--video-grad-accum-iter",
        type=int,
        default=1,
        help="Number of gradient accumulation iteration to reduce batch size for video",
    )
    parser.add_argument(
        "--loader-drop-last",
        action="store_true",
        help="Drop last parameter in DataLoader",
    )
    return parser


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args)
