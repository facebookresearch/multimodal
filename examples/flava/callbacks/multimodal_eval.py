# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging

import torch
from data import default_text_transform, VL_MAX_LENGTH_DEFAULT
from imagenet_zeroshot_data import imagenet_classnames, openai_imagenet_template
from pytorch_lightning import Callback, LightningDataModule
from pytorch_lightning.utilities import rank_zero_only
from tqdm import tqdm


logger = logging.getLogger(__name__)


def _zero_shot_classifier(model, device, text_transform, *args, **kwargs):
    zeroshot_weights = []
    for classname in tqdm(imagenet_classnames):
        texts = text_transform(
            [template(classname) for template in openai_imagenet_template]
        )["input_ids"]
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


@rank_zero_only
def run_imagenet_zero_shot(model, dataloader, device, text_transform, *args, **kwargs):
    logger.info("Starting ImageNet Zero-Shot Eval")
    logger.info("Building classifier")
    classifier = _zero_shot_classifier(model, device, text_transform)
    logger.info("Classifier built")
    top1, top5, n = 0.0, 0.0, 0.0
    for images, target in tqdm(dataloader):
        images = images["image"].to(device)
        target = target.to(device)

        # predict
        # if hasattr(model, "module"):
        #     image_features = model.module.encode_image({"image": images})
        # else:
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
    results["imagenet-zeroshot-val-top1"] = top1
    results["imagenet-zeroshot-val-top5"] = top5
    return results


class MultimodalEvalCallback(Callback):
    def __init__(self, imagenet_datamodule: LightningDataModule, *args, **kwargs):
        super().__init__()
        self.imagenet_val_dataloader = imagenet_datamodule.val_dataloader()
        self.text_transform = default_text_transform(
            max_text_length=VL_MAX_LENGTH_DEFAULT
        )

    @torch.no_grad()
    def on_validation_start(self, trainer, pl_module, **kwargs) -> None:
        metrics = run_imagenet_zero_shot(
            pl_module.model,
            self.imagenet_val_dataloader,
            pl_module.device,
            self.text_transform,
        )
        if metrics is not None:
            for key in metrics:
                self.log(
                    f"val/{key}",
                    metrics[key],
                    prog_bar=True,
                    logger=True,
                    rank_zero_only=True,
                )
