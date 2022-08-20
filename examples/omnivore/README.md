# [Omnivore: A Single Model for Many Visual Modalities](https://arxiv.org/abs/2201.08377)

## Abstract

Prior work has studied different visual modalities in isolation and developed separate architectures for recognition of images, videos, and 3D data. Instead, in this paper, we propose a single model which excels at classifying images, videos, and single-view 3D data using exactly the same model parameters. Our 'Omnivore' model leverages the flexibility of transformer-based architectures and is trained jointly on classification tasks from different modalities. Omnivore is simple to train, uses off-the-shelf standard datasets, and performs at-par or better than modality-specific models of the same size. A single Omnivore model obtains 86.0% on ImageNet, 84.1% on Kinetics, and 67.1% on SUN RGB-D. After finetuning, our models outperform prior work on a variety of vision tasks and generalize across modalities. Omnivore's shared visual representation naturally enables cross-modal recognition without access to correspondences between modalities. We hope our results motivate researchers to model visual modalities together.

## Authors

Rohit Girdhar, Mannat Singh, Nikhila Ravi, Laurens van der Maaten, Armand Joulin, Ishan Misra

## Install Python Package to Run Training and Evaluation

```
# Install nightly pytorch core and domains
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch-nightly

# Install pytorch MultiModal (see [reference](https://github.com/facebookresearch/multimodal/blob/main/README.md))
cd <torchmultimodal root folder>
pip install -e .

# Other dependency
pip install scipy==1.8.1 av==9.2.0
```

## Datasets

For both examples on training and evaluation, we use ImageNet1K, Kinetics400, and SunRGBD datasets.
Here are the expected folder structure for each dataset:

### ImageNet1K
```
root
    train
        class_1
            image_1.jpeg
            image_2.jpeg
            ...
        class_2
            image_1.jpeg
            image_2.jpeg
            ...
        ...
    val
        class_1
            image_1.jpeg
            image_2.jpeg
            ...
        class_2
            image_1.jpeg
            image_2.jpeg
            ...
        ...
```

### Kinetics400
```
root
    train
        class_1
            video_1.mp4
            video_2.mp4
            ...
        class_2
            video_1.mp4
            video_2.mp4
            ...
        ...
```

### SunRGBD
```
root
    SUNRGBD
        kv1
            NYUdata
                image_dir_1
                    image
                        image_1.jpg
                    depth_bfx
                        depth_1.png
                    intrinsics.txt
                    scene.txt
                ...
            ...
        kv2
            align_kv2
                image_dir_1
                    image
                        image_1.jpg
                    depth_bfx
                        depth_1.png
                    intrinsics.txt
                    scene.txt
                ...
            ...
        ...
    SUNRGBDtoolbox
        traintestSUNRGBD
            allsplit.mat
            ...
        ...
```

## Training

Example command to run training on ImageNet1K, Kinetics400, and SunRGBD datasets
```
torchrun --nproc_per_node=8 --nnodes=8 train.py \
    --batch-size=128 --workers=6 --extra-video-dataloader-workers=6 \
    --cache-video-dataset --eval-every-num-epoch=10 --model="omnivore_swin_t" \
    --lr=0.002 --lr-warmup-epochs=25 --lr-warmup-method=linear \
    --lr-scheduler="cosineannealinglr" --lr-min=0.0000001 \
    --epochs=500 --weight-decay=0.05 \
    --label-smoothing=0.1 --mixup-alpha=0.8 --cutmix-alpha=1.0 \
    --train-crop-size=224 --val-resize-size=224 \
    --opt="adamw" --random-erase=0.25 \
    --color-jitter-factor 0.4 0.4 0.4 0.4 \
    --model-ema --model-ema-steps=1 --model-ema-decay=0.99998 \
    --video-grad-accum-iter=32 \
    --modalities image video rgbd \
    --val-data-sampling-factor 1 1 1 \
    --train-data-sampling-factor 1 1 10 \
    --imagenet-data-path="${IMAGENET_PATH}" \
    --kinetics-data-path="${KINETICS_PATH}" \
    --sunrgbd-data-path="${SUNRGBD_PATH}" \
```

## Evaluating Pretrained Weight

Example command to run evaluation on Omnivore with Swin Transformer Tiny variant.
```
torchrun --nproc_per_node=8 --nnodes=1 train.py \
    --batch-size=128 --workers=6 --mode="omnivore_swin_t" \
    --cache-video-dataset \
    --extra-video-dataloader-workers=7 \
    --kinetics-dataset-workers=12 \
    --val-resize-size=224 \
    --video-grad-accum-iter=32 \
    --modalities image video rgbd \
    --val-data-sampling-factor 1 1 1 \
    --test-only --pretrained \
    --imagenet-data-path="${IMAGENET_PATH}" \
    --kinetics-data-path="${KINETICS_PATH}" \
    --sunrgbd-data-path="${SUNRGBD_PATH}" \
```

We get the following result for the evaluation:
```
[2022-07-27 17:52:40,277] INFO - Test:  image Acc@1 81.056 image Acc@5 95.366
[2022-07-27 17:52:40,298] INFO - Test:  Clip Acc@1 70.967 Clip Acc@5 88.766
[2022-07-27 17:52:40,298] INFO - Test:  Video Acc@1 77.985 Video Acc@5 93.462
[2022-07-27 17:52:40,298] INFO - Test:  rgbd Acc@1 61.923 rgbd Acc@5 87.809
```
