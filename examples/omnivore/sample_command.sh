# to run the command, we need to set the path for imagenet, kinetics, and sunrgbd dataset

# command to training omnivore model swin_t
torchrun --nproc_per_node=8 --nnodes=4 train.py \
    --batch-size=128 --workers=5 --model="omnivore_swin_t" \
    --extra-kinetics-dataloader-workers=5 \
    --cache-video-dataset --eval-every-num-epoch=10 \
    --lr=0.002 --lr-warmup-epochs=25 --lr-warmup-method=linear \
    --lr-scheduler="cosineannealinglr" --lr-min=0.0000001 \
    --epochs=500 --weight-decay=0.05 \
    --label-smoothing=0.1 --mixup-alpha=0.2 --cutmix-alpha=1.0 \
    --train-crop-size=224 --val-resize-size=224 \
    --opt="adamw" --random-erase=0.25 \
    --color-jitter-factor 0.4 0.4 0.4 0.4 \
    --model-ema --model-ema-steps=1 --model-ema-decay=0.9999 \
    --video-grad-accum-iter=32 \
    --modalities image video rgbd \
    --val-data-sampling-factor 1 1 1 \
    --train-data-sampling-factor 1 1 10 \
    --imagenet-data-path="${IMAGENET_PATH}" \
    --kinetics-data-path="${KINETICS_PATH}" \
    --sunrgbd-data-path="${SUNRGBD_PATH}" \

# sample command to evaluate the pretrained weight
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
