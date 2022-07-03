# to run the command, we need to set the path for imagenet, kinetics, and sunrgbd dataset

torchrun --nproc_per_node=8 --nnodes=4 train.py \
    --batch-size=128 --workers=5 \
    --cache-video-dataset \
    --lr=0.002 --lr-warmup-epochs=5 --lr-warmup-method=linear \
    --epochs=500 --weight-decay=0.05 \
    --label-smoothing=0.1 --mixup-alpha=0.2 --cutmix-alpha=1.0 \
    --train-crop-size=176 --val-resize-size=232 \
    --extra-kinetics-dataloader-workers=5 \
    --opt="adamw" \
    --random-erase=0.1 \
    --color-jitter-factor 0.1 0.1 0.1 0.1 \
    --video-grad-accum-iter=32 \
    --modalities image video rgbd \
    --val-data-sampling-factor 1 1 1 \
    --train-data-sampling-factor 1 1 10 \
    --imagenet-data-path="${IMAGENET_PATH}" \
    --kinetics-data-path="${KINETICS_PATH}" \
    --sunrgbd-data-path="${SUNRGBD_PATH}" \
