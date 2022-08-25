# MUGEN Retrieval

This directory contains model components for MUGEN's video-text retrieval model, a tutorial notebook for the model usage (better viewed on [nbviewer](https://nbviewer.org/github/facebookresearch/multimodal/blob/main/examples/mugen/retrieval/evaluation.ipynb)), and training and evaluation scripts.

## Model
MUGEN's video-text retrieval model follows from [VideoCLIP](https://arxiv.org/abs/2109.14084), a contrastive model for video and text.

The name "VideoCLIP" refers to its similarities to OpenAI's [CLIP](https://arxiv.org/abs/2103.00020), which was originally proposed for zero-shot learning of image classification tasks by “drawing cues” from text data with the corresponding visual concepts. Unlike various predecessor models based on supervised learning, CLIP does not have to be trained on the task-specific datasets or fine-tuned with a task-specific head. The model learns a joint embedding space for both image and text data and optimizes a scaled cosine similarity function between the image and text embedding vectors. The loss function is the sum of the normalized cosine similarities for every pair of image-and-text samples. Each embedding is trained with a unimodal encoder, e.g., a transformer for text, vision transformer (ViT) or ResNet for image.

The VideoCLIP model follows the CLIP architecture but replaces the image encoder with a video encoder. VideoCLIP's video encoder is backed by [Separable 3D CNN (S3D)](https://arxiv.org/abs/1712.04851), a video classification model, and the text encoder is backed by [DistilBERT](https://arxiv.org/abs/1910.01108), a lightweight transformer for language modeling.

## Prerequisites
Follow the dev setup instructions in [CONTRIBUTING.md](https://github.com/facebookresearch/multimodal/blob/main/CONTRIBUTING.md). Then additionally install MUGEN-specific dependencies:
```
pip install -r examples/mugen/requirements.txt
```
Our training and evaluation scripts rely on the MUGEN dataset. Follow the instructions to download the MUGEN dataset in the [dataset README.md](https://github.com/facebookresearch/multimodal/blob/main/examples/mugen/data/README.md).

## Training
The configurable parameters for training can be found in `configs/train.yaml`. Note that the training script supports training on 1 or more devices on a single node. Then run the following command:
```
python train.py config=configs/train.yaml
```
A checkpoint file with the best-performing weights will be saved under `{default_root_dir}/lightning_logs/`, where `default_root_dir` is specified in the training config. If `default_root_dir` is `null`, then it will act as your working directory.

## Evaluation
The configurable parameters for evaluation can be found in `configs/eval.yaml`. You can choose to replace `checkpoint_path` with the path to your checkpoint from the training step, or keep the default `checkpoint_path` to load the MUGEN authors' weights (fit to our implementation). Then run the following command:
```
python eval.py config=configs/eval.yaml
```

Using the default arguments in `configs/eval.yaml` (including the MUGEN authors' published weights), we ran the evaluation script on the full MUGEN test set and got the following results:

| Metric (%)                | MUGEN Results | TorchMultimodal Results   |
| -----------               | -----------   | -----------               |
| Text2video top-1 recall   | 8.54          | 8.26                      |
| Text2video top-5 recall   | 22.50         | 22.34                     |
| Text2video top-10 recall  | 31.71         | 31.68                     |
| Video2text top-1 recall   | 10.61         | 10.79                     |
| Video2text top-5 recall   | 25.72         | 25.70                     |
| Video2text top-10 recall  | 34.70         | 34.60                     |
