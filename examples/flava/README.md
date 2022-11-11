# FLAVA: A Foundational Language And Vision Alignment Model

## Abstract

State-of-the-art vision and vision-and-language models rely on large-scale visio-linguistic pretraining for obtaining good performance on a variety of downstream tasks. Generally, such models are often either cross-modal (contrastive) or multi-modal (with earlier fusion) but not both; and they often only target specific modalities or tasks. A promising direction would be to use a single holistic universal model, as a "foundation", that targets all modalities at once -- a true vision and language foundation model should be good at vision tasks, language tasks, and cross- and multi-modal vision and language tasks. We introduce FLAVA as such a model and demonstrate impressive performance on a wide range of 35 tasks spanning these target modalities.

## Authors

Amanpreet Singh, Ronghang Hu, Vedanuj Goswami, Guillaume Couairon, Wojciech Galuba, Marcus Rohrbach, Douwe Kiela.


## Training

### Setup repo

You should probably create a [conda](https://docs.conda.io/projects/conda/en/latest/index.html) environment before running the following step.

First, clone the repo, install `multimodal` and then install requirements for this project by using:

```
git clone https://github.com/facebookresearch/multimodal.git
cd multimodal
pip install -e .
cd examples
pip install -r flava/requirements.txt
```

### Access ImageNet

To access the ImageNet dataset, you must first create an account at [HuggingFace](https://huggingface.co/join). Once your account is created and your email is confirmed, log in, click on your profile, and go to Settings -> Access Tokens. Create a new token with READ access and copy it to clipboard. Then run `huggingface-cli login` in your terminal and paste the access token there. It should create an auth token at `~/.huggingface/token` that will be used to authenticate the dataset download request. Finally, visit the [dataset page](https://huggingface.co/datasets/imagenet-1k) and accept the terms and conditions of the dataset while logged into your account.

### Launching and test pretraining

After making sure your access token was saved to `~/.huggingface/token`, launch your FLAVA debug pretraining job by running the following command:

For GPU (default config):
```
python -m flava.train config=flava/configs/pretraining/debug.yaml
```

For CPU (for testing):
```
python -m flava.train config=flava/configs/pretraining/debug.yaml training.lightning.accelerator=cpu training.lightning.gpus=0 training.lightning.strategy=null
```

Note that:
- Running this command will take space on your disk. To change cache or other space options for HuggingFace datasets, please refer their [documentation](https://huggingface.co/docs/datasets/cache).
- The first launch will take time as it extracts ImageNet, but later launches should be fast.

### Configuration

You can update the configuration by changing the config specified by `config` parameter or you can specify the parameters to be overridden by using a dotlist. For example, if you want to run the model with different numbers of training steps, you can do:

```
python -m flava.train config=flava/configs/pretraining/debug.yaml training.lightning.max_steps=1000
```

Similarly, let's say you want to use a pretrained model for your pretraining/finetuning.

```
python -m flava.train config=flava/configs/pretraining/debug.yaml model.pretrained=True
```

### Full Pretraining

<TODO: Depends on addition of PMD which WIP>

### Finetuning

Similarly to pretraining, finetuning can be launched by following command:

```
python -m flava.finetune config=flava/configs/finetuning/qnli.yaml model.pretrained=True
```

### COCO zero shot
Download COCO dataset and annotations to disk

```
python -m flava.coco_zero_shot --data_root <folder with dataset> --annotations <path to annotation json>
```

### Linear Probe

<TODO: Yet to be added>

This model was added by [@apsdehal](https://github.com/apsdehal)
