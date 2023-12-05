[![Unit-tests](https://github.com/facebookresearch/multimodal/actions/workflows/unit_test.yaml/badge.svg)](https://github.com/facebookresearch/multimodal/actions/workflows/unit_test.yaml)
[![Python version](https://img.shields.io/pypi/pyversions/torchmultimodal-nightly.svg)](https://www.python.org/downloads/)
[![Downloads](https://static.pepy.tech/personalized-badge/torchmultimodal-nightly?period=total&units=international_system&left_color=blue&right_color=orange&left_text=Downloads%20(nightly))](https://pepy.tech/project/torchmultimodal-nightly)

# TorchMultimodal (Beta Release)

[**Models**](#models) | [**Example scripts**](#example-scripts) | [**Getting started**](#getting-started) | [**Code overview**](#code-overview) | [**Installation**](#installation) | [**Contributing**](#contributing) | [**License**](#license)

## Introduction
**TorchMultimodal** is a PyTorch library for training state-of-the-art multimodal multi-task models at scale, including both content understanding and generative models. TorchMultimodal contains:
- A repository of modular and composable building blocks (fusion layers, loss functions, datasets and utilities).
- A collection of common multimodal model classes built up from said building blocks with pretrained weights for canonical configurations.
- A set of examples that show how to combine these building blocks with components and common infrastructure from across the PyTorch Ecosystem to replicate state-of-the-art models published in the literature. These examples should serve as baselines for ongoing research in the field, as well as a starting point for future work.

## Models

TorchMultimodal contains a number of models, including

- ALBEF: [model class](https://github.com/facebookresearch/multimodal/blob/main/torchmultimodal/models/albef/model.py#L55), [paper](https://arxiv.org/abs/2107.07651)
- BLIP-2: [model class](https://github.com/facebookresearch/multimodal/blob/main/torchmultimodal/models/blip2/blip2.py#L39), [paper]()
- CLIP: [model class](https://github.com/facebookresearch/multimodal/blob/main/torchmultimodal/models/clip/model.py#L37), [paper](https://arxiv.org/abs/2301.12597)
- CoCa: [model class](https://github.com/facebookresearch/multimodal/blob/main/torchmultimodal/models/coca/coca_model.py#L33), [paper](https://arxiv.org/abs/2205.01917)
- DALL-E 2: [model](https://github.com/facebookresearch/multimodal/blob/main/torchmultimodal/diffusion_labs/models/dalle2/dalle2_decoder.py#L19), [paper](https://arxiv.org/abs/2204.06125)
- FLAVA: [model class](https://github.com/facebookresearch/multimodal/blob/main/torchmultimodal/models/flava/model.py#L106), [paper](https://arxiv.org/abs/2112.04482)
- MAE/Audio MAE: [model class](https://github.com/facebookresearch/multimodal/blob/main/torchmultimodal/models/masked_auto_encoder/model.py#L42), [MAE paper](https://arxiv.org/abs/2111.06377), [Audio MAE paper](https://arxiv.org/abs/2207.06405)
- MDETR: [model class](https://github.com/facebookresearch/multimodal/blob/main/torchmultimodal/models/mdetr/model.py#L37), [paper](https://arxiv.org/abs/2104.12763)

## Example scripts

In addition to the above models, we provide example scripts for training, fine-tuning, and evaluation of models on popular multimodal tasks. Examples can be found under [examples/](https://github.com/facebookresearch/multimodal/tree/main/examples) and include

|                  Model                   |     Supported Tasks     |
| :--------------------------------------: | :----------------------: |
|         ALBEF          |      [Retrieval](https://github.com/facebookresearch/multimodal/blob/main/examples/albef/README.md#retrieval) <br/> [Visual Question Answering](https://github.com/facebookresearch/multimodal/blob/main/examples/albef/README.md#visual-question-answering)         |
|         DDPM           |      [Training and Inference](https://github.com/facebookresearch/multimodal/blob/main/torchmultimodal/diffusion_labs/mnist_training.ipynb) (notebook)
|           FLAVA           |    [Pretraining](https://github.com/facebookresearch/multimodal/tree/main/examples/flava#launching-and-test-pretraining) <br/> [Fine-tuning](https://github.com/facebookresearch/multimodal/tree/main/examples/flava#finetuning) <br/> [Zero-shot](https://github.com/facebookresearch/multimodal/tree/main/examples/flava#coco-zero-shot)|
|        MDETR         |       [Phrase grounding](https://github.com/facebookresearch/multimodal/tree/main/examples/mdetr#phrase-grounding) <br/> [Visual Question Answering](https://github.com/facebookresearch/multimodal/blob/main/examples/mdetr/vqa_finetune.py#L154)        |
|             MUGEN             |     [Text-to-video retrieval](https://github.com/facebookresearch/multimodal/tree/main/examples/mugen/retrieval#mugen-retrieval) <br/> [Text-to-video generation](https://github.com/facebookresearch/multimodal/tree/main/examples/mugen/generation#text-to-video-generation-with-mugen)                |
|           Omnivore           |           [Pre-training](https://github.com/facebookresearch/multimodal/tree/main/examples/omnivore#training) <br/> [Evaluation](https://github.com/facebookresearch/multimodal/tree/main/examples/omnivore#evaluating-pretrained-weight)           |

## Getting started

Below we give minimal examples of how you can write a simple training or zero-shot evaluation script using components from TorchMultimodal.

  <details>
    <summary>FLAVA zero-shot example</summary>

  ```python
import torch
from PIL import Image
from torchmultimodal.models.flava.model import flava_model
from torchmultimodal.transforms.bert_text_transform import BertTextTransform
from torchmultimodal.transforms.flava_transform import FLAVAImageTransform

# Define helper function for zero-shot prediction
def predict(zero_shot_model, image, labels):
    zero_shot_model.eval()
    with torch.no_grad():
        image = image_transform(img)["image"].unsqueeze(0)
        texts = text_transform(labels)
        _, image_features = zero_shot_model.encode_image(image, projection=True)
        _, text_features = zero_shot_model.encode_text(texts, projection=True)
        scores = image_features @ text_features.t()
        probs = torch.nn.Softmax(dim=-1)(scores)
        label = labels[torch.argmax(probs)]
        print(
            "Label probabilities: ",
            {labels[i]: probs[:, i] for i in range(len(labels))},
        )
        print(f"Predicted label: {label}")


image_transform = FLAVAImageTransform(is_train=False)
text_transform = BertTextTransform()
zero_shot_model = flava_model(pretrained=True)
img = Image.open("my_image.jpg")  # point to your own image
predict(zero_shot_model, img, ["dog", "cat", "house"])

# Example output:
# Label probabilities:  {'dog': tensor([0.80590]), 'cat': tensor([0.0971]), 'house': tensor([0.0970])}
# Predicted label: dog
  ```
  </details>

  <details>
    <summary>MAE training example</summary>

  ```python
import torch
from torch.utils.data import DataLoader
from torchmultimodal.models.masked_auto_encoder.model import vit_l_16_image_mae
from torchmultimodal.models.masked_auto_encoder.utils import (
    CosineWithWarmupAndLRScaling,
)
from torchmultimodal.modules.losses.reconstruction_loss import ReconstructionLoss
from torchmultimodal.transforms.mae_transform import ImagePretrainTransform

mae_transform = ImagePretrainTransform()
dataset = MyDatasetClass(transforms=mae_transform)  # you should define this
dataloader = DataLoader(dataset, batch_size=8)

# Instantiate model and loss
mae_model = vit_l_16_image_mae()
mae_loss = ReconstructionLoss()

# Define optimizer and lr scheduler
optimizer = torch.optim.AdamW(mae_model.parameters())
lr_scheduler = CosineWithWarmupAndLRScaling(
    optimizer, max_iters=1000, warmup_iters=100  # you should set these
)

# Train one epoch
for batch in dataloader:
    model_out = mae_model(batch["images"])
    loss = mae_loss(model_out.decoder_pred, model_out.label_patches, model_out.mask)
    loss.backward()
    optimizer.step()
    lr_scheduler.step()
  ```
  </details>

## Code overview

### [torchmultimodal/diffusion_labs](https://github.com/facebookresearch/multimodal/tree/main/torchmultimodal/diffusion_labs)
diffusion_labs contains components for building diffusion models. For more details on these components, see [diffusion_labs/README.md](https://github.com/facebookresearch/multimodal/blob/main/torchmultimodal/diffusion_labs/README.md).

### [torchmultimodal/models](https://github.com/facebookresearch/multimodal/tree/main/torchmultimodal/models)
Look here for model classes as well as any other modeling code specific to a given architecture. E.g. the directory [torchmultimodal/models/blip2](https://github.com/facebookresearch/multimodal/tree/main/torchmultimodal/models/blip2) contains modeling components specific to BLIP-2.

### [torchmultimodal/modules](https://github.com/facebookresearch/multimodal/tree/main/torchmultimodal/modules)
Look here for common generic building blocks that can be stitched together to build a new architecture. This includes [layers](https://github.com/facebookresearch/multimodal/tree/main/torchmultimodal/modules/layers) like [codebooks](https://github.com/facebookresearch/multimodal/blob/main/torchmultimodal/modules/layers/codebook.py#L31), [patch embeddings](https://github.com/facebookresearch/multimodal/blob/main/torchmultimodal/modules/layers/patch_embedding.py#L26), or [transformer encoder/decoders](https://github.com/facebookresearch/multimodal/blob/main/torchmultimodal/modules/layers/transformer.py), [losses](https://github.com/facebookresearch/multimodal/tree/main/torchmultimodal/modules/losses) like [contrastive loss with temperature](https://github.com/facebookresearch/multimodal/blob/main/torchmultimodal/modules/losses/contrastive_loss_with_temperature.py#L121) or [reconstruction loss](https://github.com/facebookresearch/multimodal/blob/main/torchmultimodal/modules/losses/reconstruction_loss.py#L10), [encoders]() like [ViT](https://github.com/facebookresearch/multimodal/blob/main/torchmultimodal/modules/encoders/vision_transformer.py#L20) and [BERT](https://github.com/facebookresearch/multimodal/blob/main/torchmultimodal/modules/encoders/bert_text_encoder.py#L17), and [fusion modules](https://github.com/facebookresearch/multimodal/tree/main/torchmultimodal/modules/fusions) like [Deep Set fusion](https://github.com/facebookresearch/multimodal/blob/main/torchmultimodal/modules/fusions/deepset_fusion.py#L14).

### [torchmultimodal/transforms](https://github.com/facebookresearch/multimodal/tree/main/torchmultimodal/modules)
Look here for common data transforms from popular models, e.g. [CLIP](https://github.com/facebookresearch/multimodal/blob/main/torchmultimodal/transforms/clip_transform.py#L349), [FLAVA](https://github.com/facebookresearch/multimodal/blob/main/torchmultimodal/transforms/flava_transform.py#L206), and [MAE](https://github.com/facebookresearch/multimodal/blob/main/torchmultimodal/transforms/mae_transform.py#L84).

## Installation

TorchMultimodal requires Python >= 3.8. The library can be installed with or without CUDA support.
The following assumes conda is installed.

### Prerequisites
1. Install conda environment

    ```
    conda create -n torch-multimodal python=\<python_version\>
    conda activate torch-multimodal
    ```

2. Install pytorch, torchvision, and torchaudio. See [PyTorch documentation](https://pytorch.org/get-started/locally/).

    ```
    # Use the current CUDA version as seen [here](https://pytorch.org/get-started/locally/)
    # Select the nightly Pytorch build, Linux as the OS, and conda. Pick the most recent CUDA version.
    conda install pytorch torchvision torchaudio pytorch-cuda=\<cuda_version\> -c pytorch-nightly -c nvidia

    # For CPU-only install
    conda install pytorch torchvision torchaudio cpuonly -c pytorch-nightly
    ```

### Install from binaries

Nightly binary on Linux for Python 3.8 and 3.9 can be installed via pip wheels.
For now we only support Linux platform through [PyPI](https://pypi.org/).

```
python -m pip install torchmultimodal-nightly
```

### Building from Source

Alternatively, you can also build from our source code and run our [examples](https://github.com/facebookresearch/multimodal/tree/main/examples):

```
git clone --recursive https://github.com/facebookresearch/multimodal.git multimodal
cd multimodal

pip install -e .
```
For developers please follow the [development installation](https://github.com/facebookresearch/multimodal/blob/main/CONTRIBUTING.md#development-installation).


## Contributing

We welcome any feature requests, bug reports, or pull requests from the community. See the [CONTRIBUTING](CONTRIBUTING.md) file for how to help out.

## License

TorchMultimodal is BSD licensed, as found in the [LICENSE](LICENSE) file.
