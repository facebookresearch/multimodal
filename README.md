# TorchMultimodal (Beta Release)

## Introduction
TorchMultimodal is a PyTorch library for training state-of-the-art multimodal multi-task models at scale. It provides:
- A repository of modular and composable building blocks (models, fusion layers, loss functions, datasets and utilities).
- A set of examples that show how to combine these building blocks with components and common infrastructure from across the PyTorch Ecosystem to replicate state-of-the-art models published in the literature. These examples should serve as baselines for ongoing research in the field, as well as a starting point for future work.
- [diffusion_labs](https://github.com/facebookresearch/multimodal/tree/main/torchmultimodal/diffusion_labs), a library of components with examples for training popular diffusion models.


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
