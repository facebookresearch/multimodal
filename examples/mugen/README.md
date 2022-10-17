# MUGEN: A Playground for Video-Audio-Text Multimodal Understanding and GENeration

Implementation of [MUGEN](https://arxiv.org/abs/2204.08058) in TorchMultimodal for text-video [retrieval](https://github.com/facebookresearch/multimodal/tree/main/examples/mugen/retrieval) and [generation](https://github.com/facebookresearch/multimodal/tree/main/examples/mugen/generation).

Authors: Thomas Hayes, Songyang Zhang, Xi Yin, Guan Pang, Sasha Sheng, Harry Yang, Songwei Ge, Qiyuan Hu, Devi Parikh


## Prerequisites
Follow the dev setup instructions in [CONTRIBUTING.md](https://github.com/facebookresearch/multimodal/blob/main/CONTRIBUTING.md). Then additionally install MUGEN-specific dependencies:
```
pip install -r examples/mugen/requirements.txt
```
MUGEN dataset is required for most of the scripts/demos in this folder. Follow the instructions to download the MUGEN dataset in the [dataset README.md](https://github.com/facebookresearch/multimodal/blob/main/examples/mugen/data/README.md).
