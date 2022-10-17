# Text-to-Video Generation with MUGEN

This directory contains the high-level model components for text-to-video generation following [MUGEN](https://arxiv.org/abs/2204.08058). They demonstrate how to use building blocks from TorchMultimodal to quickly assemble a new auto-regressive generative model for different pairs of modalities. Here is a [colab demo](https://colab.research.google.com/drive/1C3ZbH_l19g_KqW3CPeX2-8Q2sOUCpmZo?usp=sharing) shows how to generate a video clip from text prompts.

https://user-images.githubusercontent.com/23155714/196074330-6f03593c-da8e-473f-8935-8bf1950baa33.mp4


## Model
The model architecture used by MUGEN follows [DALL-E](https://arxiv.org/abs/2102.12092) but with the image components replaced by those for video following [VideoGPT](https://arxiv.org/abs/2104.10157).

Multimodal generation involves generation of samples in one modality given inputs from another. As in the text-to-image generation model DALL-E, it typically involves a two-stage process of first learning a discrete latent representation for each modality and then using a [GPT](https://openai.com/blog/language-unsupervised/) transformer decoder to learn a joint prior for both modalities in the latent space. For text data, the latent representation is obtained through tokenization such as [BPE](https://en.wikipedia.org/wiki/Byte_pair_encoding) used in this example. For high dimensional data such as video and image, a [VQ-VAE](https://arxiv.org/abs/1711.00937) model is used to learn a set of downsampled discrete embedding vectors through nearest-neighbor lookups from a "codebook" where the chosen indices are referred to as the token ids following convention from language modeling.

VideoGPT is a generative model for video using a VQ-VAE model with video encoder/decoder and a GPT transformer decoder for token generation. The encoder and the decoder use 3D-convolution and self axial-attention to learn video information.

## Generation
In this example generation refers to the auto-regressive process where we iteratively predict the next token id from the current until reaching the desired output length, a technique initially used by language modeling but has been extended to multimodal generation. To control the generation process, a top level abstraction is provided as a utility in [generate.py](https://github.com/facebookresearch/multimodal/blob/main/torchmultimodal/utils/generate.py) which takes the model as an input.
