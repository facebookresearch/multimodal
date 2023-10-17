# diffusion_labs

Diffusion labs provides components for building diffusion models and for end-to-end training of those models. This
includes definitions for popular models such as

- Dalle2
- Latent Diffusion Models (LDM)

and all the components needed for defining and training these models. All of these modules are compatible with
Pytorch distributed and PT2.

# Concepts

1. Models

This includes diffusion model definitions, like LDM, as well as models used within the diffusion model such as a
U-Net or Transformer. A common model used for denoising within diffusion training is the U-Net from
[ADM](https://arxiv.org/abs/2105.05233), which is available  at `diffusion_labs/models/adm_unet`.

2. Adapters

Adapters adapt the underlying architecture to handle various types of conditional inputs both at training and
inference time. They act as wrappers around the model and multiple adapters can be wrapped around each other to
handle multiple types of inputs. All Adapters have the same `forward` signature allowing them to be stacked.

3. Predictor

Predictor defines what the model is trained to predict (e.g. added noise or a clean image). This is used to convert
the model output into a denoised data point.

4. Schedule

The schedule defines the diffusion process being applied to the data. This includes defining what kind of noise,
and how much noise to apply to each diffusion step. The Schedule class contains the noise values along with any
necessary computations related to it.

5. Sampler

The sampler wraps around the model to denoise the input data given the diffusion schedule. This class takes is
defined with the model, the Predictor and the Schedule as inputs. In train mode the Sampler calls the model for one
step while in eval mode it will call the model for the entire diffusion schedule.


6. Transform

diffusion_labs introduces several helper transforms for diffusion that can be used in conjunction with other data
transforms such as vision transforms. All transforms are implemented as nn.Modules and take in a dict of data and
then output and updated dict. This allows all transforms to be stacked together with nn.Sequential and to be
compiled.


# Tutorial

[How to train diffusion on
MNIST](https://github.com/facebookresearch/multimodal/tree/main/torchmultimodal/diffusion_labs/mnist_training.ipynb)
