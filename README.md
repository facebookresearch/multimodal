# TorchMultimodal (Alpha Release)
A brief summary of what TorchMultimodal library is.

## Installation
TorchMultimodal requires Python >= 3.7. The library can be installed with or without CUDA support.

### Building from Source
1. Create conda environment
    ```
    conda create -n torch-multimodal
    conda activate torch-multimodal
    ```
2. Install pytorch, torchvision, and torchtext. See [PyTorch documentation](https://pytorch.org/get-started/locally/).
   ```
   conda install pytorch torchvision torchtext cudatoolkit=11.3 -c pytorch-nightly

   # For CPU-only install
   conda install pytorch torchvision torchtext cpuonly -c pytorch-nightly
   ```
3. Download and install TorchMultimodal and remaining requirements.
    ```
    git clone --recursive https://github.com/facebookresearch/multimodal.git torchmultimodal
    cd torchmultimodal
    pip install -r requirements.txt

    # For development, replace install with develop
    python setup.py install
    ```

## Documentation
...

## Contributing
See the [CONTRIBUTING](CONTRIBUTING.md) file for how to help out.

## License

TorchMultimodal is BSD licensed, as found in the [LICENSE](LICENSE) file.
