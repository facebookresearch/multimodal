# Table of Contents

<!-- toc -->

- [Contributing to TorchMultimodal](#contributing-to-torchmultimodal)
- [Development Installation](#development-installation)
- [Development Process](#development-process)
- [License](#license)

<!-- tocstop -->

## Contributing to TorchMultimodal
We want to make contributing to this project as easy and transparent as
possible.

### Contributor License Agreement ("CLA")
In order to accept your pull request, we need you to submit a CLA. You only need
to do this once to work on any of Meta's open source projects.

Complete your CLA here: <https://code.facebook.com/cla>

### Issues
We use GitHub issues to track public bugs. Please ensure your description is
clear and has sufficient instructions to be able to reproduce the issue.

Meta has a [bounty program](https://www.facebook.com/whitehat/) for the safe
disclosure of security bugs. In those cases, please go through the process
outlined on that page and do not file a public issue.

## Development Installation

### Install Dependencies

```
conda install pytorch torchvision torchtext cudatoolkit=11.3 -c pytorch-nightly

# For CPU-only install
conda install pytorch torchvision torchtext cpuonly -c pytorch-nightly
```

### Install TorchMultimodal

```
git clone --recursive https://github.com/facebookresearch/multimodal.git torchmultimodal
cd torchmultimodal
pip install -r requirements.txt
python setup.py develop
```

## Development Process

... (in particular how this is synced with internal changes to the project)

### Pull Requests
We actively welcome your pull requests.

1. Fork the repo and create your branch from `main`.
2. If you've added code that should be tested, add tests.
3. If you've changed APIs, update the documentation.
4. Ensure the test suite passes.
5. Make sure your code lints.
6. If you haven't already, complete the Contributor License Agreement ("CLA").

### Coding Style
TorchMultimodal uses pre-commit hooks to ensure style consistency and prevent common mistakes. Enable it by:

```
pip install pre-commit && pre-commit install
```

After this pre-commit hooks will be run before every commit.

Ideally, ufmt should be run via pre-commit hooks.
But if for some reason you want to run ufmt separately follow this:

```
pip install ufmt==1.3.0
ufmt format (examples|test|torchmultimodal)
```

### Unit Tests
Before committing you should verify your changes by running the unit tests. To do so:

```
python3 -m pytest test -v
```

## License
By contributing to TorchMultimodal, you agree that your contributions will be licensed
under the LICENSE file in the root directory of this source tree.
