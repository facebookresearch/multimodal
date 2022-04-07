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

Same as in [README](README.md) with the exception of:
```
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

Ideally, flake and ufmt should be run via pre-commit hooks.
But if for some reason you want to run them separately follow this:

```
pip install flake8==4.0.1 ufmt==1.3.0 black==21.4b2 usort==0.6.4
flake8 (examples|test|torchmultimodal)
ufmt format (examples|test|torchmultimodal)
```

Alternatively, you can run on only those files you have modified, e.g.

```
flake8 `git diff main --name-only`
ufmt format `git diff main --name-only`
```


### Unit Tests
Please add unit tests for adding a new feature or a bug-fix. To run a specific test:

```
pytest test/<test-module.py> -vv -k <test_myfunc>
# e.g. pytest test/models/test_cnn_lstm.py -vv -k TestCNNLSTMModule
```

To run all tests:
```
pytest test -vv
```

## License
By contributing to TorchMultimodal, you agree that your contributions will be licensed
under the LICENSE file in the root directory of this source tree.
