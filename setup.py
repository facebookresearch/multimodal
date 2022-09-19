#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
import re

from setuptools import find_packages, setup


def clean_html(raw_html):
    cleanr = re.compile("<.*?>")
    cleantext = re.sub(cleanr, "", raw_html).strip()
    return cleantext


def get_version():
    # get version string from version.py
    version_file = os.path.join(os.path.dirname(__file__), "version.py")
    version_regex = r"__version__ = ['\"]([^'\"]*)['\"]"
    with open(version_file, "r") as f:
        version = re.search(version_regex, f.read(), re.M).group(1)
        return version


def fetch_long_description():
    with open("README.md", encoding="utf8") as f:
        readme = f.read()
        # https://stackoverflow.com/a/12982689
        readme = clean_html(readme)
    return readme


def read_requirements(file):
    with open(file) as f:
        reqs = f.read()

    return reqs.strip().split("\n")


if __name__ == "__main__":

    setup(
        name="torchmultimodal",
        include_package_data=True,
        packages=find_packages(exclude=("examples*", "test*")),  # Excluded folders don't get packaged
        python_requires=">=3.7",
        install_requires=read_requirements("requirements.txt"),
        version=get_version(),
        description="PyTorch Multimodal Library",
        long_description=fetch_long_description(),
        long_description_content_type="text/markdown",
        url="https://github.com/facebookresearch/multimodal",
        author="PyTorch Multimodal Team",
        author_email="kartikayk@fb.com",  # TODO: Get a group email address to manage packaging
        classifiers=[
            "Programming Language :: Python :: 3.7",
            "Programming Language :: Python :: 3.8",
            "Programming Language :: Python :: 3.9",
            "License :: OSI Approved :: BSD License",
            "Topic :: Scientific/Engineering :: Artificial Intelligence",
        ],
        extras_require={"dev": read_requirements("dev-requirements.txt")},
    )
