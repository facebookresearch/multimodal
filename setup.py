#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os
import re
import sys
from datetime import date

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


def get_nightly_version():
    today = date.today()
    return f"{today.year}.{today.month}.{today.day}"


def parse_args(argv):  # Pass in a list of string from CLI
    parser = argparse.ArgumentParser(description="torchmultimodal setup")
    parser.add_argument(
        "--package_name",
        type=str,
        default="torchmultimodal",
        help="The name of this output wheel",
    )
    return parser.parse_known_args(argv)


if __name__ == "__main__":
    args, unknown = parse_args(sys.argv[1:])

    # Set up package name and version
    name = args.package_name
    is_nightly = "nightly" in name

    version = get_nightly_version() if is_nightly else get_version()

    print(f"-- {name} building version: {version}")

    sys.argv = [sys.argv[0]] + unknown

    setup(
        name=name,
        include_package_data=True,
        packages=find_packages(
            exclude=("examples*", "test*")
        ),  # Excluded folders don't get packaged
        python_requires=">=3.7",
        install_requires=read_requirements("requirements.txt"),
        version=version,
        description="PyTorch Multimodal Library",
        long_description=fetch_long_description(),
        long_description_content_type="text/markdown",
        url="https://github.com/facebookresearch/multimodal",
        author="PyTorch Multimodal Team",
        author_email="torchmultimodal@fb.com",
        classifiers=[
            "Programming Language :: Python :: 3.7",
            "Programming Language :: Python :: 3.8",
            "Programming Language :: Python :: 3.9",
            "License :: OSI Approved :: BSD License",
            "Topic :: Scientific/Engineering :: Artificial Intelligence",
        ],
        extras_require={"dev": read_requirements("dev-requirements.txt")},
    )
