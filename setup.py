#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
import re

from setuptools import setup, find_packages


def clean_html(raw_html):
    cleanr = re.compile("<.*?>")
    cleantext = re.sub(cleanr, "", raw_html).strip()
    return cleantext


def _get_version():
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


DISTNAME = "torchmultimodal"
DESCRIPTION = "Multimodal modeling in PyTorch"
LONG_DESCRIPTION = fetch_long_description()
LONG_DESCRIPTION_CONTENT_TYPE = "text/markdown"
AUTHOR = "PyTorch Multimodal"
AUTHOR_EMAIL = "kartikayk@fb.com"
# Need to exclude folders in test as well so as they don't create an extra package
EXCLUDES = ("examples", "test")

if __name__ == "__main__":
    setup(
        name=DISTNAME,
        include_package_data=True,
        packages=find_packages(exclude=EXCLUDES),
        python_requires=">=3.7",
        version=_get_version(),
        description=DESCRIPTION,
        long_description=LONG_DESCRIPTION,
        long_description_content_type=LONG_DESCRIPTION_CONTENT_TYPE,
        url="https://github.com/facebookresearch/multimodal",
        author=AUTHOR,
        author_email=AUTHOR_EMAIL,
        classifiers=[
            "Programming Language :: Python :: 3.7",
            "Programming Language :: Python :: 3.8",
            "Programming Language :: Python :: 3.9",
            "License :: OSI Approved :: BSD License",
            "Topic :: Scientific/Engineering :: Artificial Intelligence",
        ],
    )
