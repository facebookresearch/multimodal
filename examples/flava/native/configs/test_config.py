# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from hydra.utils import instantiate
from omegaconf import OmegaConf
from rich import print

cli_conf = OmegaConf.from_cli()
yaml_conf = OmegaConf.load(cli_conf.config)
conf = instantiate(yaml_conf)
conf = OmegaConf.merge(conf, cli_conf)
print(conf.get("somethinginvalid", None))
