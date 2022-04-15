# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


def _get_path_manager():
    try:
        from torchmultimodal.fb.utils.file_io import (
            register_handlers,
            FBPathManager,
        )

        pm = FBPathManager()
        register_handlers(pm)
        return pm
    except ImportError:
        from iopath.common.file_io import PathManager

        pm = PathManager()
        return pm
