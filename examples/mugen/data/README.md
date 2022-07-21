This folder contains code for interfacing the MUGEN dataset (https://mugen-org.github.io/data).

Before using this code,

1. Download the 3.2s-video dataset [here](https://mugen-org.github.io/download) and save as `datasets/coinrun` in your pwd.
2. Download the MUGEN dataset assets [here](https://github.com/mugen-org/MUGEN_baseline/tree/main/lib/data/coinrun/assets) and save under `datasets/coinrun` as `datasets/coinrun/assets` in your pwd.
    * Downloading the assets from GitHub requires `git clone`-ing the original MUGEN repo and copying the assets directory located at `MUGEN_baseline/lib/data/coinrun/assets`.

Note: saving the dataset and assets to locations other than those listed above requires passing custom arguments to `MUGENDataModuleBase` or `MUGENDataset` through `MUGENDatasetArgs.data_path` and `MUGENDatasetArgs.asset_path`, respectively.
