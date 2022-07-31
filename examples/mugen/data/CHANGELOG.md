# Change Log

The format of this document is based on [Keep a Changelog](http://keepachangelog.com/).

## [Unreleased] - 2022-07-26

Changes are noted with respect to the [original MUGEN dataset API](https://github.com/mugen-org/MUGEN_baseline/tree/main/lib/data).

### Added
- Added `README.md` instructions on how to download assets.
- Added optional data transform arguments to data module.

### Changed
- Replaced command-line arguments with `MUGENDatasetArgs` and direct arguments to `MugenDataModule`.
- Renamed `data.py` --> `mugen_datamodules.py` and renamed `VideoData` class to `MUGENDataModule`.
- Renamed `mugen_data.py` --> `mugen_dataset.py`.
- Replaced dependencies on OpenAI's `jukebox` and MUGEN's `audio_vqvae` by placing relevant functionality in `audio_utils.py`.

### Removed
- Removed `data/coinrun/assets` folder.
