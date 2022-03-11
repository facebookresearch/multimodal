from functools import partial

from data import (
    ImageDataModule,
    MLMDataModule,
    HFDatasetsInfo,
    VLDataModule,
    MultiDataModule,
)
from model import FLAVALightningModule
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor


AVAIL_GPUS = 2
SEED = 1234

IMAGENET_TRAIN_ROOT = ""
IMAGENET_VAL_ROOT = ""
NUM_WORKERS = 4
MAX_STEPS = 450000
BATCH_SIZE = 8
ALLOW_UNEVEN_BATCHES = False


def main():
    # TODO: Check this later, since this is causing all workers to load same data.
    # seed_everything(SEED, workers=True)
    dm = ImageDataModule(
        train_root=IMAGENET_TRAIN_ROOT,
        val_root=IMAGENET_VAL_ROOT,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        allow_unenven_batchs=ALLOW_UNEVEN_BATCHES,
    )
    mlm_datamodule = MLMDataModule(
        [HFDatasetsInfo("wikitext", "wikitext-103-raw-v1")],
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        allow_unenven_batchs=ALLOW_UNEVEN_BATCHES,
    )
    vl_datamodule = MultiDataModule(
        [
            VLDataModule(
                train_dataset_infos=[
                    HFDatasetsInfo(
                        key="red_caps",
                        subset="mycology",
                        rename_columns=[("caption", "text")],
                    )
                ],
                val_dataset_infos=[
                    HFDatasetsInfo(
                        key="red_caps",
                        subset="mycology",
                        rename_columns=[("caption", "text")],
                        split_key_mapping={"validation": "train"},
                    )
                ],
                batch_size=BATCH_SIZE,
                num_workers=NUM_WORKERS,
                allow_unenven_batchs=ALLOW_UNEVEN_BATCHES,
            )
        ]
    )
    datamodule = MultiDataModule([dm, mlm_datamodule, vl_datamodule])

    datamodule.setup("fit")
    model = FLAVALightningModule()

    trainer = Trainer(
        max_steps=MAX_STEPS,
        gpus=AVAIL_GPUS,
        progress_bar_refresh_rate=50,
        callbacks=[LearningRateMonitor(logging_interval="step")],
    )
    trainer.fit(model, datamodule=datamodule)


if __name__ == "__main__":
    main()
