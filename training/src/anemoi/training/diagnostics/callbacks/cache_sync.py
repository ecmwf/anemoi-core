"""Callback to synchronize distributed cache registry after first validation epoch."""

import logging

import pytorch_lightning as pl

from anemoi.training.utils.dataset_cache import DatasetCache

LOGGER = logging.getLogger(__name__)


class CacheSyncCallback(pl.Callback):
    """Synchronize cache registry across all ranks after first validation epoch.

    This callback ensures all ranks call update_global_view() at the same synchronized point,
    avoiding deadlocks from async calls during data fetching.
    """

    def __init__(self, cache: object) -> None:
        super().__init__()
        self.sync_done = False
        self.cache = cache  # need to consume cache arg

    def on_train_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:  # noqa: ARG002
        """Initialise the distributed cache in the main process on every rank.

        ``DatasetCache.cache_setup()`` performs collective operations (new_group, all_gather,
        barrier) so it must run once per rank in the main training process, after the
        distributed process group has been initialised. It is *not* triggered by the
        DataLoader because the loader iterates the underlying ``MultiDataset`` directly,
        never the cache wrapper.
        """
        datamodule = trainer.datamodule
        if isinstance(datamodule, DatasetCache) and not datamodule.is_initalised:
            LOGGER.info("Rank %s: Initialising distributed dataset cache...", trainer.global_rank)
            datamodule.cache_setup()
            LOGGER.info("Rank %s: Distributed dataset cache initialised.", trainer.global_rank)

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:  # noqa: ARG002

        if not self.sync_done:
            # Check if datamodule has the cache functionality
            if hasattr(trainer.datamodule, "update_global_view"):
                LOGGER.info(
                    "Rank %s: Synchronizing distributed cache registry... %s",
                    trainer.global_rank,
                    trainer.datamodule.cache_registry,
                )
                trainer.datamodule.update_global_view()
                LOGGER.info("Rank %s: Cache registry synchronized successfully.", trainer.global_rank)
                self.sync_done = True
            else:
                LOGGER.warning(
                    "Rank %s: DataModule does not have update_global_view() method. Skipping cache synchronization.",
                    trainer.global_rank,
                )
                self.sync_done = True  # Don't keep trying
