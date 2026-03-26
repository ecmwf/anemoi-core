"""Callback to synchronize distributed cache registry after first validation epoch."""

import logging

import pytorch_lightning as pl

LOGGER = logging.getLogger(__name__)


class CacheSyncCallback(pl.Callback):
    """Synchronize cache registry across all ranks after first validation epoch.
    
    This callback ensures all ranks call update_global_view() at the same synchronized point,
    avoiding deadlocks from async calls during data fetching.
    """

    def __init__(self, cache=None):
        super().__init__()
        self.sync_done = False
        self.cache=cache # need to consume cache arg

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Called when the validation epoch ends."""
        if not self.sync_done:
            # Check if datamodule has the cache functionality
            if hasattr(trainer.datamodule, "update_global_view"):
                LOGGER.info(f"Rank {trainer.global_rank}: Synchronizing distributed cache registry... {trainer.datamodule.cache_registry=}")
                trainer.datamodule.update_global_view()
                LOGGER.info(f"Rank {trainer.global_rank}: Cache registry synchronized successfully.")
                self.sync_done = True
            else:
                LOGGER.warning(
                    f"Rank {trainer.global_rank}: DataModule does not have update_global_view() method. "
                    "Skipping cache synchronization."
                )
                self.sync_done = True  # Don't keep trying
