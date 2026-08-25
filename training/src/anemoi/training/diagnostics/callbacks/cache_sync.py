"""Trainer-safe synchronization for the distributed dataset cache view."""

import logging

import pytorch_lightning as pl
LOGGER = logging.getLogger(__name__)


class CacheSyncCallback(pl.Callback):
    """Synchronize the cache view before validation and checkpointing."""

    def __init__(self, cache=None, sync_every_n_epochs: int = 1):
        super().__init__()
        if sync_every_n_epochs < 1:
            raise ValueError("sync_every_n_epochs must be at least 1")
        self.cache = cache
        self.sync_every_n_epochs = sync_every_n_epochs
        self._last_synced_epoch = None

    def on_validation_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Exchange cache locations before rank-zero checkpoint I/O can separate ranks."""
        if trainer.sanity_checking or "fit" not in str(trainer.state.fn).lower():
            return
        if (trainer.current_epoch + 1) % self.sync_every_n_epochs:
            return
        if self._last_synced_epoch == trainer.current_epoch:
            return
        self._sync(trainer)
        self._last_synced_epoch = trainer.current_epoch

    def _sync(self, trainer: pl.Trainer) -> None:
        cache = self.cache or trainer.datamodule
        if hasattr(cache, "update_global_view"):
            LOGGER.info("Rank %s synchronizing distributed cache view", trainer.global_rank)
            cache.update_global_view()

    def state_dict(self) -> dict:
        return {
            "sync_every_n_epochs": self.sync_every_n_epochs,
            "last_synced_epoch": self._last_synced_epoch,
        }

    def load_state_dict(self, state_dict: dict) -> None:
        self.sync_every_n_epochs = state_dict.get("sync_every_n_epochs", self.sync_every_n_epochs)
        self._last_synced_epoch = state_dict.get("last_synced_epoch")

    def teardown(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        stage: str,
    ) -> None:
        cache = self.cache or trainer.datamodule
        if hasattr(cache, "teardown"):
            cache.teardown(stage)
