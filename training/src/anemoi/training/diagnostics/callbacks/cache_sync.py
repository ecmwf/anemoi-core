"""Trainer-safe synchronization for distributed dataset cache registries."""

import logging

import pytorch_lightning as pl
LOGGER = logging.getLogger(__name__)


class CacheSyncCallback(pl.Callback):
    """Synchronize cache registries at epoch boundaries on every rank."""

    def __init__(self, cache=None, sync_every_n_epochs: int = 1):
        super().__init__()
        if sync_every_n_epochs < 1:
            raise ValueError("sync_every_n_epochs must be at least 1")
        self.cache = cache
        self.sync_every_n_epochs = sync_every_n_epochs

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if (trainer.current_epoch + 1) % self.sync_every_n_epochs:
            return
        self._sync(trainer)

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if "validate" in str(trainer.state.fn).lower():
            self._sync(trainer)

    def on_test_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if "test" in str(trainer.state.fn).lower():
            self._sync(trainer)

    def _sync(self, trainer: pl.Trainer) -> None:
        cache = self.cache or trainer.datamodule
        if hasattr(cache, "update_global_view"):
            LOGGER.info("Rank %s synchronizing distributed cache registry", trainer.global_rank)
            cache.update_global_view()

    def state_dict(self) -> dict:
        return {"sync_every_n_epochs": self.sync_every_n_epochs}

    def load_state_dict(self, state_dict: dict) -> None:
        self.sync_every_n_epochs = state_dict.get("sync_every_n_epochs", self.sync_every_n_epochs)

    def teardown(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        stage: str,
    ) -> None:
        cache = self.cache or trainer.datamodule
        if hasattr(cache, "teardown"):
            cache.teardown(stage)
