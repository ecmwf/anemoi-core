# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import logging
import time
import uuid
from pathlib import Path

import pytorch_lightning as pl
import torch
import torchinfo
from omegaconf import OmegaConf
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.utilities import rank_zero_only

from anemoi.training.utils.checkpoint import check_classes
from anemoi.utils.checkpoints import save_metadata

LOGGER = logging.getLogger(__name__)


class AnemoiCheckpoint(ModelCheckpoint):
    """A checkpoint callback that saves the model after every validation epoch."""

    def __init__(self, config: OmegaConf, **kwargs: dict) -> None:
        """Initialise the AnemoiCheckpoint callback.

        Parameters
        ----------
        config : OmegaConf
            Config object
        kwargs : dict
            Additional keyword arguments for Pytorch ModelCheckpoint

        """
        super().__init__(**kwargs)

        self.config = config
        self.start = time.time()
        self._model_metadata = None
        self._tracker_metadata = None
        self._tracker_name = None

    @staticmethod
    def _torch_drop_down(trainer: pl.Trainer) -> torch.nn.Module:
        # Get the model from the DataParallel wrapper, for single and multi-gpu cases
        assert hasattr(trainer, "model"), "Trainer has no attribute 'model'! Is the Pytorch Lightning version correct?"
        return trainer.model.module.model if hasattr(trainer.model, "module") else trainer.model.model

    @rank_zero_only
    def model_metadata(self, model: torch.nn.Module) -> dict:
        if self._model_metadata is not None:
            return self._model_metadata

        self._model_metadata = {
            "model": model.__class__.__name__,
            "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "total_parameters": sum(p.numel() for p in model.parameters()),
            "summary": repr(
                torchinfo.summary(
                    model,
                    depth=50,
                    verbose=0,
                    row_settings=["var_names"],
                ),
            ),
        }

        return self._model_metadata

    def _adjust_epoch_progress(self, trainer: pl.Trainer) -> None:
        """Adjust the epoch progress when saving a mid-epoch checkpoint.

        Since Pytorch Lightning advances one epoch at end of training (on_train-end),
        we need to correct the checkpoint epoch progress to avoid inconsistencies.
        """
        trainer.fit_loop.epoch_progress.current.processed = trainer.fit_loop.epoch_progress.current.processed - 1
        trainer.fit_loop.epoch_progress.current.completed = trainer.fit_loop.epoch_progress.current.completed - 1
        trainer.fit_loop.epoch_progress.total.processed = trainer.fit_loop.epoch_progress.total.processed - 1
        trainer.fit_loop.epoch_progress.total.completed = trainer.fit_loop.epoch_progress.total.completed - 1

    def on_train_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Save the last checkpoint at the end of training.

        If the candidates aren't better than the last checkpoint, then no checkpoints are saved.
        Note - this method if triggered when using max_epochs, it won't save any checkpoints
        since the monitor candidates won't show any changes with regard the the 'on_train_epoch_end' hook.
        """
        del pl_module
        if not self._should_skip_saving_checkpoint(trainer) and not self._should_save_on_train_epoch_end(trainer):
            if trainer.fit_loop.epoch_progress.current.completed == trainer.fit_loop.epoch_progress.current.ready:
                self._adjust_epoch_progress(trainer)
            monitor_candidates = self._monitor_candidates(trainer)
            self._save_topk_checkpoint(trainer, monitor_candidates)
            self._save_last_checkpoint(trainer, monitor_candidates)

    def tracker_metadata(self, trainer: pl.Trainer) -> dict:
        if self._tracker_metadata is not None:
            return {self._tracker_name: self._tracker_metadata}

        if self.config.diagnostics.log.wandb.enabled:
            self._tracker_name = "wand"
            import wandb

            run = wandb.run
            if run is not None:
                self._tracker_metadata = {
                    "id": run.id,
                    "name": run.name,
                    "url": run.url,
                    "project": run.project,
                }
            return {self._tracker_name: self._tracker_metadata}

        if self.config.diagnostics.log.mlflow.enabled:
            self._tracker_name = "mlflow"

            from anemoi.training.diagnostics.mlflow.logger import AnemoiMLflowLogger

            mlflow_logger = next(logger for logger in trainer.loggers if isinstance(logger, AnemoiMLflowLogger))
            run_id = mlflow_logger.run_id
            run = mlflow_logger._mlflow_client.get_run(run_id)

            if run is not None:
                self._tracker_metadata = {
                    "id": run.info.run_id,
                    "name": run.info.run_name,
                    "url": run.info.artifact_uri,
                    "project": run.info.experiment_id,
                }
            return {self._tracker_name: self._tracker_metadata}

        return {}

    def _remove_checkpoint(self, trainer: pl.Trainer, filepath: str) -> None:
        """Calls the strategy to remove the checkpoint file."""
        super()._remove_checkpoint(trainer, filepath)
        trainer.strategy.remove_checkpoint(self._get_inference_checkpoint_filepath(filepath))

    def _get_inference_checkpoint_filepath(self, filepath: str) -> str:
        """Defines the filepath for the inference checkpoint."""
        return Path(filepath).parent / Path("inference-" + str(Path(filepath).name))

    def on_train_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Check that model's metadata does not contain Pydantic schemas references."""
        del pl_module

        if trainer.is_global_zero:
            model = self._torch_drop_down(trainer)
            check_classes(model)

    def _save_checkpoint(self, trainer: pl.Trainer, lightning_checkpoint_filepath: str) -> None:
        if trainer.is_global_zero:
            model = self._torch_drop_down(trainer)

            # We want a different uuid each time we save the model
            # so we can tell them apart in the catalogue (i.e. different epochs)
            checkpoint_uuid = str(uuid.uuid4())
            trainer.lightning_module._hparams["metadata"]["uuid"] = checkpoint_uuid

            trainer.lightning_module._hparams["metadata"]["model"] = self.model_metadata(model)
            trainer.lightning_module._hparams["metadata"]["tracker"] = self.tracker_metadata(trainer)

            trainer.lightning_module._hparams["metadata"]["training"] = {
                "current_epoch": trainer.current_epoch,
                "global_step": trainer.global_step,
                "elapsed_time": time.time() - self.start,
            }

            Path(lightning_checkpoint_filepath).parent.mkdir(parents=True, exist_ok=True)

            save_config = model.config
            model.config = None

            tmp_metadata = model.metadata
            model.metadata = None

            tmp_supporting_arrays = model.supporting_arrays
            model.supporting_arrays = None

            # Make sure we don't accidentally modify these
            metadata = tmp_metadata.copy()
            supporting_arrays = tmp_supporting_arrays.copy()

            inference_checkpoint_filepath = self._get_inference_checkpoint_filepath(lightning_checkpoint_filepath)

            torch.save(model, inference_checkpoint_filepath)

            save_metadata(inference_checkpoint_filepath, metadata, supporting_arrays=supporting_arrays)

            model.config = save_config
            model.metadata = tmp_metadata
            model.supporting_arrays = tmp_supporting_arrays

            self._last_global_step_saved = trainer.global_step

        trainer.strategy.barrier()

        # saving checkpoint used for pytorch-lightning based training
        trainer.save_checkpoint(lightning_checkpoint_filepath, self.save_weights_only)

        self._last_global_step_saved = trainer.global_step
        self._last_checkpoint_saved = lightning_checkpoint_filepath

        if trainer.is_global_zero:
            from importlib.metadata import version
            from weakref import proxy

            from packaging.version import Version

            if Version(version("torch")) >= Version("2.6"):
                # Add a new uuid
                checkpoint_uuid = str(uuid.uuid4())
                trainer.lightning_module._hparams["metadata"]["uuid"] = checkpoint_uuid

                # Extract and save metadata for lightning checkpoint
                model = self._torch_drop_down(trainer)
                metadata = model.metadata.copy()
                supporting_arrays = model.supporting_arrays.copy()

                save_metadata(lightning_checkpoint_filepath, metadata, supporting_arrays=supporting_arrays)

            # notify loggers
            for logger in trainer.loggers:
                logger.after_save_checkpoint(proxy(self))
