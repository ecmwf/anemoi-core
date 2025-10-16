# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Checkpoint pipeline integration callback for PyTorch Lightning."""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING
from typing import Any

from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities import rank_zero_only

if TYPE_CHECKING:
    import pytorch_lightning as pl
    from omegaconf import DictConfig

    from anemoi.training.checkpoint.base import CheckpointContext
    from anemoi.training.checkpoint.pipeline import CheckpointPipeline

LOGGER = logging.getLogger(__name__)


class CheckpointPipelineCallback(Callback):
    """PyTorch Lightning callback that integrates with the checkpoint pipeline system.

    This callback enables checkpoint loading and model modifications using the
    new pipeline system during training setup, while maintaining compatibility
    with existing PyTorch Lightning workflows.
    """

    def __init__(self, config: DictConfig):
        """Initialize the checkpoint pipeline callback.

        Parameters
        ----------
        config : DictConfig
            Configuration object containing pipeline settings
        """
        super().__init__()
        self.config = config
        self._pipeline = None
        self._context = None

    def _has_pipeline_config(self) -> bool:
        """Check if checkpoint pipeline configuration is present."""
        return (
            hasattr(self.config.training, "checkpoint_pipeline")
            and self.config.training.checkpoint_pipeline is not None
            and hasattr(self.config.training.checkpoint_pipeline, "stages")
            and len(self.config.training.checkpoint_pipeline.stages) > 0
        )

    def _create_pipeline(self) -> CheckpointPipeline | None:  # type: ignore[name-defined]
        """Create the checkpoint pipeline from configuration."""
        if not self._has_pipeline_config():
            return None

        try:
            from anemoi.training.checkpoint import CheckpointPipeline

            pipeline = CheckpointPipeline.from_config(self.config.training.checkpoint_pipeline)
            LOGGER.info("Created checkpoint pipeline with %d stages", len(pipeline))
            return pipeline  # noqa: TRY300  # Return on success, exception handler below

        except ImportError as e:
            LOGGER.warning("Checkpoint pipeline not available: %s. Falling back to standard checkpoint loading.", e)
            return None
        except (ValueError, TypeError, KeyError) as e:
            LOGGER.warning("Failed to create checkpoint pipeline: %s. Falling back to standard checkpoint loading.", e)
            return None

    @rank_zero_only
    def on_fit_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Execute checkpoint pipeline at the start of training."""
        if not self._has_pipeline_config():
            LOGGER.debug("No checkpoint pipeline configuration found, skipping pipeline execution")
            return

        self._pipeline = self._create_pipeline()
        if self._pipeline is None:
            return

        try:
            # Extract the actual model from the Lightning module
            model = pl_module.model if hasattr(pl_module, "model") else pl_module

            # Get optimizer and scheduler if available
            optimizer = trainer.optimizers[0] if trainer.optimizers else None
            scheduler = trainer.lr_schedulers[0] if trainer.lr_schedulers else None

            # Create checkpoint context
            from anemoi.training.checkpoint import CheckpointContext

            self._context = CheckpointContext(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                config=self.config,
                metadata={},
                pl_module=pl_module,  # Store Lightning module for potential updates
            )

            # Execute pipeline
            if asyncio.iscoroutinefunction(self._pipeline.execute):
                # Handle async execution
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    result = loop.run_until_complete(self._pipeline.execute(self._context))
                finally:
                    loop.close()
            else:
                # Handle sync execution
                result = self._pipeline.execute(self._context)

            # Update components from pipeline results
            self._update_training_state_from_context(trainer, pl_module, result)

            # Log pipeline execution results
            if result.metadata:
                LOGGER.info("Checkpoint pipeline executed successfully. Metadata: %s", result.metadata)
            else:
                LOGGER.info("Checkpoint pipeline executed successfully")

        except (ImportError, ValueError, TypeError, KeyError, RuntimeError):
            LOGGER.exception("Checkpoint pipeline execution failed")
            # Don't fail training, just log the error

    def _update_model_state(
        self,
        pl_module: pl.LightningModule,
        context: CheckpointContext,  # type: ignore[name-defined]
    ) -> None:
        """Update model from pipeline context."""
        if context.model is not None:
            original_model = pl_module.model if hasattr(pl_module, "model") else pl_module
            if context.model is not original_model:
                if hasattr(pl_module, "model"):
                    pl_module.model = context.model
                    LOGGER.debug("Updated Lightning module model from pipeline")
                else:
                    LOGGER.warning("Pipeline modified model but cannot update Lightning module directly")

    def _update_optimizer_scheduler(
        self,
        trainer: pl.Trainer,
        context: CheckpointContext,  # type: ignore[name-defined]
    ) -> None:
        """Update optimizer and scheduler from pipeline context."""
        if context.optimizer is not None:
            trainer.optimizers = [context.optimizer]
            LOGGER.debug("Updated trainer optimizer from pipeline")

        if context.scheduler is not None:
            trainer.lr_schedulers = [context.scheduler]
            LOGGER.debug("Updated trainer scheduler from pipeline")

    def _restore_training_metadata(
        self,
        trainer: pl.Trainer,
        context: CheckpointContext,  # type: ignore[name-defined]
    ) -> None:
        """Restore training metadata from checkpoint."""
        if context.metadata.get("epoch") is not None:
            trainer.fit_loop.epoch_progress.current.processed = context.metadata["epoch"]
            trainer.current_epoch = context.metadata["epoch"]
            LOGGER.debug("Restored epoch from checkpoint: %s", context.metadata["epoch"])

        if context.metadata.get("global_step") is not None:
            trainer.global_step = context.metadata["global_step"]
            trainer.fit_loop.epoch_loop.automatic_optimization.optim_progress.optimizer.step.total.completed = (
                context.metadata["global_step"]
            )
            LOGGER.debug("Restored global step from checkpoint: %s", context.metadata["global_step"])

        if context.metadata.get("best_metric") is not None:
            for callback in trainer.callbacks:
                if hasattr(callback, "best_model_score"):
                    callback.best_model_score = context.metadata["best_metric"]
                    LOGGER.debug("Restored best metric from checkpoint: %s", context.metadata["best_metric"])
                    break

    def _update_training_state_from_context(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        context: CheckpointContext,  # type: ignore[name-defined]
    ) -> None:
        """Update training state from pipeline execution results."""
        self._update_model_state(pl_module, context)
        self._update_optimizer_scheduler(trainer, context)
        self._restore_training_metadata(trainer, context)

    def on_save_checkpoint(
        self,
        _trainer: pl.Trainer,
        _pl_module: pl.LightningModule,
        checkpoint: dict,
    ) -> None:
        """Enhance saved checkpoint with pipeline metadata."""
        if self._context and self._context.metadata:
            # Add pipeline metadata to checkpoint
            checkpoint["checkpoint_pipeline_metadata"] = self._context.metadata.copy()

            # Add pipeline version info
            checkpoint["checkpoint_pipeline_version"] = "1.0"

            LOGGER.debug("Added pipeline metadata to checkpoint")

    def state_dict(self) -> dict[str, Any]:  # type: ignore[name-defined]
        """Return callback state for checkpointing."""
        return {
            "pipeline_metadata": self._context.metadata if self._context else {},
            "has_pipeline_config": self._has_pipeline_config(),
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:  # type: ignore[name-defined]
        """Restore callback state from checkpoint."""
        if "pipeline_metadata" in state_dict:
            # Restore metadata if we have a context
            if self._context:
                self._context.metadata.update(state_dict["pipeline_metadata"])
            else:
                # Store for later use
                self._stored_metadata = state_dict["pipeline_metadata"]

    def on_load_checkpoint(
        self,
        _trainer: pl.Trainer,
        _pl_module: pl.LightningModule,
        checkpoint: dict,
    ) -> None:
        """Handle checkpoint loading in Lightning's checkpoint restoration."""
        if not self._has_pipeline_config():
            return

        # Check if this checkpoint was created by our pipeline system
        if "checkpoint_pipeline_metadata" in checkpoint:
            LOGGER.debug("Loading checkpoint with pipeline metadata")

            # Store pipeline metadata for potential use
            pipeline_metadata = checkpoint["checkpoint_pipeline_metadata"]
            if self._context:
                self._context.metadata.update(pipeline_metadata)
            else:
                # Create a temporary context to store metadata
                from anemoi.training.checkpoint import CheckpointContext

                self._context = CheckpointContext(metadata=pipeline_metadata)

            LOGGER.debug("Restored pipeline metadata from checkpoint: %s", pipeline_metadata)

    def on_train_start(
        self,
        _trainer: pl.Trainer,
        _pl_module: pl.LightningModule,
    ) -> None:
        """Additional setup at training start (after checkpoint loading)."""
        # This runs after on_load_checkpoint, so we can handle any final state restoration
        if self._context and self._context.metadata:
            # If we have restored metadata but haven't applied pipeline effects,
            # we might need to do some final state synchronization here
            LOGGER.debug("Training start with pipeline context metadata: %s", self._context.metadata)
