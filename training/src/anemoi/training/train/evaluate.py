# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import logging
from functools import cached_property

import hydra
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig
from omegaconf import OmegaConf

from anemoi.training.diagnostics.callbacks import CallbacksContext
from anemoi.training.diagnostics.callbacks import get_callbacks
from anemoi.training.train.train import AnemoiTrainer

LOGGER = logging.getLogger(__name__)


class AnemoiEvaluator(AnemoiTrainer):
    """Utility class for evaluating a trained model.

    Inherits all setup from :class:`AnemoiTrainer` (datamodule, graph, model,
    callbacks, logger, strategy).  The key differences from training are:

    - :meth:`pl.Trainer.validate` is called instead of :meth:`pl.Trainer.fit`.
    - A checkpoint **must** be specified via ``training.run_id`` or
      ``system.input.warm_start``; omitting both raises a :exc:`RuntimeError`
      immediately rather than silently evaluating a randomly-initialised model.
    - Checkpointing and weight-averaging callbacks are disabled regardless of
      the diagnostics config — evaluation is a read-only operation on a trained
      model and should never write new checkpoint files.
    """

    @cached_property
    def callbacks(self) -> list[pl.callbacks.Callback]:
        """Callbacks for evaluation — checkpointing and weight averaging disabled."""
        diag_cfg = OmegaConf.merge(self.config.diagnostics, {"enable_checkpointing": False})
        context = CallbacksContext(
            diagnostics=diag_cfg,
            checkpoints_output=self.config.system.output.checkpoints,
            plots_output=self.config.system.output.plots,
            wandb_enabled=getattr(getattr(self.config.diagnostics.log, "wandb", None), "enabled", False),
            mlflow_enabled=getattr(getattr(self.config.diagnostics.log, "mlflow", None), "enabled", False),
            weight_averaging_config=None,
        )
        return get_callbacks(context)

    def evaluate(self) -> None:
        """Evaluation entry point — runs one full validation pass."""
        if not self.start_from_checkpoint:
            msg = (
                "No checkpoint specified for evaluation. "
                "Set 'training.run_id', 'training.fork_run_id', or "
                "'system.input.warm_start' to point to a trained checkpoint."
            )
            raise RuntimeError(msg)

        LOGGER.debug("Setting up evaluator trainer..")

        trainer = pl.Trainer(
            accelerator=self.accelerator,
            callbacks=self.callbacks,
            detect_anomaly=self.config.diagnostics.debug.anomaly_detection,
            strategy=self.strategy,
            devices=self.config.system.hardware.num_gpus_per_node,
            num_nodes=self.config.system.hardware.num_nodes,
            precision=self.config.training.precision,
            logger=self.logger,
            log_every_n_steps=self.config.diagnostics.log.interval,
            limit_val_batches=self.config.dataloader.limit_batches.validation,
            use_distributed_sampler=False,
            enable_progress_bar=self.config.diagnostics.enable_progress_bar,
        )

        LOGGER.debug("Starting evaluation..")

        # When weights-only loading is used the model is already fully initialised
        # with the checkpoint weights; pass ckpt_path=None to avoid a second load.
        # Otherwise let PL restore the full training state from the checkpoint.
        ckpt_path = None if self.load_weights_only else self.last_checkpoint

        trainer.validate(
            model=self.model,
            datamodule=self.datamodule,
            ckpt_path=ckpt_path,
        )

        if self.config.diagnostics.print_memory_summary:
            LOGGER.info("memory summary: %s", torch.cuda.memory_summary(device=0))

        LOGGER.debug("---- DONE. ----")


@hydra.main(version_base=None, config_path="../config", config_name="config")
def evaluate(config: DictConfig) -> None:
    AnemoiEvaluator(config).evaluate()


if __name__ == "__main__":
    evaluate()
