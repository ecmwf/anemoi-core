# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import logging
from typing import Any

import pytorch_lightning as pl
import torch
from omegaconf import DictConfig
from pytorch_lightning.callbacks import Callback

LOGGER = logging.getLogger(__name__)

EMA_CHECKPOINT_KEY = "ema"


class ExponentialMovingAverage(Callback):
    """Maintain an exponential moving average of model weights during training."""

    def __init__(
        self,
        config: DictConfig,
        decay: float | None = None,
        update_after_step: int | None = None,
    ) -> None:
        super().__init__()
        self.config = config

        ema_config = getattr(getattr(config, "training", None), "ema", None)
        self.decay = float(decay if decay is not None else getattr(ema_config, "decay", 0.999))
        self.update_after_step = int(
            update_after_step
            if update_after_step is not None
            else getattr(ema_config, "update_after_step", 0)
        )
        self.ema_state_dict: dict[str, torch.Tensor] | None = None

    @staticmethod
    def _unwrap_model(trainer: pl.Trainer) -> torch.nn.Module:
        assert hasattr(
            trainer, "model"
        ), "Trainer has no attribute 'model'! Is the Pytorch Lightning version correct?"
        return (
            trainer.model.module.model
            if hasattr(trainer.model, "module")
            else trainer.model.model
        )

    @staticmethod
    def _clone_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {name: value.detach().clone() for name, value in state_dict.items()}

    def _update_ema(self, state_dict: dict[str, torch.Tensor]) -> None:
        assert self.ema_state_dict is not None, "EMA state must be initialised before updating."

        for name, value in state_dict.items():
            value = value.detach()
            if torch.is_floating_point(value) or torch.is_complex(value):
                self.ema_state_dict[name].mul_(self.decay).add_(value, alpha=1.0 - self.decay)
            else:
                self.ema_state_dict[name] = value.clone()

    def on_before_zero_grad(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,  # noqa: ARG002
        optimizer: torch.optim.Optimizer,  # noqa: ARG002
    ) -> None:
        if trainer.global_step < self.update_after_step:
            return

        model = self._unwrap_model(trainer)
        current_state_dict = model.state_dict()

        if self.ema_state_dict is None:
            self.ema_state_dict = self._clone_state_dict(current_state_dict)
            LOGGER.info(
                "Initialised EMA state at global step %s with decay %s.",
                trainer.global_step,
                self.decay,
            )
            return

        self._update_ema(current_state_dict)

    def on_save_checkpoint(
        self,
        trainer: pl.Trainer,  # noqa: ARG002
        pl_module: pl.LightningModule,  # noqa: ARG002
        checkpoint: dict[str, Any],
    ) -> None:
        if self.ema_state_dict is None:
            return

        checkpoint[EMA_CHECKPOINT_KEY] = {
            "decay": self.decay,
            "update_after_step": self.update_after_step,
            "state_dict": {name: value.detach().cpu().clone() for name, value in self.ema_state_dict.items()},
        }

    def on_load_checkpoint(
        self,
        trainer: pl.Trainer,  # noqa: ARG002
        pl_module: pl.LightningModule,
        checkpoint: dict[str, Any],
    ) -> None:
        if getattr(pl_module.config.training, "load_weights_only", False):
            return

        ema_checkpoint = checkpoint.get(EMA_CHECKPOINT_KEY)
        if ema_checkpoint is None:
            return

        self.ema_state_dict = ema_checkpoint["state_dict"]
        self.decay = float(ema_checkpoint.get("decay", self.decay))
        self.update_after_step = int(
            ema_checkpoint.get("update_after_step", self.update_after_step)
        )
