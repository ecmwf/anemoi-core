# (C) Copyright 2024- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Callback logging the per-level diagnostics of RefractivityOperatorLoss."""

import logging

import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback

from anemoi.training.losses.refractivity import RefractivityOperatorLoss
from anemoi.training.losses.target_identity import TargetIdentityLoss

LOGGER = logging.getLogger(__name__)


class RefractivityLevelLogger(Callback):
    """Log the diagnostics that ``RefractivityOperatorLoss`` and ``TargetIdentityLoss`` keep after each forward.

    Training values are single-batch and noisy, so by default only aggregates are logged
    during training (``train_refrac/<dataset>/mean``, ``train_refrac_count/<dataset>/total``,
    ``train_refrac_bias/<dataset>/mean_abs``) together with the column-health scalars
    (unbracketed, ambiguous and disordered-layer fractions, minimum-thickness hinge).
    Validation values are epoch means, so per-level loss and bias are logged there by default
    (``val_refrac/<dataset>/<level>``, ``val_refrac_bias/<dataset>/<level>``).

    Parameters
    ----------
    every_n_batches : int
        Training-batch logging frequency.
    per_level_training : bool
        Also log per-level loss, count and bias during training.
    per_level_validation : bool
        Log per-level loss and bias during validation (otherwise aggregates only).
    health_scalars : bool
        Log the column-health scalars.
    """

    def __init__(
        self,
        every_n_batches: int = 100,
        per_level_training: bool = False,
        per_level_validation: bool = True,
        health_scalars: bool = True,
    ) -> None:
        super().__init__()
        self.every_n_batches = max(1, int(every_n_batches))
        self.per_level_training = per_level_training
        self.per_level_validation = per_level_validation
        self.health_scalars = health_scalars

    @staticmethod
    def _refractivity_losses(pl_module: pl.LightningModule) -> list[tuple[str, RefractivityOperatorLoss]]:
        found = []
        for dataset_name, loss in getattr(pl_module, "loss", {}).items():
            leaves = loss.iter_leaf_losses() if hasattr(loss, "iter_leaf_losses") else [loss]
            found.extend((dataset_name, leaf) for leaf in leaves if isinstance(leaf, RefractivityOperatorLoss))
        return found

    def _emit(self, pl_module: pl.LightningModule, name: str, value: float, *, on_step: bool) -> None:
        pl_module.log(
            name,
            float(value),
            on_step=on_step,
            on_epoch=not on_step,
            logger=getattr(pl_module, "logger_enabled", True),
            sync_dist=not on_step,
            rank_zero_only=on_step,
        )

    def _log(self, pl_module: pl.LightningModule, prefix: str, *, on_step: bool, per_level: bool) -> None:
        for dataset_name, loss in self._refractivity_losses(pl_module):
            if loss.last_level_losses is None:
                continue
            losses = loss.last_level_losses
            counts = loss.last_level_counts
            biases = loss.last_level_bias
            active = counts > 0
            n_active = int(active.sum())
            self._emit(
                pl_module,
                f"{prefix}/{dataset_name}/mean",
                losses[active].mean() if n_active else 0.0,
                on_step=on_step,
            )
            self._emit(pl_module, f"{prefix}_count/{dataset_name}/total", counts.sum(), on_step=on_step)
            if biases is not None:
                self._emit(
                    pl_module,
                    f"{prefix}_bias/{dataset_name}/mean_abs",
                    biases[active].abs().mean() if n_active else 0.0,
                    on_step=on_step,
                )
            if per_level:
                bias_list = biases.tolist() if biases is not None else [None] * len(loss.observation_variables)
                for name, value, count, bias in zip(
                    loss.observation_variables,
                    losses.tolist(),
                    counts.tolist(),
                    bias_list,
                    strict=False,
                ):
                    self._emit(pl_module, f"{prefix}/{dataset_name}/{name}", value, on_step=on_step)
                    self._emit(pl_module, f"{prefix}_count/{dataset_name}/{name}", count, on_step=on_step)
                    if bias is not None:
                        self._emit(pl_module, f"{prefix}_bias/{dataset_name}/{name}", bias, on_step=on_step)
            if self.health_scalars:
                scalars = {
                    "unbracketed_fraction": loss.last_unbracketed_fraction,
                    "ambiguous_fraction": loss.last_ambiguous_fraction,
                    "disordered_layer_fraction": loss.last_disordered_layer_fraction,
                    "monotonicity_penalty": loss.last_monotonicity_penalty,
                }
                for key, value in scalars.items():
                    if value is not None:
                        self._emit(pl_module, f"{prefix}_{key}/{dataset_name}", value, on_step=on_step)

    @staticmethod
    def _identity_losses(pl_module: pl.LightningModule) -> list[tuple[str, TargetIdentityLoss]]:
        found = []
        for dataset_name, loss in getattr(pl_module, "loss", {}).items():
            leaves = loss.iter_leaf_losses() if hasattr(loss, "iter_leaf_losses") else [loss]
            found.extend((dataset_name, leaf) for leaf in leaves if isinstance(leaf, TargetIdentityLoss))
        return found

    def _log_identity(self, pl_module: pl.LightningModule, prefix: str, *, on_step: bool) -> None:
        """Per-pair loss, bias and count of every TargetIdentityLoss leaf (few pairs, always per pair)."""
        for dataset_name, loss in self._identity_losses(pl_module):
            if loss.last_pair_losses is None:
                continue
            for name, value, bias, count in zip(
                loss.observation_variables,
                loss.last_pair_losses.tolist(),
                loss.last_pair_bias.tolist(),
                loss.last_pair_counts.tolist(),
                strict=False,
            ):
                self._emit(pl_module, f"{prefix}/{dataset_name}/{name}", value, on_step=on_step)
                self._emit(pl_module, f"{prefix}_bias/{dataset_name}/{name}", bias, on_step=on_step)
                if not on_step:
                    self._emit(pl_module, f"{prefix}_count/{dataset_name}/{name}", count, on_step=on_step)

    def on_train_batch_end(
        self,
        trainer: pl.Trainer,  # noqa: ARG002
        pl_module: pl.LightningModule,
        outputs: object,  # noqa: ARG002
        batch: object,  # noqa: ARG002
        batch_idx: int,
    ) -> None:
        if batch_idx % self.every_n_batches == 0:
            self._log(pl_module, "train_refrac", on_step=True, per_level=self.per_level_training)
            self._log_identity(pl_module, "train_identity", on_step=True)

    def on_validation_batch_end(
        self,
        trainer: pl.Trainer,  # noqa: ARG002
        pl_module: pl.LightningModule,
        outputs: object,  # noqa: ARG002
        batch: object,  # noqa: ARG002
        batch_idx: int,  # noqa: ARG002
        dataloader_idx: int = 0,  # noqa: ARG002
    ) -> None:
        self._log(pl_module, "val_refrac", on_step=False, per_level=self.per_level_validation)
        self._log_identity(pl_module, "val_identity", on_step=False)
