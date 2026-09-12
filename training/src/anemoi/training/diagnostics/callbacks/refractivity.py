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

LOGGER = logging.getLogger(__name__)


class RefractivityLevelLogger(Callback):
    """Log per-level refractivity losses, valid-observation counts and the unbracketed fraction.

    ``RefractivityOperatorLoss`` keeps its last per-level values on the module after
    every forward; this callback reads them from every such leaf in the training loss
    tree and logs them as ``train_refrac/<dataset>/<level>`` (every ``every_n_batches``
    training batches) and ``val_refrac/<dataset>/<level>`` (epoch-averaged), together with
    the observation counts and the column-health scalars (unbracketed, ambiguous and
    disordered-layer fractions, minimum-thickness hinge).

    Parameters
    ----------
    every_n_batches : int
        Training-batch logging frequency.
    """

    def __init__(self, every_n_batches: int = 100) -> None:
        super().__init__()
        self.every_n_batches = max(1, int(every_n_batches))

    @staticmethod
    def _refractivity_losses(pl_module: pl.LightningModule) -> list[tuple[str, RefractivityOperatorLoss]]:
        found = []
        for dataset_name, loss in getattr(pl_module, "loss", {}).items():
            leaves = loss.iter_leaf_losses() if hasattr(loss, "iter_leaf_losses") else [loss]
            found.extend((dataset_name, leaf) for leaf in leaves if isinstance(leaf, RefractivityOperatorLoss))
        return found

    def _log(self, pl_module: pl.LightningModule, prefix: str, *, on_step: bool) -> None:
        for dataset_name, loss in self._refractivity_losses(pl_module):
            if loss.last_level_losses is None:
                continue
            for name, value, count in zip(
                loss.observation_variables,
                loss.last_level_losses.tolist(),
                loss.last_level_counts.tolist(),
                strict=False,
            ):
                pl_module.log(
                    f"{prefix}/{dataset_name}/{name}",
                    value,
                    on_step=on_step,
                    on_epoch=not on_step,
                    logger=getattr(pl_module, "logger_enabled", True),
                    sync_dist=not on_step,
                    rank_zero_only=on_step,
                )
                pl_module.log(
                    f"{prefix}_count/{dataset_name}/{name}",
                    float(count),
                    on_step=on_step,
                    on_epoch=not on_step,
                    logger=getattr(pl_module, "logger_enabled", True),
                    sync_dist=not on_step,
                    rank_zero_only=on_step,
                )
            scalars = {
                "unbracketed_fraction": loss.last_unbracketed_fraction,
                "ambiguous_fraction": loss.last_ambiguous_fraction,
                "disordered_layer_fraction": loss.last_disordered_layer_fraction,
                "monotonicity_penalty": loss.last_monotonicity_penalty,
            }
            for key, value in scalars.items():
                if value is None:
                    continue
                pl_module.log(
                    f"{prefix}_{key}/{dataset_name}",
                    float(value),
                    on_step=on_step,
                    on_epoch=not on_step,
                    logger=getattr(pl_module, "logger_enabled", True),
                    sync_dist=not on_step,
                    rank_zero_only=on_step,
                )

    def on_train_batch_end(
        self,
        trainer: pl.Trainer,  # noqa: ARG002
        pl_module: pl.LightningModule,
        outputs: object,  # noqa: ARG002
        batch: object,  # noqa: ARG002
        batch_idx: int,
    ) -> None:
        if batch_idx % self.every_n_batches == 0:
            self._log(pl_module, "train_refrac", on_step=True)

    def on_validation_batch_end(
        self,
        trainer: pl.Trainer,  # noqa: ARG002
        pl_module: pl.LightningModule,
        outputs: object,  # noqa: ARG002
        batch: object,  # noqa: ARG002
        batch_idx: int,  # noqa: ARG002
        dataloader_idx: int = 0,  # noqa: ARG002
    ) -> None:
        self._log(pl_module, "val_refrac", on_step=False)
