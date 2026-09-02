# (C) Copyright 2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Training task for the deterministic local downscaler (fine-scale epic, Track E stage 1).

Same data plumbing as ``GraphDiffusionDownscaler`` (raw batch, residual target through
``compute_residuals``, explicit input normalisation, ``add_interp_to_state`` for the full
prediction), but no noise, no preconditioning and no per-noise-level weights: one forward of
``AnemoiLocalDownscaler`` and the lane's weighted mean-squared error on the normalised residual.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping

import torch
from torch.utils.checkpoint import checkpoint

from anemoi.training.train.tasks.diffusiondownscaler import GraphDiffusionDownscaler

LOGGER = logging.getLogger(__name__)


class DeterministicLocalDownscaler(GraphDiffusionDownscaler):
    """Deterministic residual downscaler task."""

    task_type = "downscaler"

    def forward(self, x: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:  # type: ignore[override]
        return self.model.model(
            x,
            model_comm_group=self.model_comm_group,
            grid_shard_sizes=self.grid_shard_sizes,
        )

    def _compute_loss(
        self,
        y_pred: torch.Tensor,
        y: torch.Tensor,
        dataset_name: str | None = None,
        weights: dict[str, torch.Tensor] | None = None,
        grid_shard_slice: slice | None = None,
        **_kwargs,
    ) -> torch.Tensor:
        """Plain weighted MSE (no noise-level weights)."""
        return self.loss[dataset_name](
            y_pred,
            y,
            grid_shard_slice=grid_shard_slice,
            group=self.model_comm_group,
        )

    def _step(
        self,
        batch: dict[str, torch.Tensor],
        validation_mode: bool = False,
    ) -> tuple[torch.Tensor, Mapping[str, torch.Tensor], list[dict[str, torch.Tensor]]]:
        x_in_lres = batch["in_lres"]
        x_in_hres = batch["in_hres"]
        y = batch["out_hres"]

        target_ds = self.model.model._decoder_datasets[0]
        source_ds = self._residual_pairs.get(target_ds)

        x_in_lres_upsampled = self.model.model.residual["in_lres"](
            x_in_lres,
            grid_shard_sizes=self.grid_shard_sizes.get("in_lres", None),
            model_comm_group=self.model_comm_group,
        )[:, :, None, :, :]

        if source_ds is not None:
            channel_indices = self.model.model.get_matching_channel_indices(target_ds).to(x_in_lres_upsampled.device)
            target = self.model.model.compute_residuals(
                y=y,
                x_interp=x_in_lres_upsampled[..., channel_indices],
                pre_processors_state=self.model.pre_processors[target_ds],
                pre_processors_tendencies=self._residual_pre_processors.get(target_ds),
                target_dataset=target_ds,
            )
        else:
            target = self.model.pre_processors[target_ds](y, in_place=False)

        x_in_lres_upsampled = self.model.pre_processors["in_lres"](x_in_lres_upsampled, in_place=False)
        x_in_hres = self.model.pre_processors["in_hres"](x_in_hres, in_place=False)
        target_dict = {target_ds: target}
        x_dict = {"in_lres": x_in_lres_upsampled, "in_hres": x_in_hres}

        y_pred = self(x_dict)

        loss, metrics_next, y_pred_out = checkpoint(
            self.compute_loss_metrics,
            y_pred,
            target_dict,
            validation_mode=validation_mode,
            use_reentrant=False,
        )

        if source_ds is not None:
            y_pred_full = self.model.model.add_interp_to_state(
                state_inp=x_in_lres_upsampled,
                model_output=y_pred[target_ds],
                post_processors_state=self.model.post_processors,
                post_processors_tendencies=(
                    dict(self.model.post_processors_tendencies)
                    if hasattr(self.model, "post_processors_tendencies")
                    and self.model.post_processors_tendencies is not None
                    else None
                ),
                target_dataset=target_ds,
                source_dataset=source_ds,
            )
        else:
            y_pred_full = y_pred_out[target_ds]

        return loss, metrics_next, [y_pred_full]
