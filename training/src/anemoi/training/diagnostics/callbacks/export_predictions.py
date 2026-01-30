# (C) Copyright 2024- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import logging
from pathlib import Path
from typing import Any

import numpy as np
import pytorch_lightning as pl
import torch
import xarray as xr

from anemoi.training.train.tasks import GraphInterpolator
from anemoi.utils.dates import frequency_to_timedelta

LOGGER = logging.getLogger(__name__)


class ExportPredictions(pl.Callback):
    """Export denormalized predictions and targets for verification."""

    def __init__(self, config: Any) -> None:
        super().__init__()
        self.config = config

        cfg = config.diagnostics.export_predictions
        self.enabled = cfg.enabled
        self.format = cfg.format
        self.every_n_batches = cfg.every_n_batches
        self.sample_idx = cfg.sample_idx
        self.parameters = cfg.parameters
        self.output_dir = Path(cfg.output_dir) if cfg.output_dir else Path(config.system.output.plots) / "exports"
        self.start = cfg.start
        self.frequency = cfg.frequency

    def _get_output_times(self, pl_module: pl.LightningModule) -> tuple[int, str]:
        if isinstance(pl_module, GraphInterpolator):
            output_times = (len(self.config.training.explicit_times.target), "time_interp")
        else:
            output_times = (getattr(pl_module, "rollout", 0), "forecast")
        return output_times

    def _build_time_coord(self, rollout: int) -> np.ndarray:
        if self.start and self.frequency:
            start = np.datetime64(self.start)
            step = frequency_to_timedelta(self.frequency)
            step_s = int(step.total_seconds())
            times = start + np.arange(rollout + 1, dtype="int64") * np.timedelta64(step_s, "s")
            return times
        return np.arange(rollout + 1, dtype="int64")

    def _select_variables(self, pl_module: pl.LightningModule) -> tuple[list[str], list[int]]:
        name_to_index = pl_module.data_indices.model.output.name_to_index
        if self.parameters:
            names = [n for n in self.parameters if n in name_to_index]
        else:
            names = list(name_to_index.keys())
        indices = [name_to_index[n] for n in names]
        return names, indices

    def _post_process(self, pl_module: pl.LightningModule, tensor: torch.Tensor) -> torch.Tensor:
        # Use post-processors to denormalize. Avoid in-place if supported.
        try:
            return pl_module.model.post_processors(tensor, in_place=False)
        except TypeError:
            return pl_module.model.post_processors(tensor)

    def on_validation_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: list,
        batch: torch.Tensor,
        batch_idx: int,
        **kwargs: Any,
    ) -> None:
        del kwargs
        if not self.enabled:
            return
        if batch_idx % self.every_n_batches != 0:
            return

        output_times = self._get_output_times(pl_module)
        rollout = output_times[0]
        if rollout <= 0:
            LOGGER.warning("No rollout steps available to export.")
            return

        # Prepare denormalized inputs/targets
        with torch.no_grad():
            input_tensor = (
                batch[
                    :,
                    pl_module.multi_step - 1 : pl_module.multi_step + rollout + 1,
                    ...,
                    pl_module.data_indices.data.output.full,
                ]
                .detach()
                .cpu()
            )
            data = self._post_process(pl_module, input_tensor)[self.sample_idx]

            preds = torch.cat(
                tuple(
                    self._post_process(pl_module, x.detach().cpu())[self.sample_idx : self.sample_idx + 1]
                    for x in outputs[1]
                ),
                dim=0,
            ).squeeze(1)

        var_names, var_idx = self._select_variables(pl_module)
        data = data[:, :, var_idx].numpy()
        preds = preds[:, :, var_idx].numpy()

        time_coord = self._build_time_coord(rollout)
        ds = xr.Dataset(
            data_vars={
                "input": (("time", "node", "variable"), data[:1]),
                "target": (("time", "node", "variable"), data[1 : rollout + 1]),
                "prediction": (("time", "node", "variable"), preds[:rollout]),
            },
            coords={
                "time": time_coord[: rollout + 1],
                "variable": var_names,
                "node": np.arange(data.shape[1], dtype="int64"),
            },
            attrs={
                "sample_idx": self.sample_idx,
                "batch_idx": batch_idx,
                "rollout": rollout,
            },
        )

        self.output_dir.mkdir(parents=True, exist_ok=True)
        base = self.output_dir / f"pred_target_epoch{trainer.current_epoch:03d}_batch{batch_idx:04d}"
        if self.format == "zarr":
            ds.to_zarr(f"{base}.zarr", mode="w")
        else:
            ds.to_netcdf(f"{base}.nc")
        LOGGER.info("Exported predictions/targets to %s.*", base)
