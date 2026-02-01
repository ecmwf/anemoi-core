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

    def _build_time_coord(self, length: int) -> np.ndarray:
        if self.start and self.frequency:
            start = np.datetime64(self.start)
            step = frequency_to_timedelta(self.frequency)
            step_s = int(step.total_seconds())
            times = start + np.arange(length, dtype="int64") * np.timedelta64(step_s, "s")
            return times
        return np.arange(length, dtype="int64")

    def _get_dataset_indices(self, pl_module: pl.LightningModule):
        data_indices = pl_module.data_indices
        if isinstance(data_indices, dict):
            if "data" in data_indices:
                return data_indices["data"]
            if len(data_indices) == 1:
                return next(iter(data_indices.values()))
            raise KeyError("Multiple datasets present but no 'data' dataset found in data_indices.")
        return data_indices

    def _get_post_processor(self, pl_module: pl.LightningModule):
        post_processors = pl_module.model.post_processors
        if isinstance(post_processors, dict) or isinstance(post_processors, torch.nn.ModuleDict):
            if "data" in post_processors:
                return post_processors["data"]
            if len(post_processors) == 1:
                return next(iter(post_processors.values()))
            raise KeyError("Multiple datasets present but no 'data' dataset found in post_processors.")
        return post_processors

    def _select_variables(self, pl_module: pl.LightningModule) -> tuple[list[str], list[int]]:
        data_indices = self._get_dataset_indices(pl_module)
        name_to_index = data_indices.model.output.name_to_index
        if self.parameters:
            names = [n for n in self.parameters if n in name_to_index]
        else:
            names = list(name_to_index.keys())
        indices = [name_to_index[n] for n in names]
        return names, indices

    def _post_process(self, pl_module: pl.LightningModule, tensor: torch.Tensor) -> torch.Tensor:
        post_processor = self._get_post_processor(pl_module)
        # Use post-processors to denormalize. Avoid in-place if supported.
        try:
            return post_processor(tensor, in_place=False)
        except TypeError:
            return post_processor(tensor)

    def _ensure_3d(self, arr: np.ndarray, name: str) -> np.ndarray:
        # Expected dims: (time, node, variable). Squeeze singleton dims if needed.
        while arr.ndim > 3 and arr.shape[0] == 1:
            arr = arr[0]
        if arr.ndim == 4:
            # Fallback: drop leading dimension (e.g., ensemble/member)
            arr = arr[0]
        if arr.ndim != 3:
            raise ValueError(f"{name} has unexpected shape {arr.shape}, expected 3D (time,node,variable).")
        return arr

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
            data_indices = self._get_dataset_indices(pl_module)
            data_batch = batch
            if isinstance(batch, dict):
                if "data" in batch:
                    data_batch = batch["data"]
                elif len(batch) == 1:
                    data_batch = next(iter(batch.values()))
                else:
                    raise KeyError("Multiple datasets present in batch but no 'data' dataset found.")
            if data_batch.ndim < 3:
                LOGGER.warning("Unexpected batch shape for export: %s", tuple(data_batch.shape))
                return

            time_len = data_batch.shape[1] if data_batch.ndim >= 2 else 1
            needed_len = pl_module.multi_step + rollout
            if time_len < needed_len:
                LOGGER.warning(
                    "Batch time length too short for export (time_len=%s, needed=%s).",
                    time_len,
                    needed_len,
                )
                return

            input_tensor = (
                data_batch[
                    :,
                    0:needed_len,
                    ...,
                    data_indices.data.output.full,
                ]
                .detach()
                .cpu()
            )
            # Post-process expects (batch,node,var); flatten time then restore.
            bsz, tlen = input_tensor.shape[0], input_tensor.shape[1]
            flat = input_tensor.reshape(bsz * tlen, *input_tensor.shape[2:])
            flat = self._post_process(pl_module, flat)
            data = flat.reshape(bsz, tlen, *flat.shape[1:])[self.sample_idx]

            pred_steps = []
            for x in outputs[1]:
                step_pred = x["data"] if isinstance(x, dict) else x
                step_pred = step_pred.detach().cpu()
                step_pred = self._post_process(pl_module, step_pred)
                pred_steps.append(step_pred[self.sample_idx : self.sample_idx + 1])
            preds = torch.cat(pred_steps, dim=0)

            if batch_idx == 0:
                LOGGER.info(
                    "ExportPredictions shapes: data_batch=%s input=%s data=%s preds=%s multi_step=%s rollout=%s",
                    tuple(data_batch.shape),
                    tuple(input_tensor.shape),
                    tuple(data.shape),
                    tuple(preds.shape),
                    pl_module.multi_step,
                    rollout,
                )

        var_names, var_idx = self._select_variables(pl_module)
        data = data[:, :, var_idx].numpy()
        preds = preds[:, :, var_idx].numpy()
        data = self._ensure_3d(data, "data")
        preds = self._ensure_3d(preds, "preds")

        data_len = data.shape[0]
        pred_len = preds.shape[0]
        target_len = min(rollout, data_len - pl_module.multi_step, pred_len)
        if target_len <= 0:
            LOGGER.warning(
                "No target/prediction steps available to export (data_len=%s, pred_len=%s, rollout=%s).",
                data_len,
                pred_len,
                rollout,
            )
            return

        time_coord = self._build_time_coord(data_len)
        ds = xr.Dataset(
            data_vars={
                "input": (("time", "node", "variable"), data[: pl_module.multi_step]),
                "target": (
                    ("time", "node", "variable"),
                    data[pl_module.multi_step : pl_module.multi_step + target_len],
                ),
                "prediction": (("time", "node", "variable"), preds[:target_len]),
            },
            coords={
                "time": time_coord,
                "variable": var_names,
                "node": np.arange(data.shape[1], dtype="int64"),
            },
            attrs={
                "sample_idx": self.sample_idx,
                "batch_idx": batch_idx,
                "rollout": target_len,
            },
        )

        self.output_dir.mkdir(parents=True, exist_ok=True)
        base = self.output_dir / f"pred_target_epoch{trainer.current_epoch:03d}_batch{batch_idx:04d}"
        if self.format == "zarr":
            ds.to_zarr(f"{base}.zarr", mode="w")
        else:
            ds.to_netcdf(f"{base}.nc")
        LOGGER.info("Exported predictions/targets to %s.*", base)
