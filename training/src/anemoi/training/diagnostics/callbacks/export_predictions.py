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

    def _build_time_coord(
        self,
        length: int,
        batch_idx: int = 0,
        batch_size: int = 1,
        sample_idx: int = 0,
    ) -> np.ndarray:
        if self.start and self.frequency:
            start = np.datetime64(self.start)
            step = frequency_to_timedelta(self.frequency)
            step_s = int(step.total_seconds())
            sample_offset = batch_idx * batch_size + sample_idx
            times = start + (sample_offset + np.arange(length, dtype="int64")) * np.timedelta64(step_s, "s")
            return times
        return np.arange(length, dtype="int64")

    def _extract_time_coord_from_batch(
        self,
        batch: torch.Tensor | dict[str, torch.Tensor],
        expected_len: int,
    ) -> np.ndarray | None:
        if not isinstance(batch, dict):
            return None

        time_key_candidates = ("sample_time_ns", "__sample_time_ns__", "time_ns")
        time_tensor = None
        for key in time_key_candidates:
            if key in batch:
                time_tensor = batch[key]
                break
        if time_tensor is None:
            return None

        if not torch.is_tensor(time_tensor):
            return None

        arr = time_tensor.detach().cpu().numpy()
        if arr.ndim == 1:
            sample_times = arr
        elif arr.ndim >= 2:
            if self.sample_idx >= arr.shape[0]:
                return None
            sample_times = arr[self.sample_idx]
        else:
            return None

        if sample_times.shape[0] < expected_len:
            return None

        return sample_times[:expected_len].astype("int64").astype("datetime64[ns]")

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

    def _get_latlons(self, pl_module: pl.LightningModule, n_nodes: int) -> tuple[np.ndarray, np.ndarray] | None:
        """Get per-node lat/lon from graph coordinates used by the model."""
        latlons = getattr(pl_module, "latlons_data", None)
        coords = None
        if latlons is not None:
            if isinstance(latlons, dict):
                if "data" in latlons:
                    coords = latlons["data"]
                elif len(latlons) == 1:
                    coords = next(iter(latlons.values()))
            else:
                coords = latlons

        # Fallback for current Anemoi training path: graph stores node coords on .x
        if coords is None:
            try:
                graph_data = pl_module.model.model._graph_data
                graph_name_data = pl_module.model.model._graph_name_data
                if isinstance(graph_data, dict):
                    if "data" in graph_data:
                        coords = graph_data["data"][graph_name_data].x
                    elif len(graph_data) == 1:
                        coords = next(iter(graph_data.values()))[graph_name_data].x
            except Exception:
                coords = None
        if coords is None:
            return None

        arr = coords.detach().cpu().numpy() if hasattr(coords, "detach") else np.asarray(coords)
        if arr.ndim != 2 or arr.shape[1] < 2:
            return None
        lat = arr[:, 0]
        lon = arr[:, 1]
        # Graph coords are in radians in training task, convert to degrees if needed.
        if np.nanmax(np.abs(lat)) <= np.pi + 1e-6 and np.nanmax(np.abs(lon)) <= 2 * np.pi + 1e-6:
            lat = np.rad2deg(lat)
            lon = np.rad2deg(lon)
        n = min(n_nodes, lat.shape[0], lon.shape[0])
        return lat[:n], lon[:n]

    def _post_process(self, pl_module: pl.LightningModule, tensor: torch.Tensor) -> torch.Tensor:
        post_processor = self._get_post_processor(pl_module)
        # Use post-processors to denormalize. Avoid in-place if supported.
        try:
            return post_processor(tensor, in_place=False)
        except TypeError:
            return post_processor(tensor)

    def _ensure_3d(self, arr: np.ndarray, name: str) -> np.ndarray:
        # Expected dims: (time, node, variable). Squeeze singleton dims if needed,
        # but never drop the time dimension unless it's truly singleton.
        if arr.ndim > 3:
            squeeze_axes = [i for i in range(1, arr.ndim) if arr.shape[i] == 1]
            if squeeze_axes:
                arr = np.squeeze(arr, axis=tuple(squeeze_axes))
        if arr.ndim == 4 and arr.shape[0] == 1:
            # If time dimension is singleton, drop it.
            arr = arr[0]
        if arr.ndim != 3:
            raise ValueError(f"{name} has unexpected shape {arr.shape}, expected 3D (time,node,variable).")
        return arr

    def _get_n_step_input(self, pl_module: pl.LightningModule) -> int:
        """Return input-step count across Anemoi versions."""
        n_step_input = getattr(pl_module, "n_step_input", None)
        if n_step_input is None:
            # Backward compatibility with older task attribute name.
            n_step_input = getattr(pl_module, "multi_step", None)
        if n_step_input is None:
            raise AttributeError("Could not find input-step attribute on Lightning module (n_step_input/multi_step).")
        return int(n_step_input)

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
        n_step_input = self._get_n_step_input(pl_module)
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
            needed_len = n_step_input + rollout
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
            )
            # Post-process expects (batch,node,var); flatten time then restore.
            bsz, tlen = input_tensor.shape[0], input_tensor.shape[1]
            flat = input_tensor.reshape(bsz * tlen, *input_tensor.shape[2:])
            flat = self._post_process(pl_module, flat)
            data = flat.reshape(bsz, tlen, *flat.shape[1:])[self.sample_idx]

            pred_steps = []
            for x in outputs[1]:
                step_pred = x["data"] if isinstance(x, dict) else x
                step_pred = step_pred.detach()
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
                    n_step_input,
                    rollout,
                )

        var_names, var_idx = self._select_variables(pl_module)
        data = self._ensure_3d(data.detach().cpu().numpy(), "data")
        preds = self._ensure_3d(preds.detach().cpu().numpy(), "preds")
        # Select variables after enforcing (time,node,variable) to avoid slicing the wrong axis.
        data = data[:, :, var_idx]
        preds = preds[:, :, var_idx]

        data_len = data.shape[0]
        pred_len = preds.shape[0]
        target_len = min(rollout, data_len - n_step_input, pred_len)
        if target_len <= 0:
            LOGGER.warning(
                "No target/prediction steps available to export (data_len=%s, pred_len=%s, rollout=%s).",
                data_len,
                pred_len,
                rollout,
            )
            return

        time_coord = self._extract_time_coord_from_batch(batch, data_len)
        if time_coord is None:
            time_coord = self._build_time_coord(
                data_len,
                batch_idx=batch_idx,
                batch_size=data_batch.shape[0],
                sample_idx=self.sample_idx,
            )
        input_time = time_coord[:n_step_input]
        target_time = time_coord[n_step_input : n_step_input + target_len]
        pred_time = target_time
        if batch_idx == 0:
            LOGGER.info(
                "ExportPredictions time window: input[%s..%s], target[%s..%s]",
                str(input_time[0]) if len(input_time) else "n/a",
                str(input_time[-1]) if len(input_time) else "n/a",
                str(target_time[0]) if len(target_time) else "n/a",
                str(target_time[-1]) if len(target_time) else "n/a",
            )
        ds = xr.Dataset(
            data_vars={
                "input": (("input_time", "node", "variable"), data[:n_step_input]),
                "target": (
                    ("target_time", "node", "variable"),
                    data[n_step_input : n_step_input + target_len],
                ),
                "prediction": (("pred_time", "node", "variable"), preds[:target_len]),
            },
            coords={
                "input_time": input_time,
                "target_time": target_time,
                "pred_time": pred_time,
                "variable": var_names,
                "node": np.arange(data.shape[1], dtype="int64"),
            },
            attrs={
                "sample_idx": self.sample_idx,
                "batch_idx": batch_idx,
                "rollout": target_len,
            },
        )

        latlons = self._get_latlons(pl_module, ds.sizes["node"])
        if latlons is not None:
            lat, lon = latlons
            ds = ds.assign_coords(
                latitude=("node", lat),
                longitude=("node", lon),
            )

        self.output_dir.mkdir(parents=True, exist_ok=True)
        base = self.output_dir / f"pred_target_epoch{trainer.current_epoch:03d}_batch{batch_idx:04d}"
        if self.format == "zarr":
            ds.to_zarr(f"{base}.zarr", mode="w")
        else:
            ds.to_netcdf(f"{base}.nc")
        LOGGER.info("Exported predictions/targets to %s.*", base)
