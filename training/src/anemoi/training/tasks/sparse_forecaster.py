# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import TYPE_CHECKING

import torch

from anemoi.training.tasks.forecaster import Forecaster

if TYPE_CHECKING:
    from anemoi.models.data_indices.collection import IndexCollection

LOGGER = logging.getLogger(__name__)


class SparseForecaster(Forecaster):
    """Forecaster variant that reuses the latest available sparse dataset timestep during rollout."""

    name: str = "sparse_forecaster"

    def __init__(
        self,
        multistep_input: int,
        multistep_output: int,
        timestep: str,
        rollout: dict | None = None,
        validation_rollout: int = 1,
        rollout_forcing_policy: str = "last_available",
        **kwargs,
    ) -> None:
        super().__init__(
            multistep_input=multistep_input,
            multistep_output=multistep_output,
            timestep=timestep,
            rollout=rollout,
            validation_rollout=validation_rollout,
            **kwargs,
        )
        if rollout_forcing_policy not in {"last_available", "exact"}:
            msg = (
                f"Unsupported sparse rollout forcing policy '{rollout_forcing_policy}'. "
                "Expected 'last_available' or 'exact'."
            )
            raise ValueError(msg)

        self.rollout_forcing_policy = rollout_forcing_policy
        self.dataset_relative_time_indices: dict[str, list[int]] = {}
        self.dataset_time_maps: dict[str, dict[int, int]] = {}
        self._rollout_sampler_warning_keys: set[tuple[str, int, int]] = set()

    def fill_metadata(self, md_dict: dict) -> None:
        """Persist sparse timing metadata for runtime rollout."""
        super().fill_metadata(md_dict)
        metadata_inference = md_dict.get("metadata_inference", {})
        dataset_names = metadata_inference.get("dataset_names", []) if isinstance(metadata_inference, Mapping) else []

        if len(dataset_names) == 0:
            return

        relative_by_dataset = self._resolve_relative_time_metadata(metadata_inference, dataset_names)
        self.dataset_relative_time_indices = {
            dataset_name: relative_by_dataset.get(
                dataset_name,
                list(range(max(self.get_batch_output_indices(rollout_step=self.num_steps - 1)) + 1)),
            )
            for dataset_name in dataset_names
        }
        self.dataset_time_maps = {
            dataset_name: {int(relative_time): batch_idx for batch_idx, relative_time in enumerate(relative_times)}
            for dataset_name, relative_times in self.dataset_relative_time_indices.items()
        }

    def _resolve_relative_time_metadata(
        self,
        metadata_inference: Mapping,
        dataset_names: list[str],
    ) -> dict[str, list[int]]:
        """Choose the richest per-dataset time window exposed by the datamodule metadata."""
        relative_by_dataset: dict[str, list[int]] = {}
        keys = (
            "relative_date_indices_validation_by_dataset",
            "relative_date_indices_training_by_dataset",
        )

        for dataset_name in dataset_names:
            dataset_meta = metadata_inference.get(dataset_name, {})
            timesteps_meta = dataset_meta.get("timesteps", {}) if isinstance(dataset_meta, Mapping) else {}

            chosen: list[int] | None = None
            for key in keys:
                raw_relative = timesteps_meta.get(key, None)
                if not isinstance(raw_relative, Mapping):
                    continue
                raw_values = raw_relative.get(dataset_name, None)
                if raw_values is None:
                    continue
                candidate = [int(value) for value in raw_values]
                if chosen is None or max(candidate, default=-1) > max(chosen, default=-1):
                    chosen = candidate

            if chosen is not None:
                relative_by_dataset[dataset_name] = chosen

        return relative_by_dataset

    def _sample_batch_position(self, *, dataset_name: str, relative_time: int) -> int:
        time_map = self.dataset_time_maps.get(dataset_name, {})
        exact_idx = time_map.get(int(relative_time), None)
        if exact_idx is not None:
            return int(exact_idx)

        available_times = sorted(int(value) for value in time_map)
        if not available_times:
            msg = f"Dataset '{dataset_name}' has no available relative times for sparse rollout."
            raise ValueError(msg)

        if self.rollout_forcing_policy == "last_available":
            candidate_times = [value for value in available_times if value <= int(relative_time)]
            if not candidate_times:
                msg = (
                    f"Dataset '{dataset_name}' has no forcing/boundary time at or before relative time "
                    f"{relative_time}. Available times: {available_times}"
                )
                raise ValueError(msg)
            sampled_time = candidate_times[-1]
        else:
            msg = (
                f"Dataset '{dataset_name}' is missing exact relative time {relative_time}. "
                f"Available times: {available_times}"
            )
            raise ValueError(msg)

        warning_key = (dataset_name, int(relative_time), int(sampled_time))
        if warning_key not in self._rollout_sampler_warning_keys:
            LOGGER.info(
                "Sparse rollout dataset=%s requested_time=%s sampled_time=%s policy=%s",
                dataset_name,
                relative_time,
                sampled_time,
                self.rollout_forcing_policy,
            )
            self._rollout_sampler_warning_keys.add(warning_key)

        return int(time_map[sampled_time])

    def get_inputs(
        self,
        batch: dict[str, torch.Tensor],
        data_indices: dict[str, IndexCollection],
        **_kwargs,
    ) -> dict[str, torch.Tensor]:
        if len(self.dataset_time_maps) == 0:
            return super().get_inputs(batch, data_indices)

        requested_relative_times = self.get_batch_input_indices()
        x = {}
        for dataset_name, dataset_batch in batch.items():
            input_positions = [
                self._sample_batch_position(dataset_name=dataset_name, relative_time=relative_time)
                for relative_time in requested_relative_times
            ]
            input_index = torch.tensor(input_positions, device=dataset_batch.device, dtype=torch.long)
            x_time = dataset_batch.index_select(1, input_index)
            x[dataset_name] = x_time[..., data_indices[dataset_name].data.input.full]
        return x

    def get_targets(self, batch: dict[str, torch.Tensor], **kwargs) -> dict[str, torch.Tensor]:
        if len(self.dataset_time_maps) == 0:
            return super().get_targets(batch, **kwargs)

        requested_relative_times = self.get_batch_output_indices(rollout_step=kwargs.get("rollout_step", 0))
        y = {}
        for dataset_name, dataset_batch in batch.items():
            target_positions = [
                self._sample_batch_position(dataset_name=dataset_name, relative_time=relative_time)
                for relative_time in requested_relative_times
            ]
            target_index = torch.tensor(target_positions, device=dataset_batch.device, dtype=torch.long)
            y[dataset_name] = dataset_batch.index_select(1, target_index)
        return y

    def _build_rollout_input_step(
        self,
        *,
        dataset_name: str,
        dataset_batch: torch.Tensor,
        y_pred_full: dict[str, torch.Tensor],
        relative_time: int,
        rollout_step: int,
        data_indices: dict[str, IndexCollection],
        output_mask: dict[str, object] | None,
        grid_shard_slice: dict[str, slice | None] | None,
    ) -> torch.Tensor:
        batch_position = self._sample_batch_position(dataset_name=dataset_name, relative_time=relative_time)
        x_step = dataset_batch[
            :,
            batch_position,
            ...,
            data_indices[dataset_name].data.input.full,
        ].clone()

        pred_start = self.num_input_timesteps + rollout_step * self.num_output_timesteps
        pred_end = pred_start + self.num_output_timesteps - 1
        if pred_start <= int(relative_time) <= pred_end and dataset_name in y_pred_full:
            pred_position = int(relative_time - pred_start)
            x_step[..., data_indices[dataset_name].model.input.prognostic] = y_pred_full[dataset_name][
                :,
                pred_position,
                ...,
                data_indices[dataset_name].model.output.prognostic,
            ]

        dataset_output_mask = None if output_mask is None else output_mask[dataset_name]
        if dataset_output_mask is not None:
            true_state = dataset_batch[:, batch_position]
            if true_state.shape[1] == 1 and x_step.shape[1] != 1:
                true_state = true_state.expand(-1, x_step.shape[1], -1, -1)
            x_step = dataset_output_mask.rollout_boundary(
                x_step,
                true_state,
                data_indices[dataset_name],
                grid_shard_slice=None if grid_shard_slice is None else grid_shard_slice[dataset_name],
            )

        x_step[..., data_indices[dataset_name].model.input.forcing] = dataset_batch[
            :,
            batch_position,
            ...,
            data_indices[dataset_name].data.input.forcing,
        ]
        return x_step

    def advance_input(
        self,
        x: dict[str, torch.Tensor],
        y_pred: dict[str, torch.Tensor],
        batch: dict[str, torch.Tensor],
        rollout_step: int = 0,
        data_indices: dict[str, IndexCollection] | None = None,
        output_mask: dict[str, object] | None = None,
        grid_shard_slice: dict[str, slice | None] | None = None,
    ) -> dict[str, torch.Tensor]:
        if len(self.dataset_time_maps) == 0:
            return super().advance_input(
                x,
                y_pred,
                batch,
                rollout_step=rollout_step,
                data_indices=data_indices,
                output_mask=output_mask,
                grid_shard_slice=grid_shard_slice,
            )

        del x
        next_input_relative_times = [
            int(relative_time + (rollout_step + 1) * self.num_output_timesteps)
            for relative_time in self.get_batch_input_indices()
        ]
        next_x: dict[str, torch.Tensor] = {}
        for dataset_name, dataset_batch in batch.items():
            next_steps = [
                self._build_rollout_input_step(
                    dataset_name=dataset_name,
                    dataset_batch=dataset_batch,
                    y_pred_full=y_pred,
                    relative_time=relative_time,
                    rollout_step=rollout_step,
                    data_indices=data_indices,
                    output_mask=output_mask,
                    grid_shard_slice=grid_shard_slice,
                )
                for relative_time in next_input_relative_times
            ]
            next_x[dataset_name] = torch.stack(next_steps, dim=1)
        return next_x
