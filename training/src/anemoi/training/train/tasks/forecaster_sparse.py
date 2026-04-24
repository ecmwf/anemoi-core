# (C) Copyright 2024 Anemoi contributors.
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
from omegaconf import ListConfig
from torch.utils.checkpoint import checkpoint

from anemoi.models.distributed.graph import gather_tensor
from anemoi.training.train.tasks.rollout import BaseRolloutGraphModule
from anemoi.training.utils.enums import TensorDim

if TYPE_CHECKING:
    from collections.abc import Generator

    from torch.distributed.distributed_c10d import ProcessGroup

LOGGER = logging.getLogger(__name__)


class GraphForecasterSparse(BaseRolloutGraphModule):
    """Forecaster task with per-dataset sparse input/target time maps."""

    task_type = "forecaster"

    def __init__(
        self,
        *,
        config,
        graph_data,
        statistics,
        statistics_tendencies,
        data_indices,
        metadata,
        supporting_arrays,
    ) -> None:
        super().__init__(
            config=config,
            graph_data=graph_data,
            statistics=statistics,
            statistics_tendencies=statistics_tendencies,
            data_indices=data_indices,
            metadata=metadata,
            supporting_arrays=supporting_arrays,
        )
        self.input_times = list(range(self.n_step_input))
        fallback_relative_indices = list(range(self.n_step_input + self.n_step_output))
        fallback_target_indices = list(range(self.n_step_input, self.n_step_input + self.n_step_output))
        (
            self.dataset_relative_time_indices,
            self.dataset_input_relative_time_indices,
            self.dataset_target_relative_time_indices,
        ) = self._resolve_dataset_time_indices(
            metadata=metadata,
            fallback_relative_indices=fallback_relative_indices,
            fallback_input_indices=self.input_times,
            fallback_target_indices=fallback_target_indices,
        )
        self.dataset_time_maps = {
            dataset_name: {int(time_idx): batch_idx for batch_idx, time_idx in enumerate(relative_indices)}
            for dataset_name, relative_indices in self.dataset_relative_time_indices.items()
        }
        self._uses_dataset_sparse_timing = any(
            self.dataset_input_relative_time_indices[dataset_name] != self.input_times
            or self.dataset_target_relative_time_indices[dataset_name] != fallback_target_indices
            for dataset_name in self.dataset_names
        )
        sparse_rollout_cfg = getattr(config.training, "sparse_rollout", {})
        self.rollout_forcing_policy = getattr(sparse_rollout_cfg, "forcing_policy", "last_available")
        self._rollout_sampler_warning_keys: set[tuple[str, int, int]] = set()
        for dataset_name in self.dataset_names:
            LOGGER.info(
                "ForecasterSparse dataset=%s relative_times=%s input_times=%s target_times=%s",
                dataset_name,
                self.dataset_relative_time_indices[dataset_name],
                self.dataset_input_relative_time_indices[dataset_name],
                self.dataset_target_relative_time_indices[dataset_name],
            )

    def _resolve_dataset_time_indices(
        self,
        *,
        metadata: dict,
        fallback_relative_indices: list[int],
        fallback_input_indices: list[int],
        fallback_target_indices: list[int],
    ) -> tuple[dict[str, list[int]], dict[str, list[int]], dict[str, list[int]]]:
        metadata_inference = metadata.get("metadata_inference", {}) if isinstance(metadata, Mapping) else {}
        relative_by_dataset: Mapping | None = None
        input_by_dataset: Mapping | None = None
        target_by_dataset: Mapping | None = None
        for dataset_name in self.dataset_names:
            dataset_meta = metadata_inference.get(dataset_name, {}) if isinstance(metadata_inference, Mapping) else {}
            timesteps_meta = dataset_meta.get("timesteps", {}) if isinstance(dataset_meta, Mapping) else {}
            candidate_relative = timesteps_meta.get("relative_date_indices_training_by_dataset", None)
            candidate_input = timesteps_meta.get("relative_date_input_indices_training_by_dataset", None)
            candidate_target = timesteps_meta.get("relative_date_target_indices_training_by_dataset", None)
            if relative_by_dataset is None and isinstance(candidate_relative, Mapping):
                relative_by_dataset = candidate_relative
            if input_by_dataset is None and isinstance(candidate_input, Mapping):
                input_by_dataset = candidate_input
            if target_by_dataset is None and isinstance(candidate_target, Mapping):
                target_by_dataset = candidate_target
            if relative_by_dataset is not None and input_by_dataset is not None and target_by_dataset is not None:
                break

        dataset_relative_indices: dict[str, list[int]] = {}
        dataset_input_indices: dict[str, list[int]] = {}
        dataset_target_indices: dict[str, list[int]] = {}
        for dataset_name in self.dataset_names:
            relative_indices = None
            if relative_by_dataset is not None:
                raw_indices = relative_by_dataset.get(dataset_name, None)
                if isinstance(raw_indices, (list, tuple, ListConfig)):
                    relative_indices = [int(v) for v in raw_indices]
            if not relative_indices:
                relative_indices = [int(v) for v in fallback_relative_indices]

            input_indices = None
            if input_by_dataset is not None:
                raw_indices = input_by_dataset.get(dataset_name, None)
                if isinstance(raw_indices, (list, tuple, ListConfig)):
                    input_indices = [int(v) for v in raw_indices]
            if not input_indices:
                input_indices = [int(v) for v in fallback_input_indices]

            target_indices = None
            if target_by_dataset is not None:
                raw_indices = target_by_dataset.get(dataset_name, None)
                if isinstance(raw_indices, (list, tuple, ListConfig)):
                    target_indices = [int(v) for v in raw_indices]
            if target_indices is None:
                target_indices = [int(v) for v in fallback_target_indices]

            relative_set = set(relative_indices)
            if any(int(value) not in relative_set for value in input_indices):
                raise ValueError(
                    f"Dataset '{dataset_name}' input relative times must be present in dataset relative times: "
                    f"input={input_indices}, relative={relative_indices}"
                )
            if any(int(value) not in relative_set for value in target_indices):
                raise ValueError(
                    f"Dataset '{dataset_name}' target relative times must be present in dataset relative times: "
                    f"target={target_indices}, relative={relative_indices}"
                )

            dataset_relative_indices[dataset_name] = relative_indices
            dataset_input_indices[dataset_name] = input_indices
            dataset_target_indices[dataset_name] = target_indices

        return dataset_relative_indices, dataset_input_indices, dataset_target_indices

    def _batch_positions(self, *, dataset_name: str, relative_times: list[int]) -> list[int]:
        time_map = self.dataset_time_maps.get(dataset_name, {})
        positions: list[int] = []
        for relative_time in relative_times:
            batch_idx = time_map.get(int(relative_time), None)
            if batch_idx is None:
                available = sorted(time_map.keys())
                raise ValueError(
                    f"Dataset '{dataset_name}' is missing exact relative time {relative_time}. "
                    f"Available times: {available}"
                )
            positions.append(int(batch_idx))
        return positions

    def _prediction_window_for_step(self, *, dataset_name: str, rollout_step: int) -> tuple[int, int]:
        input_relative_times = self.dataset_input_relative_time_indices[dataset_name]
        anchor_time = max(input_relative_times) + rollout_step * self.n_step_output
        return anchor_time + 1, anchor_time + self.n_step_output

    def _next_input_relative_times(self, *, dataset_name: str, rollout_step: int) -> list[int]:
        shift = (rollout_step + 1) * self.n_step_output
        return [int(relative_time + shift) for relative_time in self.dataset_input_relative_time_indices[dataset_name]]

    def _sample_batch_position(self, *, dataset_name: str, relative_time: int) -> int:
        time_map = self.dataset_time_maps.get(dataset_name, {})
        exact_idx = time_map.get(int(relative_time), None)
        if exact_idx is not None:
            return int(exact_idx)

        available_times = sorted(int(value) for value in time_map)
        if not available_times:
            raise ValueError(f"Dataset '{dataset_name}' has no available relative times for sparse rollout.")

        if self.rollout_forcing_policy == "last_available":
            candidate_times = [value for value in available_times if value <= int(relative_time)]
            if not candidate_times:
                raise ValueError(
                    f"Dataset '{dataset_name}' has no forcing/boundary time at or before relative time "
                    f"{relative_time}. Available times: {available_times}"
                )
            sampled_time = candidate_times[-1]
        elif self.rollout_forcing_policy == "exact":
            raise ValueError(
                f"Dataset '{dataset_name}' is missing exact relative time {relative_time}. "
                f"Available times: {available_times}"
            )
        else:
            raise ValueError(
                f"Unsupported sparse rollout forcing policy '{self.rollout_forcing_policy}'. "
                "Expected 'last_available' or 'exact'."
            )

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

    def _build_rollout_input_step(
        self,
        *,
        dataset_name: str,
        dataset_batch: torch.Tensor,
        y_pred_full: dict[str, torch.Tensor],
        relative_time: int,
        rollout_step: int,
    ) -> torch.Tensor:
        batch_position = self._sample_batch_position(dataset_name=dataset_name, relative_time=relative_time)
        x_step = dataset_batch[
            :,
            batch_position,
            ...,
            self.data_indices[dataset_name].data.input.full,
        ].clone()

        pred_start, pred_end = self._prediction_window_for_step(dataset_name=dataset_name, rollout_step=rollout_step)
        pred_position = int(relative_time - pred_start)
        has_prediction = pred_start <= int(relative_time) <= pred_end and dataset_name in y_pred_full
        if has_prediction:
            x_step[..., self.data_indices[dataset_name].model.input.prognostic] = y_pred_full[dataset_name][
                :,
                pred_position,
                ...,
                self.data_indices[dataset_name].model.output.prognostic,
            ]

        x_step = self.output_mask[dataset_name].rollout_boundary(
            x_step,
            dataset_batch[:, batch_position],
            self.data_indices[dataset_name],
            grid_shard_slice=self.grid_shard_slice[dataset_name],
        )

        x_step[..., self.data_indices[dataset_name].model.input.forcing] = dataset_batch[
            :,
            batch_position,
            ...,
            self.data_indices[dataset_name].data.input.forcing,
        ]
        return x_step

    def _advance_input(
        self,
        x: dict[str, torch.Tensor],
        y_pred: dict[str, torch.Tensor],
        batch: dict[str, torch.Tensor],
        rollout_step: int,
    ) -> dict[str, torch.Tensor]:
        del x
        next_x: dict[str, torch.Tensor] = {}
        for dataset_name, dataset_batch in batch.items():
            next_input_relative_times = self._next_input_relative_times(
                dataset_name=dataset_name,
                rollout_step=rollout_step,
            )
            next_steps = [
                self._build_rollout_input_step(
                    dataset_name=dataset_name,
                    dataset_batch=dataset_batch,
                    y_pred_full=y_pred,
                    relative_time=relative_time,
                    rollout_step=rollout_step,
                )
                for relative_time in next_input_relative_times
            ]
            next_x[dataset_name] = torch.stack(next_steps, dim=1)
        return next_x

    def _rollout_targets_for_step(self, *, dataset_name: str, rollout_step: int) -> tuple[list[int], list[int]]:
        input_relative_times = self.dataset_input_relative_time_indices[dataset_name]
        target_relative_times = self.dataset_target_relative_time_indices[dataset_name]
        anchor_time = max(input_relative_times) + rollout_step * self.n_step_output
        step_start = anchor_time + 1
        step_end = anchor_time + self.n_step_output
        step_target_relative_times = [
            int(relative_time)
            for relative_time in target_relative_times
            if step_start <= int(relative_time) <= step_end
        ]
        if len(step_target_relative_times) == 0:
            raise ValueError(
                f"Dataset '{dataset_name}' has no target times in rollout window [{step_start}, {step_end}]."
            )
        pred_positions = [int(relative_time - step_start) for relative_time in step_target_relative_times]
        return step_target_relative_times, pred_positions

    def _rollout_step(
        self,
        batch: dict,
        rollout: int | None = None,
        validation_mode: bool = False,
    ) -> Generator[tuple[torch.Tensor | None, dict, list]]:
        """Rollout step for the sparse forecaster."""
        rollout_steps = rollout or self.rollout

        x = {}
        for dataset_name, dataset_batch in batch.items():
            input_positions = self._batch_positions(
                dataset_name=dataset_name,
                relative_times=self.dataset_input_relative_time_indices[dataset_name],
            )
            input_index = torch.tensor(input_positions, device=dataset_batch.device, dtype=torch.long)
            x_time = dataset_batch.index_select(1, input_index)
            x[dataset_name] = x_time[..., self.data_indices[dataset_name].data.input.full]

        for rollout_step in range(rollout_steps):
            y_pred_full = self(x)
            y_pred = {}
            y = {}
            for dataset_name, dataset_batch in batch.items():
                if dataset_name not in y_pred_full:
                    continue
                if len(self.dataset_target_relative_time_indices[dataset_name]) == 0:
                    continue

                target_relative_times, pred_positions = self._rollout_targets_for_step(
                    dataset_name=dataset_name,
                    rollout_step=rollout_step,
                )
                batch_positions = self._batch_positions(
                    dataset_name=dataset_name,
                    relative_times=target_relative_times,
                )
                pred_index = torch.tensor(pred_positions, device=dataset_batch.device, dtype=torch.long)
                batch_index = torch.tensor(batch_positions, device=dataset_batch.device, dtype=torch.long)
                y_time = dataset_batch.index_select(1, batch_index)
                y[dataset_name] = y_time
                y_pred[dataset_name] = y_pred_full[dataset_name].index_select(1, pred_index)

            loss, metrics_next, y_pred = checkpoint(
                self.compute_loss_metrics,
                y_pred,
                y,
                step=rollout_step,
                validation_mode=validation_mode,
                input_context=x,
                use_reentrant=False,
            )

            if rollout_step < rollout_steps - 1:
                x = self._advance_input(x, y_pred_full, batch, rollout_step=rollout_step)

            yield loss, metrics_next, y_pred


class GraphEnsForecasterSparse(GraphForecasterSparse):
    """Sparse forecaster task with ensemble loss gathering."""

    task_type = "forecaster"

    def __init__(
        self,
        *,
        config,
        graph_data,
        statistics,
        statistics_tendencies,
        data_indices,
        metadata,
        supporting_arrays,
    ) -> None:
        super().__init__(
            config=config,
            graph_data=graph_data,
            statistics=statistics,
            statistics_tendencies=statistics_tendencies,
            data_indices=data_indices,
            metadata=metadata,
            supporting_arrays=supporting_arrays,
        )

        self.model_comm_group_size = config.system.hardware.num_gpus_per_model
        num_gpus_per_model = config.system.hardware.num_gpus_per_model
        num_gpus_per_ensemble = config.system.hardware.num_gpus_per_ensemble

        assert num_gpus_per_ensemble % num_gpus_per_model == 0, (
            "Invalid ensemble vs. model size GPU group configuration: "
            f"{num_gpus_per_ensemble} mod {num_gpus_per_model} != 0."
        )

        self.lr = (
            config.system.hardware.num_nodes
            * config.system.hardware.num_gpus_per_node
            * config.training.lr.rate
            / num_gpus_per_ensemble
        )
        LOGGER.info("Base (config) learning rate: %e -- Effective learning rate: %e", config.training.lr.rate, self.lr)

        self.nens_per_device = config.training.ensemble_size_per_device
        self.nens_per_group = self.nens_per_device * num_gpus_per_ensemble // num_gpus_per_model
        LOGGER.info("Ensemble size: per device = %d, per ens-group = %d", self.nens_per_device, self.nens_per_group)

        self.ens_comm_group = None
        self.ens_comm_group_id = None
        self.ens_comm_group_rank = None
        self.ens_comm_num_groups = None
        self.ens_comm_group_size = None
        self.ens_comm_subgroup = None
        self.ens_comm_subgroup_id = None
        self.ens_comm_subgroup_rank = None
        self.ens_comm_subgroup_num_groups = None
        self.ens_comm_subgroup_size = None

    def set_ens_comm_group(
        self,
        ens_comm_group: ProcessGroup,
        ens_comm_group_id: int,
        ens_comm_group_rank: int,
        ens_comm_num_groups: int,
        ens_comm_group_size: int,
    ) -> None:
        self.ens_comm_group = ens_comm_group
        self.ens_comm_group_id = ens_comm_group_id
        self.ens_comm_group_rank = ens_comm_group_rank
        self.ens_comm_num_groups = ens_comm_num_groups
        self.ens_comm_group_size = ens_comm_group_size

    def set_ens_comm_subgroup(
        self,
        ens_comm_subgroup: ProcessGroup,
        ens_comm_subgroup_id: int,
        ens_comm_subgroup_rank: int,
        ens_comm_subgroup_num_groups: int,
        ens_comm_subgroup_size: int,
    ) -> None:
        self.ens_comm_subgroup = ens_comm_subgroup
        self.ens_comm_subgroup_id = ens_comm_subgroup_id
        self.ens_comm_subgroup_rank = ens_comm_subgroup_rank
        self.ens_comm_subgroup_num_groups = ens_comm_subgroup_num_groups
        self.ens_comm_subgroup_size = ens_comm_subgroup_size

    def compute_dataset_loss_metrics(
        self,
        y_pred: torch.Tensor,
        y: torch.Tensor,
        validation_mode: bool = False,
        dataset_name: str | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor | None, dict[str, torch.Tensor], torch.Tensor]:
        y_pred_ens = gather_tensor(
            y_pred.clone(),
            dim=TensorDim.ENSEMBLE_DIM,
            shapes=[y_pred.shape] * self.ens_comm_subgroup_size,
            mgroup=self.ens_comm_subgroup,
        )
        return super().compute_dataset_loss_metrics(
            y_pred_ens,
            y,
            validation_mode=validation_mode,
            dataset_name=dataset_name,
            **kwargs,
        )

    def _build_rollout_input_step(
        self,
        *,
        dataset_name: str,
        dataset_batch: torch.Tensor,
        y_pred_full: dict[str, torch.Tensor],
        relative_time: int,
        rollout_step: int,
    ) -> torch.Tensor:
        batch_position = self._sample_batch_position(dataset_name=dataset_name, relative_time=relative_time)
        x_step = dataset_batch[
            :,
            batch_position,
            ...,
            self.data_indices[dataset_name].data.input.full,
        ].clone()

        if self.nens_per_device > 1:
            x_step = torch.cat([x_step] * self.nens_per_device, dim=1)

        pred_start, pred_end = self._prediction_window_for_step(dataset_name=dataset_name, rollout_step=rollout_step)
        pred_position = int(relative_time - pred_start)
        has_prediction = pred_start <= int(relative_time) <= pred_end and dataset_name in y_pred_full
        if has_prediction:
            x_step[..., self.data_indices[dataset_name].model.input.prognostic] = y_pred_full[dataset_name][
                :,
                pred_position,
                ...,
                self.data_indices[dataset_name].model.output.prognostic,
            ]

        boundary = dataset_batch[:, batch_position]
        if self.nens_per_device > 1:
            boundary = torch.cat([boundary] * self.nens_per_device, dim=1)

        x_step = self.output_mask[dataset_name].rollout_boundary(
            x_step,
            boundary,
            self.data_indices[dataset_name],
            grid_shard_slice=self.grid_shard_slice[dataset_name],
        )

        forcing = dataset_batch[
            :,
            batch_position,
            ...,
            self.data_indices[dataset_name].model.input.forcing,
        ]
        if self.nens_per_device > 1:
            forcing = torch.cat([forcing] * self.nens_per_device, dim=1)
        x_step[..., self.data_indices[dataset_name].model.input.forcing] = forcing
        return x_step

    def _rollout_step(
        self,
        batch: dict,
        rollout: int | None = None,
        validation_mode: bool = False,
    ) -> Generator[tuple[torch.Tensor | None, dict, list]]:
        """Rollout step for the sparse ensemble forecaster."""
        rollout_steps = rollout or self.rollout

        x = {}
        for dataset_name, dataset_batch in batch.items():
            input_positions = self._batch_positions(
                dataset_name=dataset_name,
                relative_times=self.dataset_input_relative_time_indices[dataset_name],
            )
            input_index = torch.tensor(input_positions, device=dataset_batch.device, dtype=torch.long)
            x_time = dataset_batch.index_select(1, input_index)
            x[dataset_name] = x_time[..., self.data_indices[dataset_name].data.input.full]

        for dataset_name in self.dataset_names:
            x[dataset_name] = torch.cat([x[dataset_name]] * self.nens_per_device, dim=2)
            LOGGER.debug("Shapes: x[%s].shape = %s", dataset_name, list(x[dataset_name].shape))
            assert (
                x[dataset_name].shape[1] == len(self.dataset_input_relative_time_indices[dataset_name])
                and x[dataset_name].shape[2] == self.nens_per_device
            ), (
                "Shape mismatch in x! "
                f"Expected ({len(self.dataset_input_relative_time_indices[dataset_name])}, {self.nens_per_device}), "
                f"got ({x[dataset_name].shape[1]}, {x[dataset_name].shape[2]})!"
            )

        for rollout_step in range(rollout_steps):
            y_pred_full = self(x, fcstep=rollout_step)
            y_pred = {}
            y = {}
            for dataset_name, dataset_batch in batch.items():
                if dataset_name not in y_pred_full:
                    continue
                if len(self.dataset_target_relative_time_indices[dataset_name]) == 0:
                    continue

                target_relative_times, pred_positions = self._rollout_targets_for_step(
                    dataset_name=dataset_name,
                    rollout_step=rollout_step,
                )
                batch_positions = self._batch_positions(
                    dataset_name=dataset_name,
                    relative_times=target_relative_times,
                )
                pred_index = torch.tensor(pred_positions, device=dataset_batch.device, dtype=torch.long)
                batch_index = torch.tensor(batch_positions, device=dataset_batch.device, dtype=torch.long)
                y_time = dataset_batch.index_select(1, batch_index)
                y[dataset_name] = y_time[:, :, 0, :, :]
                y_pred[dataset_name] = y_pred_full[dataset_name].index_select(1, pred_index)

            loss, metrics_next, y_pred_ens = checkpoint(
                self.compute_loss_metrics,
                y_pred,
                y,
                step=rollout_step,
                validation_mode=validation_mode,
                input_context=x,
                use_reentrant=False,
            )

            if rollout_step < rollout_steps - 1:
                x = self._advance_input(x, y_pred_full, batch, rollout_step=rollout_step)

            yield loss, metrics_next, y_pred_ens
