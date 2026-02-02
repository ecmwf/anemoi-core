# (C) Copyright 2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch
from torch.utils.checkpoint import checkpoint

from anemoi.training.train.objectives import DiffusionObjective
from anemoi.training.train.objectives import DirectPredictionObjective
from anemoi.training.train.objectives import FlowObjective

from .base import BaseGraphModule

if TYPE_CHECKING:
    from torch_geometric.data import HeteroData

    from anemoi.models.data_indices.collection import IndexCollection
    from anemoi.training.schemas.base_schema import BaseSchema

LOGGER = logging.getLogger(__name__)


class GraphTendForecaster(BaseGraphModule):
    """Graph neural network forecaster for tendency prediction."""

    task_type = "forecaster"
    supported_objectives = (DirectPredictionObjective, DiffusionObjective, FlowObjective)

    def __init__(
        self,
        *,
        config: BaseSchema,
        graph_data: dict[str, HeteroData],
        statistics: dict,
        statistics_tendencies: dict,
        data_indices: dict[str, IndexCollection],
        metadata: dict,
        supporting_arrays: dict,
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
        self._tendency_pre_processors: dict[str, object] = {}
        self._tendency_post_processors: dict[str, object] = {}
        self._validate_tendency_processors()

    def get_input(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        x = {}
        for dataset_name, dataset_batch in batch.items():
            msg = (
                f"Batch length not sufficient for requested multi_step length for {dataset_name}!"
                f", {dataset_batch.shape[1]} !>= {self.multi_step + self.multi_out}"
            )
            assert dataset_batch.shape[1] >= self.multi_step + self.multi_out, msg
            x[dataset_name] = dataset_batch[
                :,
                0 : self.multi_step,
                ...,
                self.data_indices[dataset_name].data.input.full,
            ]
            LOGGER.debug("SHAPE: x[%s].shape = %s", dataset_name, list(x[dataset_name].shape))
        return x

    def get_target(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        y = {}
        for dataset_name, dataset_batch in batch.items():
            start = self.multi_step
            y_time = dataset_batch.narrow(1, start, self.multi_out)
            var_idx = self.data_indices[dataset_name].data.output.full.to(device=dataset_batch.device)
            y[dataset_name] = y_time.index_select(-1, var_idx)
            LOGGER.debug("SHAPE: y[%s].shape = %s", dataset_name, list(y[dataset_name].shape))
        return y

    def _validate_tendency_processors(self) -> None:
        stats = self.statistics_tendencies
        assert stats is not None, "Tendency statistics are required for tendency models."

        pre_processors_tendencies = getattr(self.model, "pre_processors_tendencies", None)
        post_processors_tendencies = getattr(self.model, "post_processors_tendencies", None)
        assert (
            pre_processors_tendencies is not None and post_processors_tendencies is not None
        ), "Per-step tendency processors are required for multi-output tendency models."

        for dataset_name in self.dataset_names:
            dataset_stats = stats.get(dataset_name) if isinstance(stats, dict) else None
            assert dataset_stats is not None, f"Tendency statistics are required for dataset '{dataset_name}'."
            lead_times = dataset_stats.get("lead_times") if isinstance(dataset_stats, dict) else None
            assert isinstance(lead_times, list), "Tendency statistics must include 'lead_times'."
            assert (
                len(lead_times) == self.multi_out
            ), f"Expected {self.multi_out} tendency statistics entries, got {len(lead_times)}."
            assert all(
                lead_time in dataset_stats for lead_time in lead_times
            ), "Missing tendency statistics for one or more output steps."

            assert (
                dataset_name in pre_processors_tendencies
            ), "Per-step tendency processors are required for multi-output tendency models."
            assert (
                dataset_name in post_processors_tendencies
            ), "Per-step tendency processors are required for multi-output tendency models."

            pre_tend = pre_processors_tendencies[dataset_name]
            post_tend = post_processors_tendencies[dataset_name]
            assert (
                len(pre_tend) == self.multi_out and len(post_tend) == self.multi_out
            ), "Per-step tendency processors must match multistep_output."
            assert all(proc is not None for proc in pre_tend), "Missing tendency pre-processors for output steps."
            assert all(proc is not None for proc in post_tend), "Missing tendency post-processors for output steps."

            self._tendency_pre_processors[dataset_name] = pre_tend
            self._tendency_post_processors[dataset_name] = post_tend

    def _compute_tendency_target(
        self,
        y: dict[str, torch.Tensor],
        x_ref: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        tendencies: dict[str, torch.Tensor] = {}
        for dataset_name, y_dataset in y.items():
            pre_tend = self._tendency_pre_processors[dataset_name]
            tendency_steps = []
            for step, pre_proc in enumerate(pre_tend):
                y_step = y_dataset[:, step : step + 1]
                x_ref_step = x_ref[dataset_name].unsqueeze(1)
                tendency_step = self.model.model.compute_tendency(
                    {dataset_name: y_step},
                    {dataset_name: x_ref_step},
                    {dataset_name: self.model.pre_processors[dataset_name]},
                    {dataset_name: pre_proc},
                    input_post_processor={dataset_name: self.model.post_processors[dataset_name]},
                    skip_imputation=True,
                )[dataset_name]
                tendency_steps.append(tendency_step)
            tendencies[dataset_name] = torch.cat(tendency_steps, dim=1)
        return tendencies

    def _reconstruct_state(
        self,
        x_ref: dict[str, torch.Tensor],
        tendency: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        states: dict[str, torch.Tensor] = {}
        for dataset_name, tendency_dataset in tendency.items():
            post_tend = self._tendency_post_processors[dataset_name]
            state_steps = []
            for step, post_proc in enumerate(post_tend):
                x_ref_step = x_ref[dataset_name].unsqueeze(1)
                tendency_step = tendency_dataset[:, step : step + 1]
                state_step = self.model.model.add_tendency_to_state(
                    {dataset_name: x_ref_step},
                    {dataset_name: tendency_step},
                    {dataset_name: self.model.post_processors[dataset_name]},
                    {dataset_name: post_proc},
                    output_pre_processor={dataset_name: self.model.pre_processors[dataset_name]},
                    skip_imputation=True,
                )[dataset_name]
                state_steps.append(state_step)
            out_dataset = torch.cat(state_steps, dim=1)
            out_dataset = self.model.model._apply_imputer_inverse(self.model.post_processors, dataset_name, out_dataset)
            states[dataset_name] = out_dataset
        return states

    def compute_dataset_loss_metrics(
        self,
        y_pred: torch.Tensor,
        y: torch.Tensor,
        dataset_name: str,
        validation_mode: bool = False,
        metrics_y_pred: torch.Tensor | None = None,
        metrics_y: torch.Tensor | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor | None, dict[str, torch.Tensor], torch.Tensor]:
        y_pred_full, y_full, grid_shard_slice = self._prepare_tensors_for_loss(
            y_pred,
            y,
            validation_mode=validation_mode,
            dataset_name=dataset_name,
        )

        loss = self._compute_loss(
            y_pred_full,
            y_full,
            grid_shard_slice=grid_shard_slice,
            dataset_name=dataset_name,
            **kwargs,
        )

        metrics_next = {}
        metrics_pred_full = None
        if validation_mode and metrics_y_pred is not None and metrics_y is not None:
            metrics_pred_full, metrics_full, grid_shard_slice = self._prepare_tensors_for_loss(
                metrics_y_pred,
                self.model.model._apply_imputer_inverse(
                    self.model.post_processors,
                    dataset_name,
                    metrics_y,
                ),
                validation_mode=validation_mode,
                dataset_name=dataset_name,
            )

            metrics_next = self._compute_metrics(
                metrics_pred_full,
                metrics_full,
                grid_shard_slice=grid_shard_slice,
                dataset_name=dataset_name,
                **kwargs,
            )

        return loss, metrics_next, metrics_pred_full if validation_mode and metrics_y_pred is not None else None

    def _step(
        self,
        batch: dict[str, torch.Tensor],
        validation_mode: bool = False,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor], list[dict[str, torch.Tensor]]]:
        assert self.objective is not None, "GraphTendForecaster requires a training.objective configuration to be set."

        x = self.get_input(batch)
        y = self.get_target(batch)

        pre_processors_tendencies = getattr(self.model, "pre_processors_tendencies", None)
        assert pre_processors_tendencies is not None and len(pre_processors_tendencies) > 0, (
            "pre_processors_tendencies not found. This is required for tendency models. "
            "Ensure that statistics_tendencies is provided during model initialization."
        )

        x_ref = self.model.model.apply_reference_state_truncation(
            x,
            self.grid_shard_shapes,
            self.model_comm_group,
        )
        # Reference state for tendency computation (last input state).
        x_ref = {dataset_name: (ref[:, -1] if ref.ndim == 5 else ref) for dataset_name, ref in x_ref.items()}

        # Target tendencies derived from reference state and future states.
        tendency_target = self._compute_tendency_target(y, x_ref)

        shapes = {k: target.shape for k, target in tendency_target.items()}
        model_impl = self.model.model
        # Objective schedule for tendency space (e.g., sigma or time).
        # Unused in direct prediction objective.
        schedule = self.objective.sample_schedule(
            shape=shapes,
            device=next(iter(batch.values())).device,
            model=model_impl,
        )

        # Build conditioning/target for tendency prediction.
        # tendency_cond is None for direct prediction objective.
        tendency_cond, tendency_target_for_loss = self.objective.build_training_pair(tendency_target, schedule)
        # Forward pass in objective space (e.g., denoising/velocity on tendencies).
        tendency_pred = self.objective.forward(
            model_impl,
            x,
            tendency_cond,
            schedule,
            model_comm_group=self.model_comm_group,
            grid_shard_shapes=self.grid_shard_shapes,
        )
        # Clean prediction in normalized tendency space for metrics.
        tendency_pred_clean, _ = self.objective.clean_pred_target_pair(
            tendency_pred,
            tendency_target,
            tendency_cond,
            schedule,
        )
        # Optional pre-loss weights (e.g., diffusion noise weighting).
        # None for direct prediction objective.
        pre_loss_weights = self.objective.pre_loss_weights(schedule, model=model_impl)

        y_pred = None
        if validation_mode:
            # Reconstruct normalized state prediction from clean tendency for metrics.
            y_pred = self._reconstruct_state(x_ref, tendency_pred_clean)

        # Provide metrics tensors for the shared loss/metrics path.
        metrics_y_pred = y_pred if validation_mode else None
        metrics_y = y if validation_mode else None

        loss, metrics, y_pred = checkpoint(
            self.compute_loss_metrics,
            tendency_pred,
            tendency_target_for_loss,
            validation_mode=validation_mode,
            metrics_y_pred=metrics_y_pred,
            metrics_y=metrics_y,
            pre_loss_weights=pre_loss_weights,
            use_reentrant=False,
        )

        return loss, metrics, [y_pred]
