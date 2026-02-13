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

from anemoi.models.preprocessing import StepwiseProcessors

from .base import BaseGraphModule

if TYPE_CHECKING:
    from torch_geometric.data import HeteroData

    from anemoi.models.data_indices.collection import IndexCollection
    from anemoi.training.schemas.base_schema import BaseSchema

LOGGER = logging.getLogger(__name__)


class BaseDiffusionForecaster(BaseGraphModule):
    """Base class for diffusion forecasters."""

    task_type = "forecaster"

    def __init__(
        self,
        *,
        config: BaseSchema,
        task,
        graph_data: dict[str, HeteroData],
        statistics: dict,
        statistics_tendencies: dict,
        data_indices: dict[str, IndexCollection],
        metadata: dict,
        supporting_arrays: dict,
    ) -> None:

        super().__init__(
            config=config,
            task=task,
            graph_data=graph_data,
            statistics=statistics,
            statistics_tendencies=statistics_tendencies,
            data_indices=data_indices,
            metadata=metadata,
            supporting_arrays=supporting_arrays,
        )

        self.rho = config.model.model.diffusion.rho

    def forward(
        self,
        x: dict[str, torch.Tensor],
        y_noised: dict[str, torch.Tensor],
        sigma: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        return self.model.model.fwd_with_preconditioning(
            x,
            y_noised,
            sigma,
            model_comm_group=self.model_comm_group,
            grid_shard_shapes=self.grid_shard_shapes,
        )

    def _compute_loss(
        self,
        y_pred: torch.Tensor,
        y: torch.Tensor,
        dataset_name: str,
        weights: torch.Tensor | None = None,
        grid_shard_slice: slice | None = None,
        **_kwargs,
    ) -> torch.Tensor:
        """Compute the diffusion loss with noise weighting.

        Parameters
        ----------
        y_pred : torch.Tensor
            Predicted values
        y : torch.Tensor
            Target values
        grid_shard_slice : slice | None
            Grid shard slice for distributed training
        weights : torch.Tensor
            Noise weights for diffusion loss computation
        **_kwargs
            Additional arguments

        Returns
        -------
        torch.Tensor
            Computed loss with noise weighting applied
        """
        assert weights is not None, f"{self.__class__.__name__} must be provided for diffusion loss computation."

        return self.loss[dataset_name](
            y_pred,
            y,
            weights=weights[dataset_name],
            grid_shard_slice=grid_shard_slice,
            group=self.model_comm_group,
        )

    def _noise_target(self, x: dict[str, torch.Tensor], sigma: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Add noise to the state."""
        return {name: x[name] + torch.randn_like(x[name]) * sigma[name] for name in x}

    def _get_noise_level(
        self,
        shape: dict[str, tuple[int]],
        sigma_max: float,
        sigma_min: float,
        sigma_data: float,
        rho: float,
        device: torch.device,
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        sigma, weight = {}, {}
        dataset_names = list(shape.keys())
        ref_shape = shape[dataset_names[0]]
        # Expected shape: (batch, time, ensemble, grid, vars)
        assert len(ref_shape) == 5, "Expected 5D tensor shape (batch, time, ensemble, grid, vars) for diffusion noise."
        batch_size = ref_shape[0]
        ensemble_size = ref_shape[2]
        for dataset_name, shape_x in shape.items():
            assert (
                len(shape_x) == 5
            ), f"Expected 5D tensor shape (batch, time, ensemble, grid, vars) for dataset '{dataset_name}'."
            assert (
                shape_x[0] == batch_size and shape_x[2] == ensemble_size
            ), "Batch or ensemble dimension mismatch across datasets when sampling diffusion noise."

        base_shape = (batch_size, ensemble_size)
        rnd_uniform = torch.rand(base_shape, device=device)
        sigma_base = (
            sigma_max ** (1.0 / rho) + rnd_uniform * (sigma_min ** (1.0 / rho) - sigma_max ** (1.0 / rho))
        ) ** rho
        weight_base = (sigma_base**2 + sigma_data**2) / (sigma_base * sigma_data) ** 2
        sigma_base = sigma_base[:, None, :, None, None]
        weight_base = weight_base[:, None, :, None, None]

        for dataset_name in shape:
            sigma[dataset_name] = sigma_base
            weight[dataset_name] = weight_base
        return sigma, weight


class DiffusionProtocol(BaseDiffusionForecaster):
    """Graph neural network forecaster for diffusion."""

    def _step(
        self,
        batch: dict[str, torch.Tensor],
        validation_mode: bool = False,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor], list[dict[str, torch.Tensor]]]:
        """Step for the forecaster.

        Will run pre_processors on batch, but not post_processors on predictions.

        Parameters
        ----------
        batch : dict[str, torch.Tensor]
            Normalized batch to use for rollout (assumed to be already preprocessed).
        validation_mode : bool, optional
            Whether in validation mode, and to calculate validation metrics, by default False
            If False, metrics will be empty

        Returns
        -------
        tuple[torch.Tensor, dict[str, torch.Tensor], list[dict[str, torch.Tensor]]]
            Loss value, metrics, and predictions (per step)
        """
        loss = torch.zeros(1, dtype=next(iter(batch.values())).dtype, device=self.device, requires_grad=False)

        x = self.task.get_inputs(batch, data_indices=self.data_indices)  # (bs, n_step_input, ens, latlon, nvar)
        y = self.task.get_targets(batch, data_indices=self.data_indices)  # (bs, n_step_output, ens, latlon, nvar)

        # get noise level and associated loss weights
        shapes = {k: y_.shape for k, y_ in y.items()}
        sigma, noise_weights = self._get_noise_level(
            shape=shapes,
            sigma_max=self.model.model.sigma_max,
            sigma_min=self.model.model.sigma_min,
            sigma_data=self.model.model.sigma_data,
            rho=self.rho,
            device=next(iter(batch.values())).device,
        )

        y_noised = self._noise_target(y, sigma)
        # prediction, fwd_with_preconditioning
        y_pred = self(x, y_noised, sigma)  # shape is (bs, time, ens, latlon, nvar)
        loss, metrics, y_pred = checkpoint(
            self.compute_loss_metrics,
            y_pred,
            y,
            validation_mode=validation_mode,
            weights=noise_weights,
            use_reentrant=False,
        )

        return loss, metrics, [y_pred]


class DiffusionTendProtocol(BaseDiffusionForecaster):
    """Graph neural network forecaster for diffusion tendency prediction."""

    def __init__(
        self,
        *,
        config: BaseSchema,
        task,
        graph_data: dict[str, HeteroData],
        statistics: dict,
        statistics_tendencies: dict,
        data_indices: dict[str, IndexCollection],
        metadata: dict,
        supporting_arrays: dict,
    ) -> None:
        super().__init__(
            config=config,
            task=task,
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

    def _validate_tendency_processors(self) -> None:
        stats = self.statistics_tendencies
        assert stats is not None, "Tendency statistics are required for diffusion tendency models."

        pre_processors_tendencies = getattr(self.model, "pre_processors_tendencies", None)
        post_processors_tendencies = getattr(self.model, "post_processors_tendencies", None)
        assert (
            pre_processors_tendencies is not None and post_processors_tendencies is not None
        ), "Per-step tendency processors are required for multi-output diffusion tendency models."

        def _wrap_if_needed(
            kind: str,
            proc: object,
            dataset_name: str,
            lead_times: list[str],
        ) -> StepwiseProcessors:
            if isinstance(proc, StepwiseProcessors):
                return proc
            assert (
                self.n_step_output == 1
            ), "Per-step tendency processors are required for multi-output diffusion tendency models."
            lead_time = lead_times[0]
            wrapped = StepwiseProcessors([lead_time])
            wrapped.set(lead_time, proc)
            LOGGER.warning(
                "Wrapping flat tendency %s-processor for dataset '%s' into stepwise (single-step).",
                kind,
                dataset_name,
            )
            return wrapped

        for dataset_name in self.dataset_names:
            dataset_stats = stats.get(dataset_name) if isinstance(stats, dict) else None
            assert dataset_stats is not None, f"Tendency statistics are required for dataset '{dataset_name}'."
            lead_times = dataset_stats.get("lead_times") if isinstance(dataset_stats, dict) else None
            assert isinstance(lead_times, list), "Tendency statistics must include 'lead_times'."
            assert (
                len(lead_times) == self.n_step_output
            ), f"Expected {self.n_step_output} tendency statistics entries, got {len(lead_times)}."
            assert all(
                lead_time in dataset_stats for lead_time in lead_times
            ), "Missing tendency statistics for one or more output steps."

            assert (
                dataset_name in pre_processors_tendencies
            ), "Per-step tendency processors are required for multi-output diffusion tendency models."
            assert (
                dataset_name in post_processors_tendencies
            ), "Per-step tendency processors are required for multi-output diffusion tendency models."

            pre_tend = pre_processors_tendencies[dataset_name]
            post_tend = post_processors_tendencies[dataset_name]
            pre_tend = _wrap_if_needed("pre", pre_tend, dataset_name, lead_times)
            post_tend = _wrap_if_needed("post", post_tend, dataset_name, lead_times)
            assert (
                len(pre_tend) == self.n_step_output and len(post_tend) == self.n_step_output
            ), "Per-step tendency processors must match n_step_output."
            assert all(
                proc is not None for proc in pre_tend
            ), "Missing tendency pre-processors for one or more output steps."
            assert all(
                proc is not None for proc in post_tend
            ), "Missing tendency post-processors for one or more output steps."

            self._tendency_pre_processors[dataset_name] = pre_tend
            self._tendency_post_processors[dataset_name] = post_tend

    def _compute_tendency_target(
        self,
        y: dict[str, torch.Tensor],
        x_ref: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        tendencies: dict[str, torch.Tensor] = {}
        for dataset_name, y_dataset in y.items():
            # y is normalized data.output.full; x_ref is normalized model.input.prognostic (subset)
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
            # x_ref is normalized model.input.prognostic; tendency is normalized model.output.* space
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
        y_pred_state: dict[str, torch.Tensor] | None = None,
        y_state: dict[str, torch.Tensor] | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor | None, dict[str, torch.Tensor], torch.Tensor]:
        """Compute loss and metrics for the given predictions and targets."""
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
        if validation_mode:
            assert y_pred_state is not None, "y_pred_state must be provided for tendency-based diffusion models."
            assert y_state is not None, "y_state must be provided for tendency-based diffusion models."
            assert (
                dataset_name in y_pred_state
            ), f"{dataset_name} must be a key in y_pred_state for tendency-based diffusion models."
            assert (
                dataset_name in y_state
            ), f"{dataset_name} must be a key in y_state for tendency-based diffusion models."
            assert (
                y_pred_state[dataset_name] is not None
            ), "y_pred_state must be provided for tendency-based diffusion models."
            assert y_state[dataset_name] is not None, "y_state must be provided for tendency-based diffusion models."

            y_pred_state_full, y_state_full, grid_shard_slice = self._prepare_tensors_for_loss(
                y_pred_state[dataset_name],
                self.model.model._apply_imputer_inverse(
                    self.model.post_processors,
                    dataset_name,
                    y_state[dataset_name],
                ),
                validation_mode=validation_mode,
                dataset_name=dataset_name,
            )

            metrics_next = self._compute_metrics(
                y_pred_state_full,
                y_state_full,
                grid_shard_slice=grid_shard_slice,
                dataset_name=dataset_name,
                **kwargs,
            )

        return loss, metrics_next, y_pred_state_full if validation_mode else None

    def _step(
        self,
        batch: dict[str, torch.Tensor],
        validation_mode: bool = False,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor], list[dict[str, torch.Tensor]]]:
        """Step for the tendency-based diffusion forecaster.

        Will run pre_processors on batch, but not post_processors on predictions.

        Parameters
        ----------
        batch : dict[str, torch.Tensor]
            Normalized batch to use for rollout (assumed to be already preprocessed).
        validation_mode : bool, optional
            Whether in validation mode, and to calculate validation metrics, by default False
            If False, metrics will be empty

        Returns
        -------
        tuple[torch.Tensor, dict[str, torch.Tensor], list[dict[str, torch.Tensor]]]
            Loss value, metrics, and predictions (per step)
        """
        # batch is already normalized in BaseGraphModule._normalize_batch
        # x: data.input.full (normalized), y: data.output.full (normalized)
        x = self.task.get_inputs(batch, data_indices=self.data_indices)  # (bs, n_step_input, ens, latlon, nvar)
        y = self.task.get_targets(batch, data_indices=self.data_indices)  # (bs, n_step_output, ens, latlon, nvar)

        pre_processors_tendencies = getattr(self.model, "pre_processors_tendencies", None)
        if pre_processors_tendencies is None or len(pre_processors_tendencies) == 0:
            msg = (
                "pre_processors_tendencies not found. This is required for tendency-based diffusion models. "
                "Ensure that statistics_tendencies is provided during model initialization."
            )
            raise AttributeError(msg)

        x_ref = self.model.model.apply_reference_state_truncation(
            x,
            self.grid_shard_shapes,
            self.model_comm_group,
        )
        # x_ref is normalized model.input.prognostic (subset), aligned to output steps
        x_ref = {dataset_name: (ref[:, -1] if ref.ndim == 5 else ref) for dataset_name, ref in x_ref.items()}

        tendency_target = self._compute_tendency_target(y, x_ref)

        # get noise level and associated loss weights
        shapes = {k: target.shape for k, target in tendency_target.items()}
        sigma, noise_weights = self._get_noise_level(
            shape=shapes,
            sigma_max=self.model.model.sigma_max,
            sigma_min=self.model.model.sigma_min,
            sigma_data=self.model.model.sigma_data,
            rho=self.rho,
            device=next(iter(batch.values())).device,
        )

        tendency_target_noised = self._noise_target(tendency_target, sigma)

        # prediction, fwd_with_preconditioning
        tendency_pred = self(x, tendency_target_noised, sigma)  # shape is (bs, time, ens, latlon, nvar)

        y_pred = None
        if validation_mode:
            y_pred = self._reconstruct_state(x_ref, tendency_pred)

        loss, metrics, y_pred = checkpoint(
            self.compute_loss_metrics,
            tendency_pred,
            tendency_target,
            y_pred_state=y_pred,
            y_state=y,
            validation_mode=validation_mode,
            weights=noise_weights,
            use_reentrant=False,
        )

        return loss, metrics, [y_pred]
