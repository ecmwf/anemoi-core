# (C) Copyright 2026 Anemoi contributors.
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
from anemoi.models.transport import reference_state_sampling_source
from anemoi.training.diagnostics.callbacks.plot_adapter import EnsemblePlotAdapterWrapper
from anemoi.training.train.methods.base import BaseTrainingModule
from anemoi.training.train.methods.edm_diffusion import EDMDiffusionTransportObjective
from anemoi.training.train.methods.stochastic_interpolant import StochasticInterpolantTransportObjective
from anemoi.training.train.methods.transport_base import PreparedPredictionTarget
from anemoi.training.train.methods.transport_base import TransportObjective
from anemoi.training.train.step_output import TrainingStepOutput
from anemoi.training.utils.index_space import IndexSpace

if TYPE_CHECKING:
    from anemoi.training.losses.loss_tree import LossTree

LOGGER = logging.getLogger(__name__)


class PredictionMode:
    """Prepare either state targets or tendency targets for transport training."""

    def __init__(self, module: BaseTransportTraining) -> None:
        self.module = module

    def prepare_target(
        self,
        batch: dict[str, torch.Tensor],
        x: dict[str, torch.Tensor],
    ) -> PreparedPredictionTarget:
        raise NotImplementedError

    def reconstruct_prediction(
        self,
        prediction: dict[str, torch.Tensor],
        prepared: PreparedPredictionTarget,
    ) -> dict[str, torch.Tensor]:
        raise NotImplementedError

    def prepare_metric_target(self, prepared: PreparedPredictionTarget) -> dict[str, torch.Tensor]:
        return prepared.metric_target


class StatePredictionMode(PredictionMode):
    """Prediction mode where the model learns the future state directly."""

    def _reference_state_target_space(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        # Use the latest input state as a source field, selecting the same
        # variables that the model predicts for the future state.
        reference: dict[str, torch.Tensor] = {}
        for dataset_name, batch_dataset in batch.items():
            var_idx = self.module.data_indices[dataset_name].data.output.full.to(device=batch_dataset.device)
            reference_step = batch_dataset.narrow(1, self.module.n_step_input - 1, 1).index_select(-1, var_idx)
            if self.module.n_step_output > 1:
                reference_step = reference_step.expand(-1, self.module.n_step_output, -1, -1, -1)
            reference[dataset_name] = reference_step
        return self.module.reduce_data_output_target_to_model_output(reference)

    def prepare_target(
        self,
        batch: dict[str, torch.Tensor],
        x: dict[str, torch.Tensor],
    ) -> PreparedPredictionTarget:
        del x
        target_full = self.module.task.get_targets(batch, data_indices=self.module.data_indices)
        target_data_output = self.module.get_data_output_target(target_full)
        model_target = self.module.reduce_data_output_target_to_model_output(target_data_output)
        return PreparedPredictionTarget(
            model_target=model_target,
            loss_target=target_full,
            loss_target_layout=IndexSpace.DATA_FULL,
            metric_target=target_full,
            aux={
                # For state prediction, the reference source is already in the
                # same state space as the target, so store it directly.
                "transport_reference_source": self._reference_state_target_space(batch),
            },
        )

    def reconstruct_prediction(
        self,
        prediction: dict[str, torch.Tensor],
        prepared: PreparedPredictionTarget,
    ) -> dict[str, torch.Tensor]:
        del prepared
        return prediction


class TendencyPredictionMode(PredictionMode):
    """Prediction mode where the model learns changes from the latest input state."""

    def __init__(self, module: BaseTransportTraining) -> None:
        super().__init__(module)
        self._tendency_pre_processors: dict[str, object] = {}
        self._tendency_post_processors: dict[str, object] = {}
        self._validate_tendency_processors()

    def _validate_tendency_processors(self) -> None:
        stats = self.module.statistics_tendencies
        assert stats is not None, "Tendency statistics are required for tendency-based transport models."

        pre_processors_tendencies = getattr(self.module.model, "pre_processors_tendencies", None)
        post_processors_tendencies = getattr(self.module.model, "post_processors_tendencies", None)
        assert (
            pre_processors_tendencies is not None and post_processors_tendencies is not None
        ), "Per-step tendency processors are required for multi-output tendency-based transport models."

        def _wrap_if_needed(
            kind: str,
            proc: object,
            dataset_name: str,
            lead_times: list[str],
        ) -> StepwiseProcessors:
            if isinstance(proc, StepwiseProcessors):
                return proc
            # Single-output tendency models may still provide one flat
            # Processors object. We wrap it so the rest so we can always
            # here iterate over per-step processors. Multi-output models
            # need an explicit processor for each lead time.
            assert (
                self.module.n_step_output == 1
            ), "Per-step tendency processors are required for multi-output tendency-based transport models."
            lead_time = lead_times[0]
            wrapped = StepwiseProcessors([lead_time])
            wrapped.set(lead_time, proc)
            LOGGER.warning(
                "Wrapping flat tendency %s-processor for dataset '%s' into stepwise (single-step).",
                kind,
                dataset_name,
            )
            return wrapped

        for dataset_name in self.module.dataset_names:
            dataset_stats = stats.get(dataset_name) if isinstance(stats, dict) else None
            assert dataset_stats is not None, f"Tendency statistics are required for dataset '{dataset_name}'."
            lead_times = dataset_stats.get("lead_times") if isinstance(dataset_stats, dict) else None
            assert isinstance(lead_times, list), "Tendency statistics must include 'lead_times'."
            assert (
                len(lead_times) == self.module.n_step_output
            ), f"Expected {self.module.n_step_output} tendency statistics entries, got {len(lead_times)}."
            assert all(
                lead_time in dataset_stats for lead_time in lead_times
            ), "Missing tendency statistics for one or more output steps."

            assert (
                dataset_name in pre_processors_tendencies
            ), "Per-step tendency processors are required for multi-output tendency-based transport models."
            assert (
                dataset_name in post_processors_tendencies
            ), "Per-step tendency processors are required for multi-output tendency-based transport models."

            pre_tend = pre_processors_tendencies[dataset_name]
            post_tend = post_processors_tendencies[dataset_name]
            pre_tend = _wrap_if_needed("pre", pre_tend, dataset_name, lead_times)
            post_tend = _wrap_if_needed("post", post_tend, dataset_name, lead_times)
            assert (
                len(pre_tend) == self.module.n_step_output and len(post_tend) == self.module.n_step_output
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
            pre_tend = self._tendency_pre_processors[dataset_name]
            tendency_steps = []
            for step, pre_proc in enumerate(pre_tend):
                y_step = y_dataset[:, step : step + 1]
                x_ref_step = x_ref[dataset_name].unsqueeze(1)
                tendency_step = self.module.model.model.compute_tendency(
                    {dataset_name: y_step},
                    {dataset_name: x_ref_step},
                    {dataset_name: self.module.model.pre_processors[dataset_name]},
                    {dataset_name: pre_proc},
                    input_post_processor={dataset_name: self.module.model.post_processors[dataset_name]},
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
                state_step = self.module.model.model.add_tendency_to_state(
                    {dataset_name: x_ref_step},
                    {dataset_name: tendency_step},
                    {dataset_name: self.module.model.post_processors[dataset_name]},
                    {dataset_name: post_proc},
                    output_pre_processor={dataset_name: self.module.model.pre_processors[dataset_name]},
                    skip_imputation=True,
                )[dataset_name]
                state_steps.append(state_step)
            out_dataset = torch.cat(state_steps, dim=1)
            out_dataset = self.module.model.model._apply_imputer_inverse(
                self.module.model.post_processors,
                dataset_name,
                out_dataset,
            )
            states[dataset_name] = out_dataset
        return states

    def prepare_target(
        self,
        batch: dict[str, torch.Tensor],
        x: dict[str, torch.Tensor],
    ) -> PreparedPredictionTarget:
        """Build tendency targets for training and state targets for validation metrics."""
        state_target = self.module.task.get_targets(batch)
        y_data_output = self.module.get_data_output_target(state_target)

        pre_processors_tendencies = getattr(self.module.model, "pre_processors_tendencies", None)
        if pre_processors_tendencies is None or len(pre_processors_tendencies) == 0:
            msg = (
                "pre_processors_tendencies not found. This is required for tendency-based transport models. "
                "Ensure that statistics_tendencies is provided during model initialization."
            )
            raise AttributeError(msg)

        x_ref = self.module.model.model.apply_reference_state_truncation(
            x,
            self.module.grid_shard_sizes,
            self.module.model_comm_group,
        )
        x_ref = {dataset_name: (ref[:, -1] if ref.ndim == 5 else ref) for dataset_name, ref in x_ref.items()}

        tendency_target_data_output = self._compute_tendency_target(y_data_output, x_ref)
        tendency_target = self.module.reduce_data_output_target_to_model_output(tendency_target_data_output)
        return PreparedPredictionTarget(
            model_target=tendency_target,
            loss_target=tendency_target_data_output,
            loss_target_layout=IndexSpace.DATA_OUTPUT,
            metric_target=state_target,
            aux={
                # x_ref is the latest input state used to turn states into
                # tendencies and tendencies back into states.
                "x_ref": x_ref,
                # Build a reference-state source only if source.kind asks for it;
                # Gaussian and zero sources do not need this projection.
                "transport_reference_source": lambda: reference_state_sampling_source(
                    x,
                    data_indices=self.module.data_indices,
                    n_step_output=self.module.n_step_output,
                ),
            },
        )

    def reconstruct_prediction(
        self,
        prediction: dict[str, torch.Tensor],
        prepared: PreparedPredictionTarget,
    ) -> dict[str, torch.Tensor]:
        return self._reconstruct_state(prepared.aux["x_ref"], prediction)

    def prepare_metric_target(self, prepared: PreparedPredictionTarget) -> dict[str, torch.Tensor]:
        return {
            dataset_name: self.module.model.model._apply_imputer_inverse(
                self.module.model.post_processors,
                dataset_name,
                target,
            )
            for dataset_name, target in prepared.metric_target.items()
        }


PREDICTION_MODE_CLASSES = {
    "state": StatePredictionMode,
    "tendency": TendencyPredictionMode,
}


TRANSPORT_OBJECTIVE_CLASSES = {
    "edm_diffusion": EDMDiffusionTransportObjective,
    "stochastic_interpolant": StochasticInterpolantTransportObjective,
}


class BaseTransportTraining(BaseTrainingModule):
    """Shared training code for transport methods that corrupt targets before prediction."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._prediction_mode = self._get_prediction_mode_cls()(self)

    def _get_prediction_mode_cls(self) -> type[PredictionMode]:
        prediction_mode = self.config.training.transport.prediction_mode
        try:
            return PREDICTION_MODE_CLASSES[prediction_mode]
        except KeyError as exc:
            msg = f"Unknown training.transport.prediction_mode '{prediction_mode}'."
            raise ValueError(msg) from exc

    @property
    def prediction_mode(self) -> PredictionMode:
        """Return the state/tendency target handler for this module."""
        return self._prediction_mode

    @property
    def plot_adapter(self) -> EnsemblePlotAdapterWrapper:
        """Wrap the task plot adapter with ensemble-dimension handling."""
        if not hasattr(self, "_ensemble_plot_adapter"):
            self._ensemble_plot_adapter = EnsemblePlotAdapterWrapper(self.task._plot_adapter)
        return self._ensemble_plot_adapter

    def get_data_output_target(self, target_full: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Select the target variables that are present in the dataset output."""
        y = {}
        for dataset_name, target_dataset in target_full.items():
            var_idx = self.data_indices[dataset_name].data.output.full.to(device=target_dataset.device)
            y[dataset_name] = target_dataset.index_select(-1, var_idx)
            LOGGER.debug(
                "SHAPE: y_data_output[%s].shape = %s",
                dataset_name,
                list(y[dataset_name].shape),
            )
        return y

    def reduce_data_output_target_to_model_output(
        self,
        y_data_output: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Select only the variables that the model predicts."""
        y_reduced = {}
        for dataset_name, y_dataset in y_data_output.items():
            dataset_indices = self.data_indices[dataset_name]
            if dataset_indices.model_output_in_data_output_is_identity:
                y_reduced[dataset_name] = y_dataset
            elif dataset_indices.model_output_in_data_output_is_contiguous:
                y_reduced[dataset_name] = y_dataset.narrow(
                    -1,
                    dataset_indices.model_output_in_data_output_contiguous_start,
                    dataset_indices.model_output_in_data_output_contiguous_length,
                )
            else:
                var_idx = torch.as_tensor(
                    dataset_indices.model_output_positions_in_data_output,
                    device=y_dataset.device,
                    dtype=torch.long,
                )
                y_reduced[dataset_name] = y_dataset.index_select(-1, var_idx)
            LOGGER.debug(
                "SHAPE: y_model_output[%s].shape = %s",
                dataset_name,
                list(y_reduced[dataset_name].shape),
            )
        return y_reduced

    def compute_dataset_loss_metrics(
        self,
        y_pred: torch.Tensor,
        y: torch.Tensor,
        dataset_name: str,
        validation_mode: bool = False,
        metric_prediction: dict[str, torch.Tensor] | None = None,
        metric_target: dict[str, torch.Tensor] | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor | LossTree | None, dict[str, torch.Tensor], torch.Tensor]:
        """Compute loss according to the objective and validation metrics in clean-state space."""
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
            assert metric_prediction is not None, "metric_prediction must be provided for validation metrics."
            assert metric_target is not None, "metric_target must be provided for validation metrics."
            assert dataset_name in metric_prediction, f"{dataset_name} must be a key in metric_prediction."
            assert dataset_name in metric_target, f"{dataset_name} must be a key in metric_target."
            assert metric_prediction[dataset_name] is not None, "metric_prediction must be provided."
            assert metric_target[dataset_name] is not None, "metric_target must be provided."

            metric_prediction_full, metric_target_full, grid_shard_slice = self._prepare_tensors_for_loss(
                metric_prediction[dataset_name],
                metric_target[dataset_name],
                validation_mode=validation_mode,
                dataset_name=dataset_name,
            )

            metric_kwargs = {k: v for k, v in kwargs.items() if k not in {"pred_layout", "target_layout"}}
            metrics_next = self._compute_metrics(
                metric_prediction_full,
                metric_target_full,
                grid_shard_slice=grid_shard_slice,
                dataset_name=dataset_name,
                pred_layout=IndexSpace.MODEL_OUTPUT,
                target_layout=IndexSpace.DATA_FULL,
                **metric_kwargs,
            )
            return loss, metrics_next, metric_prediction_full

        return loss, metrics_next, y_pred


class TransportTraining(BaseTransportTraining):
    """Training module for EDM diffusion and stochastic-interpolant transport models."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._validate_model_transport_objective()
        self._transport_objective = self._get_transport_objective_cls()(self)

    def _validate_model_transport_objective(self) -> None:
        """Check that training and inference use the same transport objective."""
        training_objective = self.config.training.transport.objective
        model_config = getattr(getattr(self.config, "model", None), "model", None)
        model_transport_config = getattr(model_config, "transport", None)
        model_objective = getattr(model_transport_config, "objective", None)
        if model_objective is not None and model_objective != training_objective:
            msg = (
                "training.transport.objective must match model.model.transport.objective "
                f"({training_objective!r} != {model_objective!r})."
            )
            raise ValueError(msg)

    def _get_transport_objective_cls(self) -> type[TransportObjective]:
        transport_objective = self.config.training.transport.objective
        try:
            return TRANSPORT_OBJECTIVE_CLASSES[transport_objective]
        except KeyError as exc:
            msg = f"Unknown training.transport.objective '{transport_objective}'."
            raise ValueError(msg) from exc

    @property
    def transport_objective(self) -> TransportObjective:
        """Return the selected transport objective for this module."""
        return self._transport_objective

    def forward(
        self,
        x: dict[str, torch.Tensor],
        conditioned_target: dict[str, torch.Tensor],
        condition: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        return self.transport_objective.forward(x, conditioned_target, condition)

    def _compute_loss(
        self,
        y_pred: torch.Tensor,
        y: torch.Tensor,
        grid_shard_slice: slice | None = None,
        dataset_name: str | None = None,
        pred_layout: IndexSpace | str | None = None,
        target_layout: IndexSpace | str | None = None,
        **kwargs,
    ) -> torch.Tensor:
        return self.transport_objective.compute_loss(
            y_pred=y_pred,
            y=y,
            grid_shard_slice=grid_shard_slice,
            dataset_name=dataset_name,
            pred_layout=pred_layout,
            target_layout=target_layout,
            **kwargs,
        )

    def _step(
        self,
        batch: dict[str, torch.Tensor],
        validation_mode: bool = False,
    ) -> TrainingStepOutput:
        """Run one training or validation step for the selected transport objective."""
        x = self.task.get_inputs(batch, data_indices=self.data_indices)
        prepared_target = self.prediction_mode.prepare_target(batch, x)
        prepared_objective = self.transport_objective.prepare(prepared_target)

        prediction = self(x, prepared_objective.conditioned_target, prepared_objective.condition)
        loss_prediction = self.transport_objective.prepare_loss_prediction(prediction, prepared_objective)

        metric_prediction = None
        metric_target = None
        plot_kwargs: dict[str, dict[str, torch.Tensor]] = {}
        if validation_mode:
            conditioned_endpoint = self.prediction_mode.reconstruct_prediction(
                prepared_objective.conditioned_target,
                prepared_target,
            )
            plot_kwargs["auxiliary_output"] = {
                dataset_name: target.detach() for dataset_name, target in conditioned_endpoint.items()
            }
            endpoint_prediction = self.transport_objective.reconstruct_endpoint(prediction, prepared_objective)
            metric_prediction = self.prediction_mode.reconstruct_prediction(endpoint_prediction, prepared_target)
            metric_target = self.prediction_mode.prepare_metric_target(prepared_target)

        loss, metrics, y_pred = checkpoint(
            self.compute_loss_metrics,
            loss_prediction,
            prepared_objective.loss_target,
            metric_prediction=metric_prediction,
            metric_target=metric_target,
            weights=prepared_objective.weights,
            validation_mode=validation_mode,
            pred_layout=prepared_objective.pred_layout,
            target_layout=prepared_objective.loss_target_layout,
            use_reentrant=False,
        )

        return TrainingStepOutput(loss=loss, metrics=metrics, predictions=[y_pred], plot_kwargs=plot_kwargs)
