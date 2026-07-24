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

from anemoi.models.data import Batch
from anemoi.models.preprocessing import StepwiseProcessors
from anemoi.models.transport import reference_state_sampling_source
from anemoi.models.transport.data_helpers import is_sparse_data
from anemoi.training.diagnostics.callbacks.plot_adapter import EnsemblePlotAdapterWrapper
from anemoi.training.train.methods.base import BaseTrainingModule
from anemoi.training.train.methods.edm_diffusion import EDMDiffusionTransportObjective
from anemoi.training.train.methods.stochastic_interpolant import StochasticInterpolantTransportObjective
from anemoi.training.train.methods.transport_base import PreparedPredictionTarget
from anemoi.training.train.methods.transport_base import TransportObjective
from anemoi.training.train.step_output import TrainingStepOutput
from anemoi.training.utils.index_space import IndexSpace

if TYPE_CHECKING:
    from anemoi.models.data.views import SourceView

LOGGER = logging.getLogger(__name__)


class PredictionMode:
    """Prepare either state targets or tendency targets for transport training."""

    def __init__(self, module: BaseTransportTraining) -> None:
        self.module = module

    def prepare_target(
        self,
        batch: Batch,
        x: Batch,
    ) -> PreparedPredictionTarget:
        raise NotImplementedError

    def reconstruct_prediction(
        self,
        prediction: Batch,
        prepared: PreparedPredictionTarget,
    ) -> Batch:
        raise NotImplementedError

    def prepare_metric_target(self, prepared: PreparedPredictionTarget) -> Batch:
        return prepared.metric_target


class StatePredictionMode(PredictionMode):
    """Prediction mode where the model learns the future state directly."""

    def _reference_state_target_space(self, batch: Batch) -> Batch:
        # Use the latest input state as a source field, selecting the same
        # variables that the model predicts for the future state. The state is
        # taken from the full batch rather than the model inputs because
        # diagnostic output variables are not part of the model input, and it
        # is normalized (with imputation) like the model inputs.
        reference_state = self.module.preprocess_inputs(batch.select(time=self.module.n_step_input - 1))
        reference_data: dict[str, torch.Tensor] = {}
        for dataset_name, state_view in reference_state.items():
            var_idx = self.module.data_indices[dataset_name].data.output.full.tolist()
            reference_step = state_view.select(variables=var_idx).data
            if self.module.n_step_output > 1:
                if isinstance(reference_step, list):
                    msg = "Multi-step reference-state transport sources are not supported for sparse datasets."
                    raise NotImplementedError(msg)
                reference_step = reference_step.expand(-1, self.module.n_step_output, -1, -1, -1)
            reference_data[dataset_name] = reference_step
        return self.module.reduce_data_output_target_to_model_output(reference_state.with_data(reference_data))

    def prepare_target(
        self,
        batch: Batch,
        x: Batch,
    ) -> PreparedPredictionTarget:
        del x
        raw_target, target_forcing = self.module.task.get_targets(batch, data_indices=self.module.data_indices)
        # Loss and metric targets keep their NaNs so missing observations are
        # masked in the loss (imputation skipped).
        target_full = self.module.preprocess_targets(raw_target)
        # The model target is corrupted and fed through the network, so it must
        # be imputed like the model inputs: NaNs would spread through message
        # passing and poison the whole prediction.
        model_state = self.module.preprocess_inputs(raw_target)
        model_target = self.module.reduce_data_output_target_to_model_output(
            self.module.get_data_output_target(model_state),
        )
        # NaN-preserving counterpart of model_target, used by objectives that
        # derive their loss target from model_target (e.g. the SI drift) to
        # re-mask missing observations.
        model_target_missing = self.module.reduce_data_output_target_to_model_output(
            self.module.get_data_output_target(target_full),
        )
        return PreparedPredictionTarget(
            model_target=model_target,
            loss_target=target_full,
            loss_target_layout=IndexSpace.DATA_FULL,
            metric_target=target_full,
            aux={
                # Build the reference-state source lazily so gaussian and zero
                # sources never pay for (or crash on) this projection.
                "transport_reference_source": lambda: self._reference_state_target_space(batch),
                # Output-time decoding forcings, normalized like the model inputs.
                "target_forcing": self.module.preprocess_inputs(target_forcing),
                "model_target_missing": model_target_missing,
            },
        )

    def reconstruct_prediction(
        self,
        prediction: Batch,
        prepared: PreparedPredictionTarget,
    ) -> Batch:
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
        batch: Batch,
        x: Batch,
    ) -> PreparedPredictionTarget:
        """Build tendency targets for training and state targets for validation metrics."""
        if any(is_sparse_data(dataset_data) for dataset_data in batch.data.values()):
            msg = "Tendency prediction mode is not implemented for sparse observation datasets."
            raise NotImplementedError(msg)

        raw_state_target, target_forcing = self.module.task.get_targets(batch, data_indices=self.module.data_indices)
        # Tendency targets are fed through the network (as the corrupted target)
        # and converted back to states for metrics, so they are imputed like the
        # model inputs; prepare_metric_target re-inserts the missing values via
        # the imputer inverse.
        state_target = self.module.preprocess_inputs(raw_state_target)
        y_data_output = self.module.get_data_output_target(state_target)

        pre_processors_tendencies = getattr(self.module.model, "pre_processors_tendencies", None)
        if pre_processors_tendencies is None or len(pre_processors_tendencies) == 0:
            msg = (
                "pre_processors_tendencies not found. This is required for tendency-based transport models. "
                "Ensure that statistics_tendencies is provided during model initialization."
            )
            raise AttributeError(msg)

        x_ref = self.module.model.model.apply_reference_state_truncation(
            x.data,
            {name: self.module._grid_shard_sizes(view) for name, view in x.items()},
            self.module.model_comm_group,
        )
        x_ref = {dataset_name: (ref[:, -1] if ref.ndim == 5 else ref) for dataset_name, ref in x_ref.items()}

        tendency_target_data_output = y_data_output.with_data(self._compute_tendency_target(y_data_output.data, x_ref))
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
                    x.data,
                    data_indices=self.module.data_indices,
                    n_step_output=self.module.n_step_output,
                ),
                # Output-time decoding forcings, normalized like the model inputs.
                "target_forcing": self.module.preprocess_inputs(target_forcing),
            },
        )

    def reconstruct_prediction(
        self,
        prediction: Batch,
        prepared: PreparedPredictionTarget,
    ) -> Batch:
        reconstructed = self._reconstruct_state(prepared.aux["x_ref"], prediction.data)
        return prepared.metric_target.with_data(reconstructed)

    def prepare_metric_target(self, prepared: PreparedPredictionTarget) -> Batch:
        metric_data = {
            dataset_name: self.module.model.model._apply_imputer_inverse(
                self.module.model.post_processors,
                dataset_name,
                target.data,
            )
            for dataset_name, target in prepared.metric_target.items()
        }
        return prepared.metric_target.with_data(metric_data)


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

    def get_data_output_target(self, target_full: Batch) -> Batch:
        """Select the target variables that are present in the dataset output."""
        y = {}
        for dataset_name, target_dataset in target_full.items():
            var_idx = self.data_indices[dataset_name].data.output.full.tolist()
            y[dataset_name] = target_dataset.select(variables=var_idx).data
            LOGGER.debug(
                "SHAPE: y_data_output[%s].shape = %s",
                dataset_name,
                y[dataset_name].shape if hasattr(y[dataset_name], "shape") else [t.shape for t in y[dataset_name]],
            )
        return target_full.with_data(y)

    def reduce_data_output_target_to_model_output(
        self,
        y_data_output: Batch,
    ) -> Batch:
        """Select only the variables that the model predicts."""
        y_reduced = {}
        for dataset_name, y_dataset in y_data_output.items():
            dataset_indices = self.data_indices[dataset_name]
            if dataset_indices.model_output_in_data_output_is_identity:
                y_reduced[dataset_name] = y_dataset.data
            elif dataset_indices.model_output_in_data_output_is_contiguous:
                start = dataset_indices.model_output_in_data_output_contiguous_start
                length = dataset_indices.model_output_in_data_output_contiguous_length
                y_reduced[dataset_name] = y_dataset.select(variables=slice(start, start + length)).data
            else:
                y_reduced[dataset_name] = y_dataset.select(
                    variables=dataset_indices.model_output_positions_in_data_output,
                ).data
            LOGGER.debug(
                "SHAPE: y_model_output[%s].shape = %s",
                dataset_name,
                (
                    y_reduced[dataset_name].shape
                    if hasattr(y_reduced[dataset_name], "shape")
                    else [t.shape for t in y_reduced[dataset_name]]
                ),
            )
        return y_data_output.with_data(y_reduced)

    def compute_dataset_loss_metrics(
        self,
        y_pred: SourceView,
        y: SourceView,
        dataset_name: str,
        validation_mode: bool = False,
        metric_prediction: Batch | None = None,
        metric_target: Batch | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor | None, dict[str, torch.Tensor], SourceView]:
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
        x: Batch,
        conditioned_target: Batch,
        condition: dict[str, torch.Tensor],
        target_forcing: Batch | None = None,
    ) -> Batch:
        return self.transport_objective.forward(x, conditioned_target, condition, target_forcing=target_forcing)

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

    def sample(
        self,
        batch: Batch,
        *,
        task_kwargs: dict | None = None,
        schedule_params: dict | None = None,
        sampler_params: dict | None = None,
        **kwargs,
    ) -> Batch:
        """Sample from a full task batch using the deterministic target-template pattern."""
        assert isinstance(batch, Batch), "batch must be a Batch instance"
        task_kwargs = {} if task_kwargs is None else task_kwargs
        x = self.task.get_inputs(batch, data_indices=self.data_indices)
        _, target_template = self.task.get_targets(batch, data_indices=self.data_indices, **task_kwargs)
        # The template carries the output-time decoding forcings in the same
        # normalization state as the caller-provided batch (matching x).
        return self.model.model.sample(
            x,
            target_template=target_template,
            model_comm_group=self.model_comm_group,
            grid_shard_sizes=self.grid_shard_sizes,
            schedule_params=schedule_params,
            sampler_params=sampler_params,
            target_forcing=target_template,
            **kwargs,
        )

    def _step(
        self,
        batch: Batch,
        validation_mode: bool = False,
    ) -> TrainingStepOutput:
        """Run one training or validation step for the selected transport objective."""
        x = self.preprocess_inputs(self.task.get_inputs(batch, data_indices=self.data_indices))
        prepared_target = self.prediction_mode.prepare_target(batch, x)
        prepared_objective = self.transport_objective.prepare(prepared_target)

        prediction = self(
            x,
            prepared_objective.conditioned_target,
            prepared_objective.condition,
            target_forcing=prepared_target.aux.get("target_forcing"),
        )
        loss_prediction = self.transport_objective.prepare_loss_prediction(prediction, prepared_objective)

        metric_prediction = None
        metric_target = None
        plot_kwargs: dict[str, dict[str, SourceView]] = {}
        if validation_mode:
            conditioned_endpoint = self.prediction_mode.reconstruct_prediction(
                prepared_objective.conditioned_target,
                prepared_target,
            )
            plot_kwargs["auxiliary_output"] = {
                dataset_name: target.apply_func(lambda data, **_: data.detach())
                for dataset_name, target in conditioned_endpoint.items()
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
