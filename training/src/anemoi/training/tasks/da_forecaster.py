# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Data-assimilation forecasting task.

Extends the :class:`Forecaster` with ``da_cycles`` assimilation steps before the
autoregressive forecast rollout. In a DA cycle the model runs a forward pass and the
prediction is blended with the observations at the output times (observation where present,
model background where missing); the blended analysis is the input of the next cycle. After
the DA cycles the forecast rollout runs with the ``corrector`` (observation-metadata) inputs
zeroed, since no observations are available then.

DA and forecast targets are contiguous in time, so one continuous ``rollout_step`` index
(``0 .. da_cycles + rollout - 1``) addresses both and the parent's offsets work unchanged. Each
step is tagged ``is_da``. The training methods know nothing of DA: the task tells them, step by
step, how to weight the loss (``step_loss_weight``), when to checkpoint the model call
(``step_uses_checkpoint``), which observations to blend (``advance_needs_observations``) and
which residual base to use (``build_skip_input``).
"""

import datetime
import logging
from typing import TYPE_CHECKING

import torch

from anemoi.models.data_indices.collection import IndexCollection
from anemoi.training.tasks.forecaster import Forecaster

if TYPE_CHECKING:
    from anemoi.models.data.batch import Batch
    from anemoi.models.data.sources import Source

LOGGER = logging.getLogger(__name__)


class DAForecaster(Forecaster):
    """Forecasting task with data-assimilation cycling before the rollout."""

    name: str = "da_forecaster"

    def __init__(
        self,
        multistep_input: int,
        multistep_output: int,
        timestep: str,
        rollout: dict | None = None,
        validation_rollout: int | None = None,
        da_cycles: int = 0,
        da_loss_weight: float = 0.0,
        da_flow_dependent_skip: bool = False,
        checkpoint_steps: int = 0,
        **kwargs,
    ) -> None:
        """Initialise the DA forecaster.

        Parameters
        ----------
        multistep_input, multistep_output, timestep, rollout, validation_rollout
            As for :class:`Forecaster`; ``rollout`` counts the forecast steps only.
        da_cycles : int
            Number of assimilation cycles before the forecast rollout.
        da_loss_weight : float
            Weight of each DA cycle's loss; 0 skips their loss (the total loss is averaged over
            the forecast steps only).
        da_flow_dependent_skip : bool
            After a DA blend, hand the model the pre-blend background as the residual base, so
            the additive skip connection stays smooth instead of stamping sparse observations into
            the output. Requires a model supporting ``skip_input``.
        checkpoint_steps : int
            Number of leading training steps (DA cycles first) whose model call is
            activation-checkpointed: trades compute for memory without changing gradients.
        **kwargs
            Passed to :class:`Forecaster`.
        """
        if da_cycles < 0:
            msg = f"da_cycles must be non-negative, got {da_cycles}."
            raise ValueError(msg)
        if da_loss_weight < 0:
            msg = f"da_loss_weight must be non-negative, got {da_loss_weight}."
            raise ValueError(msg)
        self.da_cycles = da_cycles
        self.da_loss_weight = da_loss_weight
        self.da_flow_dependent_skip = da_flow_dependent_skip
        self.checkpoint_steps = checkpoint_steps
        super().__init__(
            multistep_input,
            multistep_output,
            timestep,
            rollout=rollout,
            validation_rollout=validation_rollout,
            **kwargs,
        )
        max_steps = da_cycles + self.rollout.maximum
        if not 0 <= checkpoint_steps <= max_steps:
            msg = f"checkpoint_steps must be in [0, da_cycles + rollout.maximum = {max_steps}], got {checkpoint_steps}."
            raise ValueError(msg)
        if da_cycles > 0:
            LOGGER.info(
                "DAForecaster: da_cycles=%d, da_loss_weight=%.3f, da_flow_dependent_skip=%s, checkpoint_steps=%d",
                da_cycles,
                da_loss_weight,
                da_flow_dependent_skip,
                checkpoint_steps,
            )
        elif da_flow_dependent_skip:
            LOGGER.warning("DAForecaster: da_flow_dependent_skip=True has no effect with da_cycles=0.")

    @property
    def requires_skip_input(self) -> bool:
        return self.da_flow_dependent_skip and self.da_cycles > 0

    def steps(self, mode: str = "training") -> tuple[dict[str, int | bool], ...]:
        """Return the DA cycles followed by the forecast rollout steps.

        Each step carries a continuous ``rollout_step`` index and an ``is_da`` flag.
        """
        num_forecast_steps = len(super().steps(mode))
        return tuple(
            {"rollout_step": i, "is_da": i < self.da_cycles} for i in range(self.da_cycles + num_forecast_steps)
        )

    def get_offsets(self, mode: str | None = None) -> list[datetime.timedelta]:
        """Return the offsets covering the DA cycles and the forecast rollout.

        Without a mode these are the offsets of the longest configured rollout, of which the
        training and validation offsets are a prefix (batch indices are taken against them).
        """
        if mode in ("training", "validation"):
            return super().get_offsets(mode)
        return self._compute_rollout_offsets(self.da_cycles + max(self.rollout.maximum, self.validation_rollout or 0))

    # -- step policy -------------------------------------------------------------------------

    @property
    def num_loss_steps(self) -> int:
        """The training loss is averaged over the forecast steps only."""
        return self.num_steps - self.da_cycles

    def step_loss_weight(self, is_da: bool = False, **_kwargs) -> float:
        return self.da_loss_weight if is_da else 1.0

    def step_uses_checkpoint(self, rollout_step: int = 0, **_kwargs) -> bool:
        return rollout_step < self.checkpoint_steps

    def advance_needs_observations(self, is_da: bool = False, **_kwargs) -> bool:
        return is_da

    def get_metric_name(self, rollout_step: int = 0, is_da: bool = False, **_kwargs) -> str:
        if is_da:
            return f"_dacycle{rollout_step}"
        return f"_rstep{rollout_step - self.da_cycles}"

    def get_step_label(self, rollout_step: int = 0, is_da: bool = False, **_kwargs) -> str:
        if is_da:
            return f"dacycle{rollout_step:02d}"
        return f"rstep{rollout_step - self.da_cycles:02d}"

    def get_metric_step(self, rollout_step: int | None = None, is_da: bool = False, **_kwargs) -> int | str | None:
        """Forecast steps keep the plain ``/1, /2, ...`` metric suffixes; DA cycles are ``/da1, /da2, ...``."""
        if rollout_step is None:
            return None
        if is_da:
            return f"da{rollout_step + 1}"
        return rollout_step - self.da_cycles

    # -- state advance -----------------------------------------------------------------------

    def advance_input(
        self,
        x: "Batch",
        y_pred: "Batch",
        output_values: "Batch",
        rollout_step: int = 0,
        is_da: bool = False,
        data_indices: dict[str, IndexCollection] | None = None,
        output_mask: dict[str, object] | None = None,
        grid_shard_slice: dict[str, slice | None] | None = None,
        observed: "Batch | None" = None,
    ) -> "Batch":
        """Advance the input state for the next step.

        A DA cycle takes the observations where present and the raw model background where they
        are missing, for the prognostic variables, and the observed metadata (``corrector``
        variables). A forecast step autoregresses as the parent does, with the metadata zeroed.

        Parameters
        ----------
        x : Batch
            Current input, in the model input variables.
        y_pred : Batch
            Raw prediction of the step, in the model output variables.
        output_values : Batch
            True state at the output times, in the full data variables, pre-processed as inputs.
        rollout_step : int
            Continuous step index.
        is_da : bool
            Whether the step is a DA cycle.
        data_indices : dict[str, IndexCollection]
            Data indices per dataset.
        output_mask : dict[str, object], optional
            Output masks per gridded dataset.
        grid_shard_slice : dict[str, slice | None], optional
            Local grid shard per gridded dataset.
        observed : Batch, optional
            The observations at the output times, pre-processed without imputation so missing
            values stay NaN. Required for a DA cycle.

        Returns
        -------
        Batch
            The input for the next step.
        """
        del rollout_step
        if is_da and observed is None:
            msg = "A DA cycle needs the observations at its output times (observed=...)."
            raise ValueError(msg)
        return x.with_sources(
            {
                dataset_name: self._advance_da_source(
                    view,
                    y_pred.get(dataset_name),
                    output_values[dataset_name],
                    None if observed is None else observed[dataset_name],
                    is_da=is_da,
                    data_indices=data_indices[dataset_name],
                    output_mask=None if output_mask is None else output_mask[dataset_name],
                    grid_shard_slice=None if grid_shard_slice is None else grid_shard_slice.get(dataset_name),
                )
                for dataset_name, view in x.items()
            },
        )

    def _advance_da_source(
        self,
        x: "Source",
        prediction: "Source | None",
        truth: "Source",
        observed: "Source | None",
        is_da: bool,
        data_indices: IndexCollection,
        output_mask: object | None = None,
        grid_shard_slice: slice | None = None,
    ) -> "Source":
        """Advance one dataset: blend (DA) or autoregress (forecast), then set the metadata inputs."""
        corrector = data_indices.data.input.corrector
        if is_da and prediction is not None:
            prediction = _blend_observations(prediction, observed, data_indices)
        if not is_da and len(corrector) > 0:
            # No observations in the forecast: their metadata reads as "nothing here".
            truth = _zero_variables(truth, corrector)

        advanced = self._advance_source(
            x,
            prediction,
            truth,
            data_indices=data_indices,
            output_mask=output_mask,
            grid_shard_slice=grid_shard_slice,
        )
        if x.is_tabular or len(corrector) == 0:
            # A tabular window that comes from the output is the truth window, metadata included.
            return advanced

        # The gridded advance only writes prognostics and forcings into the new input windows.
        new_data = advanced.data.clone()
        model_corrector = data_indices.model.input.corrector
        for out_idx, new_idx in self._advance_map["outin"]:
            new_data[:, new_idx, ..., model_corrector] = truth.data[:, out_idx, ..., corrector].to(new_data.dtype)
        return advanced.clone(data=new_data)

    # -- residual base -----------------------------------------------------------------------

    def build_skip_input(
        self,
        x: "Batch",
        y_pred: "Batch | None" = None,
        data_indices: dict[str, IndexCollection] | None = None,
        is_da: bool = False,
        **_kwargs,
    ) -> "Batch | None":
        """Residual base of the next model call: the pure model background after a DA blend.

        Passed to the model as ``skip_input``, it becomes the base of the additive skip connection
        while the encoder still sees the observations. The skip therefore stays smooth and
        flow-dependent, and observations reach the output only through the encoder.

        Before the first step (``y_pred=None``) no background exists, and every position is zero,
        standing for "nothing known here": a ``ClimatologySkipConnection(missing_value=0.0,
        fill_missing_only=True)`` then returns climatology for the first DA cycle, a plain
        ``SkipConnection`` no residual. After a forecast step the input is already a raw
        prediction, so ``None`` (no override) is returned.

        Parameters
        ----------
        x : Batch
            Model input after the advance (DA: with the observations blended in).
        y_pred : Batch, optional
            Raw prediction of the step that was just advanced, or ``None`` before the first step.
        data_indices : dict[str, IndexCollection], optional
            Data indices per dataset. Required with ``y_pred``.
        is_da : bool
            Whether the step that was just advanced is a DA cycle.

        Returns
        -------
        Batch | None
            The residual base, or ``None`` to keep the input.
        """
        if not self.requires_skip_input:
            return None
        if y_pred is None:
            return x.map_data(torch.zeros_like)
        if not is_da:
            return None

        sources = {}
        for dataset_name, view in x.items():
            prediction = y_pred.get(dataset_name)
            if view.is_tabular or prediction is None:
                sources[dataset_name] = view
                continue
            background = view.data.clone()
            prog_in = data_indices[dataset_name].model.input.prognostic
            prog_out = data_indices[dataset_name].model.output.prognostic
            for out_idx, new_idx in self._advance_map["outin"]:
                background[:, new_idx, ..., prog_in] = prediction.data[:, out_idx, ..., prog_out].to(background.dtype)
            sources[dataset_name] = view.clone(data=background)
        return x.with_sources(sources)


def _zero_variables(source: "Source", indices: torch.Tensor) -> "Source":
    """Return ``source`` with the given variables (positions in its variable axis, last) set to 0."""

    def zero(data: torch.Tensor) -> torch.Tensor:
        data = data.clone()
        data[..., indices.to(data.device)] = 0.0
        return data

    return source.map_data(zero)


def _blend_observations(prediction: "Source", observed: "Source", data_indices: IndexCollection) -> "Source":
    """Return the prediction with its prognostic values replaced by the observations where present.

    ``observed`` holds the full data variables at the same output times and nodes, with missing
    observations as NaN; a single observed member is broadcast to the predicted members.
    """
    prog_out = data_indices.model.output.prognostic
    prog_data = data_indices.data.input.prognostic

    def blend(predicted: torch.Tensor, obs: torch.Tensor) -> torch.Tensor:
        obs_prog = obs[..., prog_data.to(obs.device)].to(predicted.dtype)
        out = predicted.clone()
        out[..., prog_out] = torch.where(torch.isnan(obs_prog), predicted[..., prog_out], obs_prog)
        return out

    if prediction.is_tabular:
        return prediction.clone(
            data=[blend(p, o) for p, o in zip(prediction.data, observed.data, strict=True)],
        )
    return prediction.clone(data=blend(prediction.data, observed.data))
