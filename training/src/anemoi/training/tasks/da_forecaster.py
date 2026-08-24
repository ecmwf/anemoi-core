# (C) Copyright 2024- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Data-assimilation forecasting task.

Extends the standard :class:`Forecaster` with ``da_cycles`` assimilation steps
before the autoregressive forecast rollout. In each DA cycle the model runs a
forward pass and the prediction is blended with observations (observation where
present, model background where NaN), and the blended analysis state is used as
the input for the next cycle. After the DA cycles, the standard forecast rollout
runs with the corrector input slots zeroed (no observations available).

DA and forecast targets are contiguous in time, so a single continuous
``rollout_step`` index (0..da_cycles+rollout-1) addresses both: the parent's
``get_output_offsets`` / ``get_batch_output_indices`` work unchanged. Each step
is tagged ``is_da`` so the training method can weight the DA loss separately and
blend (rather than autoregress) the DA-cycle state.
"""

import datetime
import logging

import torch

from anemoi.models.data_indices.collection import IndexCollection
from anemoi.training.tasks.forecaster import Forecaster

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
        **kwargs,
    ) -> None:
        self.da_cycles = da_cycles
        self.da_loss_weight = da_loss_weight
        self.da_flow_dependent_skip = da_flow_dependent_skip
        super().__init__(
            multistep_input,
            multistep_output,
            timestep,
            rollout=rollout,
            validation_rollout=validation_rollout,
            **kwargs,
        )
        if da_cycles > 0:
            LOGGER.info(
                "DAForecaster: da_cycles=%d, da_loss_weight=%.3f, da_flow_dependent_skip=%s",
                da_cycles,
                da_loss_weight,
                da_flow_dependent_skip,
            )
        elif da_flow_dependent_skip:
            LOGGER.warning(
                "DAForecaster: da_flow_dependent_skip=True has no effect with da_cycles=0.",
            )

    def steps(self, mode: str = "training") -> tuple[dict[str, int | bool], ...]:
        """Return the DA cycles followed by the forecast rollout steps.

        Each step dict carries a continuous ``rollout_step`` index and an
        ``is_da`` flag distinguishing assimilation cycles from forecast steps.
        """
        max_rollout = self.rollout.step
        if mode == "validation" and self.validation_rollout is not None:
            max_rollout = self.validation_rollout
        return tuple({"rollout_step": i, "is_da": i < self.da_cycles} for i in range(self.da_cycles + max_rollout))

    def get_metric_name(self, rollout_step: int = 0, is_da: bool = False, **_kwargs) -> str:
        """Get the metric name suffix for the current step."""
        if is_da:
            return f"_dacycle{rollout_step}"
        return f"_rstep{rollout_step - self.da_cycles}"

    def build_decoder_forcings(
        self,
        batch: dict[str, torch.Tensor],
        data_indices: dict[str, IndexCollection],
        **step_kwargs,
    ) -> dict[str, torch.Tensor] | None:
        """Extract decoder-forcing tensors from the (preprocessed) batch for one step.

        Returns per-dataset tensors of shape
        ``(batch, n_step_output, ensemble, grid, num_decoder_forcing_channels)``
        sliced at the current step's target times, ready to be passed to the model
        as ``decoder_forcings=...``. Returns ``None`` when no dataset declares any
        ``decoder_forcing`` variables (the common case), so the forward call is
        unaffected.

        Parameters
        ----------
        batch : dict[str, torch.Tensor]
            Full preprocessed batch keyed by dataset name.
        data_indices : dict[str, IndexCollection]
            Data indices per dataset.
        **step_kwargs
            Forwarded to ``get_batch_output_indices`` (e.g. ``rollout_step``).

        Returns
        -------
        dict[str, torch.Tensor] | None
            Decoder-forcing tensors per dataset, or ``None`` if unused.
        """
        step_kwargs.pop("is_da", None)
        output_batch_indices = self.get_batch_output_indices(**step_kwargs)

        decoder_forcings = {}
        any_present = False
        for dataset_name, dataset_batch in batch.items():
            df_idx = data_indices[dataset_name].data.input.decoder_forcing
            if len(df_idx) == 0:
                continue
            any_present = True
            df_idx = df_idx.to(device=dataset_batch.device)
            df = dataset_batch[:, output_batch_indices][..., df_idx]
            decoder_forcings[dataset_name] = df

        return decoder_forcings if any_present else None

    def build_skip_input(
        self,
        x: dict[str, torch.Tensor],
        y_pred: dict[str, torch.Tensor] | None = None,
        data_indices: dict[str, IndexCollection] | None = None,
    ) -> dict[str, torch.Tensor] | None:
        """Build the residual base for a DA-fed step: the pure model background.

        Passed to the model as ``skip_input=``, this becomes the base of the additive
        skip connection while the encoder still sees the observations. The skip
        therefore stays smooth and flow-dependent, and observations reach the output
        only through the encoder rather than through an additive identity path.

        With ``y_pred=None`` (step 0, where no background exists yet) every position is
        zero, standing for "nothing known here", so a
        ``ClimatologySkipConnection(missing_value=0.0, fill_missing_only=True)`` returns
        pure climatology instead of a climatology/observation mosaic.

        That pairing is a **config requirement, not a guarantee**: it holds when the
        residual's ``missing_value`` is ``0.0`` and the variables are mean-std
        normalized, which is what ``InputOnlyImputer(default: 0.0)`` already assumes
        when it marks absent observations. It does not hold for an imputer filling a
        literal raw-space ``0.0`` (which normalizes to ``-mean/stdev``), nor under
        ``min-max``/``none`` normalizers. With a plain ``SkipConnection`` the zero base
        simply disables the residual for step 0.

        Otherwise ``x`` is the state produced by :meth:`_advance_dataset_input_da`,
        whose prognostic positions carry the observation where present; this returns
        the same state with every prognostic position in the freshly written steps
        reset to the raw model background, undoing the observation copy.

        Parameters
        ----------
        x : dict[str, torch.Tensor]
            Model input state per dataset, as fed to the encoder.
        y_pred : dict[str, torch.Tensor] | None
            Raw model output per dataset from the step that produced ``x``, or
            ``None`` at step 0.
        data_indices : dict[str, IndexCollection] | None
            Data indices per dataset. Required unless ``y_pred`` is ``None``.

        Returns
        -------
        dict[str, torch.Tensor] | None
            Skip-connection input per dataset, or ``None`` when
            ``da_flow_dependent_skip`` is off or there are no DA cycles, leaving the
            forward call unchanged.
        """
        # Without DA cycles every step is a plain forecast whose input is already a
        # pure prediction, so overriding the base there would only discard it.
        if not self.da_flow_dependent_skip or self.da_cycles == 0:
            return None

        if y_pred is None:
            # Zeroes every channel, not just the prognostic ones. That is safe for the
            # residuals used here -- all are elementwise or per-channel across the grid,
            # and _assemble_output consumes only the prognostic columns of x_skip.
            return {dataset_name: torch.zeros_like(x_ds) for dataset_name, x_ds in x.items()}

        keep_steps = min(self.num_input_steps, self.num_output_steps)

        skip_input = {}
        for dataset_name, x_ds in x.items():
            prog_in = data_indices[dataset_name].model.input.prognostic
            prog_out = data_indices[dataset_name].model.output.prognostic
            background = x_ds.clone()
            for i in range(keep_steps):
                t = self.num_output_steps - keep_steps + i
                background[:, -(keep_steps - i), ..., prog_in] = y_pred[dataset_name][:, t][..., prog_out]
            skip_input[dataset_name] = background
        return skip_input

    def get_offsets(self, mode: str | None = None) -> list[datetime.timedelta]:
        """Return the offsets covering the DA cycles plus the forecast rollout.

        The dataloader loads exactly these time steps, so extending the rollout
        span by ``da_cycles`` sizes the batch to ``n_in + (da_cycles + rollout)
        * n_out`` frames.
        """
        if mode == "training":
            rollout_step = self.rollout.step
        elif mode == "validation":
            rollout_step = self.rollout.step if self.validation_rollout is None else self.validation_rollout
        else:
            validation_rollout = self.rollout.maximum if self.validation_rollout is None else self.validation_rollout
            rollout_step = max(self.rollout.maximum, validation_rollout)

        return self._compute_rollout_offsets(rollout_step + self.da_cycles)

    def advance_input(
        self,
        x: dict[str, torch.Tensor],
        y_pred: dict[str, torch.Tensor],
        batch: dict[str, torch.Tensor],
        rollout_step: int = 0,
        is_da: bool = False,
        data_indices: dict[str, IndexCollection] | None = None,
        output_mask: dict[str, object] | None = None,
        grid_shard_slice: dict[str, slice | None] | None = None,
    ) -> dict[str, torch.Tensor]:
        """Advance the input state for the next step.

        DA cycles blend observations into the state; forecast steps autoregress
        using the parent logic (with corrector input slots zeroed).
        """
        if is_da:
            for dataset_name in x:
                x[dataset_name] = self._advance_dataset_input_da(
                    x[dataset_name],
                    y_pred[dataset_name],
                    batch[dataset_name],
                    rollout_step=rollout_step,
                    data_indices=data_indices[dataset_name],
                )
            return x

        return super().advance_input(
            x,
            y_pred,
            batch,
            rollout_step=rollout_step,
            data_indices=data_indices,
            output_mask=output_mask,
            grid_shard_slice=grid_shard_slice,
        )

    def _advance_dataset_input_da(
        self,
        x: torch.Tensor,
        y_pred: torch.Tensor,
        batch: torch.Tensor,
        rollout_step: int = 0,
        data_indices: IndexCollection | None = None,
    ) -> torch.Tensor:
        """Blend observations into a single dataset's input state for the next DA cycle.

        For each kept output step, start from the observation tensor (which
        carries forcing and corrector variables) and replace prognostic
        positions with the raw model background where the observation is NaN.
        The corrector MLP is intentionally NOT applied here, keeping the blend
        consistent with inference.
        """
        keep_steps = min(self.num_input_steps, self.num_output_steps)
        output_batch_indices = self.get_batch_output_indices(rollout_step=rollout_step)

        prog_in = data_indices.model.input.prognostic
        prog_out = data_indices.model.output.prognostic
        input_full = data_indices.data.input.full

        blended = []
        for t in range(self.num_output_steps):
            obs = batch[:, output_batch_indices[t], ..., input_full]  # (bs, [members], grid, var_in)
            pred_t = y_pred[:, t]  # (bs, members, grid, var_out)
            if obs.shape[1] != pred_t.shape[1]:
                obs = obs.expand(-1, pred_t.shape[1], -1, -1)
            b = obs.clone()
            b[..., prog_in] = torch.where(
                torch.isnan(obs[..., prog_in]),
                pred_t[..., prog_out],
                obs[..., prog_in],
            )
            blended.append(b)

        x = x.roll(-keep_steps, dims=1)
        for i in range(keep_steps):
            t = self.num_output_steps - keep_steps + i
            x[:, -(keep_steps - i), ...] = blended[t]
        return x

    def _advance_dataset_input(
        self,
        x: torch.Tensor,
        y_pred: torch.Tensor,
        batch: torch.Tensor,
        rollout_step: int = 0,
        data_indices: IndexCollection | None = None,
        output_mask: object | None = None,
        grid_shard_slice: slice | None = None,
    ) -> torch.Tensor:
        """Advance a forecast-rollout step, zeroing corrector input slots.

        During the forecast phase no observations are available, so the
        corrector variables (satellite geometry, report type, etc.) are set to
        zero in the freshly-advanced input steps.
        """
        x = super()._advance_dataset_input(
            x,
            y_pred,
            batch,
            rollout_step=rollout_step,
            data_indices=data_indices,
            output_mask=output_mask,
            grid_shard_slice=grid_shard_slice,
        )
        corrector_idx = data_indices.model.input.corrector
        if len(corrector_idx) > 0:
            keep_steps = min(self.num_input_steps, self.num_output_steps)
            for i in range(keep_steps):
                x[:, -(i + 1), ..., corrector_idx] = 0.0
        return x
