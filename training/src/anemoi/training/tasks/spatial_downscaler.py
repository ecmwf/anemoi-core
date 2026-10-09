# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import logging

import torch

from anemoi.models.data_indices.collection import IndexCollection
from anemoi.training.diagnostics.callbacks.plot_adapter import SpatialDownscalerPlotAdapter
from anemoi.training.utils.time_indices import normalize_time_indices
from anemoi.utils.dates import as_timedelta
from anemoi.utils.dates import frequency_to_string

from .base import BaseSingleStepTask

LOGGER = logging.getLogger(__name__)


class SpatialDownscaler(BaseSingleStepTask):
    """Spatial downscaling task implementation.

    Distinguishes input-only datasets (e.g. ``in_lres``, ``in_hres``) from
    output-only datasets (e.g. ``out_hres``) by explicit name lists.
    """

    name: str = "spatial_downscaler"

    def __init__(
        self,
        input_datasets: list[str],
        target_datasets: list[str],
        input_offsets: list[str] | None = None,
        output_offsets: list[str] | None = None,
        **_kwargs,
    ) -> None:
        super().__init__(
            input_offsets=[as_timedelta(o) for o in (input_offsets or ["0H"])],
            output_offsets=[as_timedelta(o) for o in (output_offsets or ["0H"])],
        )
        self.input_datasets = input_datasets
        self.target_datasets = target_datasets
        # Rejects plot callbacks until downscaling diagnostics exist.
        self._plot_adapter = SpatialDownscalerPlotAdapter(self)

    def validate_dataset_roles(self, input_datasets: list[str], target_datasets: list[str]) -> None:
        """Require the task's input and target datasets to be exactly the model's."""
        errors = [
            f"task.{role} {sorted(declared)} != {sorted(model_datasets)} in the model ({source})"
            for role, declared, model_datasets, source in (
                ("input_datasets", self.input_datasets, input_datasets, "model.encoders.*.source_datasets"),
                ("target_datasets", self.target_datasets, target_datasets, "model.decoders.*.target_datasets"),
            )
            if set(declared) != set(model_datasets)
        ]
        if errors:
            msg = "Dataset roles of the task and the model disagree: " + "; ".join(errors) + "."
            raise ValueError(msg)

    def _get_timestep_for_metadata(self) -> str:
        """Get the timestep string for metadata.

        Zero, as for any timeless task: the outputs are valid at the same times
        as the inputs. See :meth:`fill_metadata` for the consequences.
        """
        return "0H"

    def fill_metadata(self, md_dict: dict) -> None:
        """Record the offsets explicitly, and that downscaling has no feedback.

        Inference derives ``lagged``, ``output_offsets`` and ``advance_map``
        from ``timestep`` unless the metadata states them. With a timeless
        ``timestep`` of ``0H`` every derived offset collapses to zero, so the
        inputs of a multi-snapshot window could not be retrieved.

        No ``rollout_shift`` is written. For a forecaster it is how far the
        state advances per model call, a property of the model; downscaling
        has no feedback, so it degenerates into how far to jump to the next
        independent window. Training does not determine that — it draws
        overlapping windows at the dataset frequency — so it belongs in the
        inference configuration.
        """
        super().fill_metadata(md_dict)

        downscaling_timesteps = {
            "input_offsets": [frequency_to_string(offset) for offset in self._input_offsets],
            "output_offsets": [frequency_to_string(offset) for offset in self._output_offsets],
            # Outputs are never fed back into the inputs.
            "advance_map": {"inin": [], "outin": []},
        }
        for dataset_name in md_dict["metadata_inference"]["dataset_names"]:
            md_dict["metadata_inference"][dataset_name]["timesteps"].update(downscaling_timesteps)

    def get_inputs(
        self,
        batch: dict[str, torch.Tensor],
        data_indices: dict[str, IndexCollection],
        **_kwargs,
    ) -> dict[str, torch.Tensor]:
        """Extract model inputs from a batch, restricted to ``input_datasets``.

        Unlike the forecaster, the split between inputs and targets is by dataset name;
        the time slots are selected by ``input_offsets``.

        Parameters
        ----------
        batch : dict[str, torch.Tensor]
            Full batch keyed by dataset name,
            shape ``(bs, num_offsets, ensemble, grid, nvar)``.
        data_indices : dict[str, IndexCollection]
            Data indices per dataset.

        Returns
        -------
        dict[str, torch.Tensor]
            Input tensors for ``input_datasets`` only, variable-filtered to
            ``data.input.full``,
            shape ``(bs, num_input_offsets, ensemble, grid, n_input_vars)``.
        """
        time_indices = normalize_time_indices(self.get_batch_input_indices())
        x = {}
        for name in self.input_datasets:
            if name not in batch:
                msg = f"Input dataset '{name}' not found in batch."
                raise ValueError(msg)
            ds = batch[name][:, time_indices]
            x[name] = ds[..., data_indices[name].data.input.full]
            LOGGER.debug("SHAPE: x[%s].shape = %s", name, list(x[name].shape))
        return x

    def get_targets(
        self,
        batch: dict[str, torch.Tensor],
        **_kwargs,
    ) -> dict[str, torch.Tensor]:
        """Extract model targets from a batch, restricted to ``target_datasets``.

        Returns full variable slices (no variable filtering); ``ResidualPredictionMode``
        applies variable selection internally.

        Parameters
        ----------
        batch : dict[str, torch.Tensor]
            shape ``(bs, num_offsets, ensemble, grid, nvar)``.

        Returns
        -------
        dict[str, torch.Tensor]
            Target tensors for ``target_datasets`` only (all variables),
            shape ``(bs, num_output_offsets, ensemble, grid, nvar)``.
        """
        time_indices = normalize_time_indices(self.get_batch_output_indices())
        y = {}
        for name in self.target_datasets:
            if name not in batch:
                msg = f"Target dataset '{name}' not found in batch."
                raise ValueError(msg)
            y[name] = batch[name][:, time_indices]
            LOGGER.debug("SHAPE: y[%s].shape = %s", name, list(y[name].shape))
        return y
