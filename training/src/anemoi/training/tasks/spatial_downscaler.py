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

from .base import BaseSingleStepTask

LOGGER = logging.getLogger(__name__)


class SpatialDownscaler(BaseSingleStepTask):
    """Spatial downscaling task implementation.

    Distinguishes input-only datasets (e.g. ``in_lres``, ``in_hres``) from
    output-only datasets (e.g. ``out_hres``) by explicit name lists rather than
    by time position.

    Supports multiple simultaneous snapshots via ``offsets``.  Each offset
    represents one time slot; the same list is used for both input and output
    so that snapshot *i* of each input dataset corresponds to snapshot *i* of
    each target dataset.  The default (``offsets=None``) is equivalent to
    ``offsets=["0H"]`` for single-snapshot behaviour.

    The batch is normalized in the standard way; ``ResidualPredictionMode``
    denormalizes lres and target internally to compute ``y - interp(x_lres)`` in
    physical space, then renormalizes the residual with tendency-space statistics.
    """

    name: str = "spatial_downscaler"

    def __init__(
        self,
        input_datasets: list[str],
        target_datasets: list[str],
        offsets: list[str] | None = None,
        **_kwargs,
    ) -> None:
        shared_offsets = [as_timedelta(o) for o in (offsets or ["0H"])]
        super().__init__(input_offsets=shared_offsets, output_offsets=shared_offsets)
        self.input_datasets = input_datasets
        self.target_datasets = target_datasets
        # No-op placeholder; a proper adapter will be added with downscaling diagnostics.
        self._plot_adapter = SpatialDownscalerPlotAdapter(self)

    def _get_timestep_for_metadata(self) -> str:
        """Get the timestep string for metadata."""
        return "0H"

    def get_inputs(
        self,
        batch: dict[str, torch.Tensor],
        data_indices: dict[str, IndexCollection],
        **_kwargs,
    ) -> dict[str, torch.Tensor]:
        """Extract model inputs from a batch, restricted to ``input_datasets``.

        Unlike the forecaster, the split between inputs and targets is bet of
        time offsets (configured via ``offsets``).

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
            shape ``(bs, num_offsets` only, variable-filtered to
            ``data.input.full``, shape ``(bs, 1, ensemble, grid, n_input_vars)``.
        """
        time_indices = normalize_time_indices(self.get_batch_input_indices())
        x = {}
        for name in self.input_datasets:
            if name not in batch:
                LOGGER.warning("Input dataset '%s' not found in batch — skipping.", name)
                continue
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
            shape ``(bs, num_offsets for ``target_datasets`` only (all variables),
            shape ``(bs, 1, ensemble, grid, nvar)``.
        """
        time_indices = normalize_time_indices(self.get_batch_output_indices())
        y = {}
        for name in self.target_datasets:
            if name not in batch:
                LOGGER.warning("Target dataset '%s' not found in batch — skipping.", name)
                continue
            y[name] = batch[name][:, time_indices]
            LOGGER.debug("SHAPE: y[%s].shape = %s", name, list(y[name].shape))
        return y
