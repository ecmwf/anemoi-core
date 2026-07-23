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

from .timeless import BaseTimelessTask

LOGGER = logging.getLogger(__name__)


class SpatialDownscaler(BaseTimelessTask):
    """Spatial downscaling task implementation.

    Distinguishes input-only datasets (e.g. ``in_lres``, ``in_hres``) from
    output-only datasets (e.g. ``out_hres``) by explicit name lists rather than
    by time position (all snapshots share the same single timestep, t=0).

    ``normalize_batch`` is ``False`` because ``ResidualPredictionMode`` needs raw
    (unnormalized) tensors to compute ``y - interp(x_lres)`` in data space before
    applying residual-specific normalization statistics.
    """

    name: str = "spatial_downscaler"

    def __init__(self, input_datasets: list[str], target_datasets: list[str], **_kwargs) -> None:
        super().__init__()
        self.input_datasets = input_datasets
        self.target_datasets = target_datasets
        # No-op placeholder; a proper adapter will be added with downscaling diagnostics.
        self._plot_adapter = SpatialDownscalerPlotAdapter(self)

    @property
    def normalize_batch(self) -> bool:
        """Skip batch-wide normalization.

        ``ResidualPredictionMode`` needs raw (unnormalized) tensors so it can
        compute ``y - interp(x_lres)`` in data space before applying residual
        normalization statistics.
        """
        return False

    def get_inputs(
        self,
        batch: dict[str, torch.Tensor],
        data_indices: dict[str, IndexCollection],
        **_kwargs,
    ) -> dict[str, torch.Tensor]:
        """Extract model inputs from a batch, restricted to ``input_datasets``.

        Unlike the forecaster, the split between inputs and targets is by dataset
        name, not by time position.  All input datasets share the same single
        timestep (t=0, from ``BaseTimelessTask``).

        Parameters
        ----------
        batch : dict[str, torch.Tensor]
            Full batch keyed by dataset name, shape ``(bs, 1, ensemble, grid, nvar)``.
        data_indices : dict[str, IndexCollection]
            Data indices per dataset.

        Returns
        -------
        dict[str, torch.Tensor]
            Input tensors for ``input_datasets`` only, variable-filtered to
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
            Full batch keyed by dataset name, shape ``(bs, 1, ensemble, grid, nvar)``.

        Returns
        -------
        dict[str, torch.Tensor]
            Target tensors for ``target_datasets`` only (all variables),
            shape ``(bs, 1, ensemble, grid, nvar)``.
        """
        time_indices = normalize_time_indices(self.get_batch_output_indices())
        y = {}
        for name in self.target_datasets:
            if name not in batch:
                # At inference time targets are not provided — this is expected.
                continue
            y[name] = batch[name][:, time_indices]
            LOGGER.debug("SHAPE: y[%s].shape = %s", name, list(y[name].shape))
        return y
