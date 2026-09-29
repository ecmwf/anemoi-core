# (C) Copyright 2025-2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import logging
from typing import TYPE_CHECKING

from torch import Tensor
from torch import nn

from anemoi.models.distributed.graph import shard_tensor
from anemoi.models.distributed.shapes import ShardSizes

if TYPE_CHECKING:
    from anemoi.models.data.sources import GriddedSource

LOGGER = logging.getLogger(__name__)


class SpatialPreprocessor(nn.Module):
    """Base class for preprocessors that operate across the spatial (grid) dimension.

    Unlike ``BasePreprocessor`` which applies variable-wise arithmetic on a fixed grid,
    ``SpatialPreprocessor`` subclasses may change the grid dimension — for example
    projecting data from a low-resolution grid onto a high-resolution grid.

    Subclasses must implement ``forward`` and expose their input and output grid
    sizes. The ``inverse`` method raises ``NotImplementedError`` by default
    because spatial projections are generally not invertible.

    Spatial preprocessors are registered on ``AnemoiModelInterface`` as
    ``self.spatial_pre_processors`` (a ``nn.ModuleDict`` keyed by dataset name)
    and are included when the complete model is serialized for inference.
    """

    @property
    def input_grid_size(self) -> int:
        """Number of spatial points expected by the preprocessor."""
        raise NotImplementedError

    @property
    def output_grid_size(self) -> int:
        """Number of spatial points produced by the preprocessor."""
        raise NotImplementedError

    def forward(
        self,
        x: Tensor,
        model_comm_group=None,
        grid_shard_sizes: ShardSizes = None,
    ) -> tuple[Tensor, ShardSizes]:
        """Project input to a (potentially different) grid.

        Parameters
        ----------
        x : Tensor
            Input tensor, shape ``(batch, time, ensemble, grid_src, vars)``.
        model_comm_group : ProcessGroup, optional
            Process group used for distributed projection.
        grid_shard_sizes : ShardSizes, optional
            Source-grid shard size for each rank, or ``None`` for replicated input.

        Returns
        -------
        tuple[Tensor, ShardSizes]
            Output tensor with shape ``(batch, time, ensemble, grid_dst, vars)``
            and the target-grid shard size for each rank. The shard sizes are
            ``None`` when the output is replicated.
        """
        raise NotImplementedError

    def inverse(self, x: Tensor) -> Tensor:
        raise NotImplementedError(f"{self.__class__.__name__} does not support inverse projection.")

    def project_source(
        self,
        source: "GriddedSource",
        target_coordinates: Tensor,
        model_comm_group=None,
    ) -> "GriddedSource":
        """Project a gridded source onto the target grid.

        Parameters
        ----------
        source : GriddedSource
            Source on the projector's input grid, with data ``(batch, time, ensemble, grid_src, vars)``.
            It may be grid-sharded, as described by its ``shard_sizes``.
        target_coordinates : Tensor
            ``(grid_dst, 2)`` coordinates of the full target grid, in the source's coordinate convention.
        model_comm_group : ProcessGroup, optional
            Process group used for distributed projection.

        Returns
        -------
        GriddedSource
            The source on the target grid: projected data, target coordinates and target-grid
            shard sizes (coordinates are sharded like the data). Variables and statistics are unchanged.
        """
        if source.layout.time_in_grid:
            raise TypeError(f"{self.__class__.__name__} only projects gridded sources, got {source.name!r}.")
        pattern = source.layout.normalized(source.data.ndim).pattern
        if pattern != "batch time ensemble grid variables":
            raise ValueError(f"{self.__class__.__name__} expects a (batch, time, ensemble, grid, variables) layout.")

        data, shard_sizes = self(source.data, model_comm_group=model_comm_group, grid_shard_sizes=source.shard_sizes)
        coordinates = target_coordinates.to(device=data.device, dtype=source.coordinates.dtype)
        if shard_sizes is not None:
            coordinates = shard_tensor(coordinates, -2, shard_sizes, model_comm_group, gather_in_backward=False)
        return source.clone(data=data, coordinates=coordinates, shard_sizes=shard_sizes)
