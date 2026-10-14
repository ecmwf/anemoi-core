# (C) Copyright 2025-2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import logging

from torch import Tensor
from torch import nn

from anemoi.models.distributed.shapes import ShardSizes

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
