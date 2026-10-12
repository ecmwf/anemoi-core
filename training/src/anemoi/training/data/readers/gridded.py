# (C) Copyright 2024-2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import logging
from functools import cached_property

import numpy as np
import torch
from einops import rearrange

from anemoi.models.data import TensorLayout
from anemoi.models.data.sample import GriddedSample
from anemoi.models.data.sample import create_batched_struct
from anemoi.models.distributed.balanced_partition import get_balanced_partition_sizes
from anemoi.models.distributed.balanced_partition import get_partition_range
from anemoi.training.data.readers.base import BaseAnemoiReader
from anemoi.training.utils.time_indices import TimeIndices

LOGGER = logging.getLogger(__name__)


class GriddedDataReader(BaseAnemoiReader):
    """Gridded dataset reader with static grid."""

    data_type = "gridded"

    #: Balanced grid partition sizes and this rank's slice of it (set by :meth:`set_reader_group_info`),
    #: or ``None`` when a single reader reads the whole grid.
    grid_shard_sizes: list[int] | None = None
    grid_shard_slice: slice | None = None

    @property
    def layout(self) -> TensorLayout:
        """Return the gridded per-sample layout."""
        return TensorLayout(time=0, ensemble=1, grid=2, variables=3)

    @property
    def grid_size(self) -> int:
        """Return dataset grid size."""
        return self.data.shape[-1]

    @property
    def supporting_arrays(self) -> dict:
        """Return dataset supporting_arrays."""
        return self.data.supporting_arrays()

    @cached_property
    def latitudes(self) -> np.ndarray:
        """Return per-grid-point latitudes in **degrees**, as stored by ``anemoi.datasets``."""
        return np.asarray(self.data.latitudes, dtype=np.float32)

    @cached_property
    def longitudes(self) -> np.ndarray:
        """Return per-grid-point longitudes in **degrees**, as stored by ``anemoi.datasets``."""
        return np.asarray(self.data.longitudes, dtype=np.float32)

    @cached_property
    def cutout_mask(self) -> np.ndarray:
        """Return cutout mask."""
        cutout_mask = np.zeros(self.grid_size, dtype=bool)
        if len(self.data.grids) <= 1:
            err_msg = "Dataset `cutout_mask` property requires a cutout grid but does not have one."
            raise ValueError(err_msg)
        cutout_mask[: self.data.grids[0]] = True
        return cutout_mask

    @cached_property
    def boundary_mask(self) -> np.ndarray:
        """Return boundary mask, defined as the complement of the cutout mask."""
        return ~self.cutout_mask

    def set_reader_group_info(self, reader_group_rank: int, reader_group_size: int) -> None:
        super().set_reader_group_info(reader_group_rank, reader_group_size)

        if reader_group_size <= 1:
            self.grid_shard_slice = None
            self.grid_shard_sizes = None
        else:
            self.grid_shard_sizes = get_balanced_partition_sizes(self.grid_size, self.reader_group_size)
            start, end = get_partition_range(self.grid_shard_sizes, self.reader_group_rank)
            self.grid_shard_slice = slice(start, end)

        LOGGER.info(
            "Gridded reader shard sizes: %s, assigned shard: %s",
            self.grid_shard_sizes,
            self.grid_shard_slice,
        )

    def get_data(
        self,
        sequence: int,
        positions: TimeIndices,
    ) -> torch.Tensor:
        """Return data tensor for the requested time/grid slice.

        Output shape: ``(dates, ensemble, gridpoints, variables)``.
        """
        del sequence
        if self.grid_shard_slice is not None:
            x = self.data[positions, :, :, self.grid_shard_slice]
        else:
            x = self.data[positions, :, :, :]

        x = rearrange(x, "dates variables ensemble gridpoints -> dates ensemble gridpoints variables")
        return torch.from_numpy(x)

    def get_latlons(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the local shard's ``(N,)`` latitudes and longitudes in **degrees**."""
        lats, lons = self.latitudes, self.longitudes

        if self.grid_shard_slice is not None:
            lats = lats[self.grid_shard_slice]
            lons = lons[self.grid_shard_slice]

        return torch.from_numpy(np.ascontiguousarray(lats)), torch.from_numpy(np.ascontiguousarray(lons))

    def get_sample(
        self,
        sequence: int,
        positions: TimeIndices,
    ) -> GriddedSample:
        """Return the per-sample payload in the unified contract."""
        latitudes, longitudes = self.get_latlons()
        data = self.get_data(sequence, positions)
        return create_batched_struct(
            data_type=self.data_type,
            data=data,
            variables=self.variables,
            layout=self.layout,
            statistics=self.statistics,
            grid_size=self.grid_size,
            latitudes=latitudes,
            longitudes=longitudes,
            shard_sizes=self.grid_shard_sizes,
        )
