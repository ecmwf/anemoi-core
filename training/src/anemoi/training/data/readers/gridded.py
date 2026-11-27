# (C) Copyright 2024-2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import logging
from collections.abc import Callable
from functools import cached_property
from functools import wraps

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


def cache_data(get_data: Callable[..., torch.Tensor]) -> Callable[..., torch.Tensor]:
    """Read the records of a sample from the dataset cache, and read only the other records from the dataset.

    A record is one position of one sequence, for the grid shard of this reader. Without a cache
    (see :meth:`GriddedDataReader.set_cache`), the wrapper calls ``get_data`` directly.
    """

    @wraps(get_data)
    def wrapper(self: "GriddedDataReader", sequence: int, positions: TimeIndices) -> torch.Tensor:
        if self._cache is None:
            return get_data(self, sequence, positions)

        positions = self.position_list(sequence, positions)
        grid_indices = self.grid_shard_slice
        records, missing = self._cache.check_cache(self._cache_dataset_id, sequence, positions, grid_indices)
        if missing:
            source_positions = [positions[index] for index in missing]
            source = get_data(self, sequence, source_positions)
            # The records keep the order of the dimensions of the zarr store. Then the values of
            # one variable are next to each other, and the blosc bitshuffle compresses them better.
            source = rearrange(
                source.numpy(),
                "dates ensemble gridpoints variables -> dates variables ensemble gridpoints",
            )
            self._cache.store_records(self._cache_dataset_id, sequence, source_positions, source, grid_indices)
            for index, value in zip(missing, source, strict=True):
                records[index] = value

        values = rearrange(
            np.stack(records),
            "dates variables ensemble gridpoints -> dates ensemble gridpoints variables",
        )
        return torch.from_numpy(values)

    return wrapper


class GriddedDataReader(BaseAnemoiReader):
    """Gridded dataset reader with static grid."""

    data_type = "gridded"

    #: Balanced grid partition sizes and this rank's slice of it (set by :meth:`set_reader_group_info`),
    #: or ``None`` when a single reader reads the whole grid.
    grid_shard_sizes: list[int] | None = None
    grid_shard_slice: slice | None = None

    #: The dataset cache (see :meth:`set_cache`). If None, all records are read from the dataset.
    _cache = None
    _cache_dataset_id: str | None = None

    def set_cache(self, cache: object, dataset_id: str) -> None:
        """Send the data reads through the dataset cache, as the records of ``dataset_id``."""
        self._cache = cache
        self._cache_dataset_id = dataset_id

    @property
    def num_sequences(self) -> int:
        """Return the number of sequences. An analysis dataset has one sequence."""
        return 1

    def sequence_length(self, sequence: int = 0) -> int:  # noqa: ARG002
        """Return the number of positions (dates) in a sequence."""
        return len(self.dates)

    def position_list(self, sequence: int, positions: TimeIndices) -> list[int]:
        """Return the positions as a list of integers."""
        if isinstance(positions, slice):
            return list(range(*positions.indices(self.sequence_length(sequence))))
        return np.atleast_1d(positions).tolist()

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
        return np.asarray(self.data.latitudes, dtype=np.float64)

    @cached_property
    def longitudes(self) -> np.ndarray:
        """Return per-grid-point longitudes in **degrees**, as stored by ``anemoi.datasets``."""
        return np.asarray(self.data.longitudes, dtype=np.float64)

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

    @cache_data
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
            values=data,
            variables=self.variables,
            layout=self.layout,
            statistics=self.statistics,
            grid_size=self.grid_size,
            latitudes=latitudes,
            longitudes=longitudes,
            shard_sizes=self.grid_shard_sizes,
        )
