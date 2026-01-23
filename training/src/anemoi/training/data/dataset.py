# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import datetime
import logging
from collections.abc import Callable
from functools import cached_property

import torch
from einops import rearrange
from rich.console import Console
from rich.tree import Tree

from anemoi.training.data.grid_indices import BaseGridIndices

LOGGER = logging.getLogger(__name__)


class NativeGridDataset:
    """Dataset for AnemoI data on the arbitrary grids."""

    def __init__(
        self,
        data_reader: Callable,
        grid_indices: type[BaseGridIndices],
        timestep: str = "6h",
    ) -> None:
        """Initialize (part of) the dataset state.

        Parameters
        ----------
        data_reader : Callable
            user function that opens and returns the anemoi-datasets array data
        grid_indices : Type[BaseGridIndices]
            indices of the grid to keep. Defaults to None, which keeps all spatial indices.
        timestep : int, optional
            the time frequency of the samples, by default '6h'
        label : str, optional
            label for the dataset, by default "generic"
        """
        self.data = data_reader
        self.timestep = timestep
        self.grid_indices = grid_indices

    @cached_property
    def statistics(self) -> dict:
        """Return dataset statistics."""
        return self.data.statistics

    @cached_property
    def statistics_tendencies(self) -> dict | None:
        """Return dataset tendency statistics."""
        return self.statistics_tendencies_for_timestep(self.timestep)

    def statistics_tendencies_for_timestep(self, timestep: str) -> dict | None:
        """Return dataset tendency statistics for a specific timestep."""
        try:
            return self.data.statistics_tendencies(timestep)
        except (KeyError, AttributeError, ValueError):
            return None

    @cached_property
    def variables(self) -> list[str]:
        """Return dataset variables."""
        return self.data.variables

    @cached_property
    def metadata(self) -> dict:
        """Return dataset metadata."""
        return self.data.metadata()

    @cached_property
    def frequency(self) -> datetime.timedelta:
        """Return dataset frequency."""
        return self.data.frequency

    @cached_property
    def supporting_arrays(self) -> dict:
        """Return dataset supporting_arrays."""
        return self.data.supporting_arrays()

    @cached_property
    def name_to_index(self) -> dict[str, int]:
        """Return dataset statistics."""
        return self.data.name_to_index

    @cached_property
    def resolution(self) -> str:
        """Return dataset resolution."""
        return self.data.resolution

    def get_sample(self, time_indices: int | list[int] | slice, reader_group_rank: int) -> torch.Tensor:
        # NOTE: this is temporary until anemoi datasets allows indexing with arrays or lists
        # data[start...] will be replaced with data[self.relative_date_indices + i]

        grid_shard_indices = self.grid_indices.get_shard_indices(reader_group_rank)
        if isinstance(grid_shard_indices, slice):
            # Load only shards into CPU memory
            x = self.data[time_indices, :, :, grid_shard_indices]

        else:
            # Load full grid in CPU memory, select grid_shard after
            # Note that anemoi-datasets currently doesn't support slicing + indexing
            # in the same operation.
            x = self.data[time_indices, :, :, :]
            x = x[..., grid_shard_indices]  # select the grid shard

        x = rearrange(x, "dates variables ensemble gridpoints -> dates ensemble gridpoints variables")

        return torch.from_numpy(x)

    def __repr__(self) -> str:
        console = Console(record=True, width=120)
        with console.capture() as capture:
            console.print(self.tree())
        return capture.get()

    def tree(self, prefix: str = "") -> Tree:
        tree = Tree(prefix + " 💾 " + f"{self.__class__.__name__}")
        tree.add(f"Dataset: {self.data}")
        tree.add(f"Timestep: {self.timestep}")
        tree.add(f"Resolution: {self.resolution}")
        tree.add(f"Num variables: {len(self.name_to_index)}")
        return tree
