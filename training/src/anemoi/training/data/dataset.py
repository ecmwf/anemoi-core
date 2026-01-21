# (C) Copyright 2024- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import datetime
import logging
from abc import abstractmethod

import torch
from einops import rearrange
from rich.console import Console
from rich.tree import Tree

from anemoi.datasets import open_dataset
from anemoi.utils.dates import frequency_to_seconds

LOGGER = logging.getLogger(__name__)


class BaseAnemoiReader:
    """Anemoi data reader for native grid datasets."""

    def __init__(
        self,
        dataset: str | dict,
        start: datetime.datetime | int | None = None,
        end: datetime.datetime | int | None = None,
        frequency: str | None = None,
        drop: list[str] | None = None,
    ):
        self.data = open_dataset(dataset, start=start, end=end, frequency=frequency, drop=drop)

    @property
    def dates(self) -> list[datetime.datetime]:
        """Return dataset dates."""
        return self.data.dates

    @property
    def statistics(self) -> dict:
        """Return dataset statistics."""
        return self.data.statistics

    @property
    def statistics_tendencies(self) -> dict | None:
        """Return dataset tendency statistics."""
        try:
            return self.data.statistics_tendencies(self.timestep)
        except (KeyError, AttributeError):
            return None

    @property
    def variables(self) -> list[str]:
        """Return dataset variables."""
        return self.data.variables

    @property
    def missing(self) -> np.ndarray:
        """Return dataset missing values mask."""
        return self.data.missing

    @property
    def metadata(self) -> dict:
        """Return dataset metadata."""
        return self.data.metadata()

    @property
    def frequency(self) -> datetime.timedelta:
        """Return dataset frequency."""
        return self.data.frequency

    @property
    def supporting_arrays(self) -> dict:
        """Return dataset supporting_arrays."""
        return self.data.supporting_arrays()

    @property
    def name_to_index(self) -> dict[str, int]:
        """Return dataset statistics."""
        return self.data.name_to_index

    @property
    def resolution(self) -> str:
        """Return dataset resolution."""
        return self.data.resolution

    @property
    @abstractmethod
    def has_trajectories(self) -> bool:
        """Return whether the dataset has trajectories."""

    def get_sample(
        self,
        time_indices: slice | int | list[int],
        grid_shard_indices: np.ndarray | None = None,
    ) -> torch.Tensor:
        """Get a sample from the dataset."""
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
        tree.add(f"Frequency: {self.frequency}")
        tree.add(f"Resolution: {self.resolution}")
        tree.add(f"Num variables: {len(self.name_to_index)}")
        return tree


class NativeGridDataset(BaseAnemoiReader):
    """Native grid dataset."""

    @property
    def has_trajectories(self) -> bool:
        """Return whether the dataset has trajectories."""
        return False


class TrajectoryDataset(BaseAnemoiReader):
    """Trajectory dataset."""

    def __init__(
        self,
        dataset: str | dict,
        trajectory_start: datetime.datetime,
        trajectory_length: int,
        start: datetime.datetime | int | None = None,
        end: datetime.datetime | int | None = None,
        frequency: str | None = None,
        drop: list[str] | None = None,
    ):
        super().__init__(dataset, start=start, end=end, frequency=frequency, drop=drop)
        self.trajectory_start = trajectory_start
        self.trajectory_length = trajectory_length

    @property
    def has_trajectories(self) -> bool:
        """Return whether the dataset has trajectories."""
        return True

    @property
    def trajectory_ids(self) -> list[str]:
        trajectory_length_seconds = self.trajectory_length * frequency_to_seconds(self.frequency)
        return (self.dates - self.trajectory_start) // np.timedelta64(trajectory_length_seconds, "s")

    def tree(self, prefix: str = "") -> Tree:
        tree = super().tree(prefix)
        tree.add(f"Trajectory start: {self.trajectory_start}")
        tree.add(f"Trajectory length: {self.trajectory_length} steps")
        return tree


def create_dataset(dataset_config: dict) -> BaseAnemoiReader:
    """Factory function to create dataset based on dataset configuration."""
    trajectory_config = dataset_config.pop("trajectory", {})
    if trajectory_config is not None and hasattr(trajectory_config, "start") and hasattr(trajectory_config, "length"):
        LOGGER.info("Creating TrajectoryDataset...")
        return TrajectoryDataset(
            **dataset_config,
            trajectory_start=trajectory_config["start"],
            trajectory_length=trajectory_config["length"],
        )

    LOGGER.info("Creating NativeGridDataset...")
    return NativeGridDataset(**dataset_config)
