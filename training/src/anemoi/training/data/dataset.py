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
from functools import cached_property

import numpy as np
import torch
from einops import rearrange
from omegaconf import DictConfig
from rich.console import Console
from rich.tree import Tree

from anemoi.datasets import open_dataset
from anemoi.utils.dates import frequency_to_seconds

LOGGER = logging.getLogger(__name__)


def _as_dict(value: str | dict | DictConfig) -> str | dict:
    """Convert DictConfig payloads to plain dicts."""
    return dict(value) if isinstance(value, DictConfig) else value


def _normalize_dataset_config(dataset_config: str | dict | DictConfig) -> str | dict:
    """Normalize dataset payload to the open_dataset dictionary contract."""
    dataset_config = _as_dict(dataset_config)
    if not isinstance(dataset_config, dict):
        return dataset_config

    if "dataset" not in dataset_config:
        msg = "dataset_config must contain the 'dataset' key."
        raise ValueError(msg)

    if dataset_config["dataset"] is None:
        msg = "dataset_config.dataset cannot be None."
        raise ValueError(msg)

    invalid_inner_keys = {"start", "end"} & set(dataset_config)
    if invalid_inner_keys:
        invalid = ", ".join(sorted(invalid_inner_keys))
        msg = f"dataset_config cannot contain [{invalid}]. Use outer keys 'start' and 'end' instead."
        raise ValueError(msg)

    # Keep only explicitly set options to avoid passing None-valued kwargs
    # (e.g. select=None), which can trigger downstream subset selection issues.
    return {key: value for key, value in dataset_config.items() if value is not None}


def _normalize_reader_config(dataset_config: dict | DictConfig) -> dict:
    """Validate and normalize reader configuration."""
    normalized = dict(dataset_config)

    if "dataset" in normalized:
        msg = (
            "Invalid dataloader dataset schema: use 'dataset_config' (outer key) "
            "and 'dataset' inside it. The legacy outer 'dataset' key is no longer supported."
        )
        raise ValueError(msg)

    base_dataset_config = normalized.pop("dataset_config", None)
    if base_dataset_config is None:
        msg = "Missing required 'dataset_config' in dataset reader configuration."
        raise ValueError(msg)

    normalized["dataset_config"] = base_dataset_config
    return normalized


class BaseAnemoiReader:
    """Anemoi data reader for native grid datasets."""

    def __init__(
        self,
        dataset: str | dict | None = None,
        dataset_config: str | dict | None = None,
        start: datetime.datetime | int | None = None,
        end: datetime.datetime | int | None = None,
    ):
        """Initialize Anemoi data reader."""
        source = dataset_config if dataset_config is not None else dataset
        if source is None:
            msg = "Either dataset or dataset_config must be provided."
            raise ValueError(msg)
        self.data = open_dataset(_normalize_dataset_config(source), start=start, end=end)

    @property
    def dates(self) -> np.ndarray:
        """Return dataset dates."""
        return self.data.dates

    @property
    def grid_size(self) -> int:
        """Return dataset grid size."""
        return sum(self.data.grids)

    @property
    def statistics(self) -> dict:
        """Return dataset statistics."""
        return self.data.statistics

    def statistics_tendencies(
        self,
        timestep: int | str | datetime.timedelta | None = None,
    ) -> dict | None:
        """Return dataset tendency statistics."""
        if timestep is None:
            timestep = getattr(self, "timestep", None)
        if timestep is None:
            msg = "timestep must be provided to compute tendency statistics."
            raise ValueError(msg)
        try:
            return self.data.statistics_tendencies(timestep)
        except (KeyError, AttributeError):
            return None

    @property
    def variables(self) -> list[str]:
        """Return dataset variables."""
        return self.data.variables

    @property
    def missing(self) -> set[int]:
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

    @property
    @abstractmethod
    def has_trajectories(self) -> bool:
        """Return whether the dataset has trajectories."""

    def get_sample(
        self,
        time_indices: slice | int | list[int],
        grid_shard_indices: np.ndarray | slice | None = None,
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
        trajectory_start: datetime.datetime,
        trajectory_length: int,
        dataset: str | dict | None = None,
        dataset_config: str | dict | None = None,
        start: datetime.datetime | int | None = None,
        end: datetime.datetime | int | None = None,
    ):
        super().__init__(dataset=dataset, dataset_config=dataset_config, start=start, end=end)
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
    dataset_config = _normalize_reader_config(dataset_config)
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
