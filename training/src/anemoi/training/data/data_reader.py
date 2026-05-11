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
from abc import ABC
from abc import abstractmethod
from functools import cached_property

import numpy as np
import torch
from einops import rearrange
from omegaconf import DictConfig
from rich.console import Console
from rich.tree import Tree

from anemoi.datasets import open_dataset
from anemoi.models.data.batch import TensorLayout
from anemoi.training.utils.time_indices import TimeIndices
from anemoi.utils.dates import frequency_to_seconds

LOGGER = logging.getLogger(__name__)



def _as_dict(value: str | dict | DictConfig) -> str | dict:
    """Convert DictConfig payloads to plain dicts."""
    return dict(value) if isinstance(value, DictConfig) else value


def _normalize_dataset_config(dataset_config: str | dict | DictConfig) -> dict:
    """Normalize dataset payload to the open_dataset dictionary contract."""
    dataset_config = _as_dict(dataset_config)
    if isinstance(dataset_config, str):
        return {"dataset": dataset_config}

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
    """Validate and normalize reader configuration.

    Arguments
    ---------
    dataset_config : dict or DictConfig
        Dataset configuration dictionary.

    Returns
    -------
    dict
        Normalized dataset configuration dictionary with the following contract:
        {
            "dataset_config": {
                "dataset": str,
                "window": int,  # optional, for tabular datasets
                "frequency": str,  # optional, for tabular datasets
                ... other open_dataset kwargs ...
            },
            "start": datetime | int | None,  # optional
            "end": datetime | int | None,  # optional
            "trajectory": {  # optional, for trajectory datasets
                "start": datetime,
                "length": int,
            }
        }
    """
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


class BaseAnemoiReader(ABC):
    """Generic anemoi data reader."""

    def __init__(
        self,
        dataset: str | dict | None = None,
        dataset_config: str | dict | None = None,
        start: datetime.datetime | int | None = None,
        end: datetime.datetime | int | None = None,
    ):
        """Initialize Anemoi data reader."""
        assert not (dataset and dataset_config), "Only one of dataset or dataset_config should be provided."
        assert dataset or dataset_config, "Either dataset or dataset_config must be provided."

        source: dict = _normalize_dataset_config(dataset_config or dataset)
        source |= {"start": start, "end": end}
        # start and end arguments have to be passed at the same level as the window and frequency arguments for
        # tabular datasets

        self.data = open_dataset(source)

    @property
    def dates(self) -> np.ndarray:
        """Return dataset dates."""
        return self.data.dates

    @property
    def grid_size(self) -> int:
        """Return dataset grid size."""
        return self.data.shape[0]

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
        except (KeyError, AttributeError, TypeError):
            return None

    @property
    @abstractmethod
    def is_static_grid(self) -> bool:
        """Whether the reader exposes a single, time-invariant grid.

        ``True`` for gridded datasets (one set of latitudes/longitudes shared
        by every sample); ``False`` for sparse observation datasets where the
        coordinates change at every sample. Used by
        :class:`~anemoi.training.data.multidataset.MultiDataset` to decide
        whether to share coordinate tensors by reference across the batch.
        """

    @property
    def is_tabular(self) -> bool:
        """Return whether the dataset is tabular (2D backing array).

        Kept for backward compatibility with consumers that branched on the
        legacy flag; new code should use :attr:`is_static_grid` instead.
        """
        return len(self.data.shape) == 2

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

    @abstractmethod
    def get_sample(
        self,
        time_indices: TimeIndices,
        grid_shard_indices: np.ndarray | slice | None = None,
    ) -> dict:
        """Return a single per-sample payload.

        Subclasses must return a dict with the unified contract:

        * ``"data"`` — :class:`torch.Tensor`. For gridded datasets the shape
          is ``(T, E, N, V)``; for sparse observation datasets the shape is
          ``(E=1, N, V)`` (no leading time axis; per-time structure lives in
          ``metadata["boundaries"]``).
        * ``"coordinates"`` — :class:`torch.Tensor` of shape ``(N, 2)`` where
          the trailing dimension stacks ``(latitude, longitude)`` in
          **radians**.
        * ``"timedeltas"`` — *(sparse only)* :class:`torch.Tensor` of shape
          ``(N,)`` carrying the per-point time offset in seconds. Omitted on
          gridded datasets, where the time axis is intrinsic to ``"data"``.
        * ``"metadata"`` — ``dict[str, Any]`` of non-tensor per-sample
          metadata (empty for gridded; carries ``"boundaries"`` for sparse).
        """
        raise NotImplementedError("Subclasses must implement get_sample() method.")

    def __repr__(self) -> str:
        console = Console(record=True, width=120)
        with console.capture() as capture:
            console.print(self.tree())
        return capture.get()

    def tree(self, prefix: str = "") -> Tree:
        tree = Tree(prefix + " 💾 " + f"{self.__class__.__name__}")
        tree.add(f"Dataset: {self.data}")
        tree.add(f"Frequency: {self.frequency}")
        tree.add(f"Num variables: {len(self.name_to_index)}")
        tree.add(f"Resolution: {self.resolution}")
        return tree


class GriddedDataReader(BaseAnemoiReader, ABC):
    """Gridded dataset reader with static grid."""

    @property
    def is_static_grid(self) -> bool:
        """Gridded readers expose a single, time-invariant grid."""
        return True

    @property
    def supporting_arrays(self) -> dict:
        """Return dataset supporting_arrays."""
        return self.data.supporting_arrays()

    @cached_property
    def latitudes(self) -> np.ndarray:
        """Return per-grid-point latitudes in **radians**.

        Backed by ``self.data.latitudes`` (which is stored in degrees by
        ``anemoi.datasets``); converted once and cached.
        """
        return np.deg2rad(np.asarray(self.data.latitudes))

    @cached_property
    def longitudes(self) -> np.ndarray:
        """Return per-grid-point longitudes in **radians**."""
        return np.deg2rad(np.asarray(self.data.longitudes))

    @property
    def has_trajectories(self) -> bool:
        """Return whether the dataset has trajectories."""
        return False

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

    def get_data(
        self,
        time_indices: TimeIndices,
        grid_shard_indices: np.ndarray | slice | None = None,
    ) -> torch.Tensor:
        """Return data tensor for the requested time/grid slice.

        Output shape: ``(dates, ensemble, gridpoints, variables)``.
        """
        if isinstance(grid_shard_indices, slice):
            # Load only the shard into CPU memory.
            x = self.data[time_indices, :, :, grid_shard_indices]
        else:
            # Load the full grid then select the shard, because anemoi-datasets
            # currently doesn't support slicing + fancy indexing in the same op.
            x = self.data[time_indices, ...]
            x = x[..., grid_shard_indices]

        x = rearrange(x, "dates variables ensemble gridpoints -> dates ensemble gridpoints variables")
        return torch.from_numpy(x)

    def get_coordinates(
        self,
        time_indices: TimeIndices | None = None,
        grid_shard_indices: np.ndarray | slice | None = None,
    ) -> torch.Tensor:
        """Return per-grid-point ``(latitude, longitude)`` coordinates.

        For a static grid the ``time_indices`` argument is ignored and the
        full grid is returned (sliced by ``grid_shard_indices`` when given).
        Subclasses with dynamic grids must override.

        Parameters
        ----------
        time_indices : TimeIndices, optional
            Time indices; ignored on static grids.
        grid_shard_indices : np.ndarray or slice, optional
            Per-rank grid shard.

        Returns
        -------
        torch.Tensor
            Tensor of shape ``(N, 2)`` stacking ``(latitudes, longitudes)``
            along the trailing dimension, in **radians**.
        """
        del time_indices  # unused for static grids
        lats = self.latitudes
        lons = self.longitudes
        if grid_shard_indices is not None:
            lats = lats[grid_shard_indices]
            lons = lons[grid_shard_indices]

        coords = np.stack(
            [np.ascontiguousarray(lats), np.ascontiguousarray(lons)],
            axis=-1,
        )
        return torch.from_numpy(coords)

    def get_sample(
        self,
        time_indices: TimeIndices,
        grid_shard_indices: np.ndarray | slice | None = None,
    ) -> dict:
        """Return the per-sample payload in the unified contract."""
        return {
            "data": self.get_data(time_indices, grid_shard_indices),
            "layout": TensorLayout(time=0, variables=2, ensemble=1, grid=3),
            "coordinates": self.get_coordinates(time_indices, grid_shard_indices),
            "metadata": {},
            "grid_size": self.grid_size,
            "grid_shard_indices": grid_shard_indices,
        }

    def tree(self, prefix: str = "") -> Tree:
        tree = super().tree(prefix)
        tree.add(f"Resolution: {self.resolution}")
        return tree


class ObservationDataReader(BaseAnemoiReader):
    """Observation dataset reader (e.g. from tabular zarrs).

    Each sample is built from a single round-trip ``self.data[time_indices, ...]``
    that returns an object exposing ``data``, ``latitudes``, ``longitudes``,
    ``timedeltas`` and ``boundaries``. The boundaries (``tuple[slice, ...]``)
    encode the per-time split of the flat ``N`` axis and travel through
    :attr:`Batch.metadata` rather than being moved to device.
    """

    @property
    def is_static_grid(self) -> bool:
        """Observation readers have a per-sample (dynamic) coordinate set."""
        return False

    @property
    def grid_size(self) -> None:
        """Return None — observation datasets have no static grid."""
        return None

    @property
    def supporting_arrays(self) -> dict:
        """Observations do not have supporting_arrays."""
        return {}

    @property
    def has_trajectories(self) -> bool:
        """Observation datasets do not have trajectories."""
        return False

    def statistics_tendencies(
        self,
        *args,
        **kwargs,
    ) -> dict | None:
        """Observation datasets do not have tendency statistics."""
        del args, kwargs
        return None

    @property
    def metadata(self) -> dict:
        """Return dataset metadata."""
        return {}

    def get_sample(
        self,
        time_indices: TimeIndices,
        grid_shard_indices: np.ndarray | slice | None = None,
    ) -> dict:
        """Get a sample from the observation dataset.

        ``grid_shard_indices`` is accepted for API compatibility with
        :class:`GriddedDataReader` but ignored — sharding for sparse
        observations is not yet implemented.

        Returns
        -------
        dict
            ``{"data": (1, N, V) tensor, "coordinates": (N, 2) tensor,
            "timedeltas": (N,) tensor, "metadata": {"boundaries": ...}}``
            with latitudes/longitudes in **radians** to match the gridded
            reader convention. ``timedeltas`` are kept separate from
            ``coordinates`` so the model layer can route them
            independently.
        """
        x = self.data[time_indices, ...]

        # Introduce a dummy ensemble dimension to align with the (E, N, V)
        # contract; the leading time axis is intentionally absent — per-time
        # structure is recoverable through ``boundaries``.
        data = torch.from_numpy(np.asarray(x.data)[None, ...])
        latitudes = np.deg2rad(np.asarray(x.latitudes))
        longitudes = np.deg2rad(np.asarray(x.longitudes))
        coordinates = torch.from_numpy(
            np.stack([latitudes, longitudes], axis=-1),
        )
        timedeltas = torch.from_numpy(np.asarray(x.timedeltas, dtype=np.float32))

        return {
            "data": data,
            "layout": TensorLayout(grid=0, variables=1, time_in_grid=True),
            "coordinates": coordinates,
            "timedeltas": timedeltas,
            "metadata": {"boundaries": x.boundaries},
            "grid_size": self.grid_size,
            "grid_shard_indices": grid_shard_indices,
        }

    def tree(self, prefix: str = "") -> Tree:
        tree = super().tree(prefix)
        if hasattr(self, "window"):
            tree.add(f"Window: {self.window}")
        return tree


class TrajectoryDataset(GriddedDataReader):
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
        return (self.dates - np.datetime64(self.trajectory_start, "s")) // np.timedelta64(
            trajectory_length_seconds,
            "s",
        )

    def tree(self, prefix: str = "") -> Tree:
        tree = super().tree(prefix)
        tree.add(f"Trajectory start: {self.trajectory_start}")
        tree.add(f"Trajectory length: {self.trajectory_length} steps")
        return tree


def create_dataset(dataset_config: dict, **_kwargs) -> BaseAnemoiReader:
    """Factory function to create dataset based on dataset configuration."""
    dataset_config = _normalize_reader_config(dataset_config)

    trajectory_config = dataset_config.pop("trajectory", {})
    if trajectory_config is not None and hasattr(trajectory_config, "start") and hasattr(trajectory_config, "length"):
        LOGGER.info("Creating a TrajectoryDataset...")
        return TrajectoryDataset(
            **dataset_config,
            trajectory_start=trajectory_config["start"],
            trajectory_length=trajectory_config["length"],
        )

    if "window" in dataset_config["dataset_config"] and "frequency" in dataset_config["dataset_config"]:
        LOGGER.info("Creating ObservationDataReader...")
        return ObservationDataReader(**dataset_config)

    LOGGER.info("Creating a GriddedDataReader...")
    return GriddedDataReader(**dataset_config)
