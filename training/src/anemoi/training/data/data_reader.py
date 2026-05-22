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
from anemoi.training.utils.time_indices import TimeIndices
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
        time_indices: TimeIndices,
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


class _ForecastZarrData:
    """Wrapper around a 5D zarr that exposes the dataset-like interface expected by the pipeline."""

    def __init__(
        self,
        zarr_group: "zarr.Group",
        init_dates: np.ndarray,
        init_indices: np.ndarray,
        variables: list[str],
        var_indices: list[int],
        resolution: str,
        step_frequency: datetime.timedelta,
        forecast_steps: int,
    ) -> None:
        self._zarr = zarr_group
        self._array = zarr_group["data"]
        self._init_dates = init_dates
        self._init_indices = init_indices
        self._variables_list = variables
        self._var_indices = var_indices
        self._resolution = resolution
        self._step_frequency = step_frequency
        self._forecast_steps = forecast_steps

    @property
    def shape(self) -> tuple:
        return self._array.shape

    @property
    def dates(self) -> np.ndarray:
        return self._init_dates

    @property
    def grids(self) -> list[int]:
        return [self._array.shape[4]]

    @property
    def variables(self) -> list[str]:
        return self._variables_list

    @property
    def frequency(self) -> datetime.timedelta:
        return self._step_frequency

    @property
    def resolution(self) -> str:
        return self._resolution

    @property
    def name_to_index(self) -> dict[str, int]:
        return {name: i for i, name in enumerate(self._variables_list)}

    @property
    def missing(self) -> set[int]:
        return set()

    @property
    def statistics(self) -> dict:
        stats = {}
        for key in ("mean", "stdev", "minimum", "maximum"):
            if key in self._zarr:
                stats[key] = self._zarr[key][:][self._var_indices]
        return stats

    def metadata(self) -> dict:
        return dict(self._zarr.attrs)

    def supporting_arrays(self) -> dict:
        result = {}
        if "latitudes" in self._zarr:
            result["latitudes"] = self._zarr["latitudes"][:]
        if "longitudes" in self._zarr:
            result["longitudes"] = self._zarr["longitudes"][:]
        return result

    def __getitem__(self, item: "tuple | int | slice | list | np.ndarray") -> np.ndarray:
        """Use zarr orthogonal indexing to support list/array indices."""
        if isinstance(item, tuple) and any(isinstance(idx, (list, np.ndarray)) for idx in item):
            return self._array.oindex[item]
        return self._array[item]

    def __repr__(self) -> str:
        return f"ForecastZarrData(shape={self.shape}, inits={len(self._init_dates)})"


class ForecastStepDataset(BaseAnemoiReader):
    """Dataset reader for 5D zarrs with an explicit forecast-step dimension.

    Expected zarr shape: ``(num_inits, variables, ensemble, forecast_steps, gridpoints)``.
    Expected zarr arrays: ``data``, ``base_dates``, ``steps``.

    This reader virtualizes the dataset as a flat time series of length
    ``num_inits * forecast_steps`` so that the rest of the training pipeline
    (tasks, usable indices, multidataset) works unchanged.  Trajectory
    boundaries are automatically placed at initialization boundaries so
    that samples never cross forecasts.
    """

    def __init__(
        self,
        forecast_steps: int,
        step_frequency: str | datetime.timedelta = "1h",
        dataset: str | dict | None = None,
        dataset_config: str | dict | None = None,
        start: datetime.datetime | int | None = None,
        end: datetime.datetime | int | None = None,
    ):
        # Bypass BaseAnemoiReader.__init__ — open_dataset doesn't support 5D zarrs.
        import zarr

        source = dataset_config if dataset_config is not None else dataset
        if source is None:
            msg = "Either dataset or dataset_config must be provided."
            raise ValueError(msg)

        # Extract the zarr path and drop list from the config
        drop: list[str] = []
        if isinstance(source, (dict, DictConfig)):
            zarr_path = source.get("dataset")
            if zarr_path is None:
                msg = "dataset_config must contain a 'dataset' key with the zarr path."
                raise ValueError(msg)
            drop = list(source.get("drop", []) or [])
        else:
            zarr_path = source

        self._zarr = zarr.open(zarr_path, mode="r")
        self._forecast_steps = forecast_steps

        if isinstance(step_frequency, str):
            from anemoi.utils.dates import frequency_to_timedelta

            self._step_frequency = frequency_to_timedelta(step_frequency)
        else:
            self._step_frequency = step_frequency

        # Validate shape: expect 5D (inits, vars, ensemble, steps, grid)
        data_array = self._zarr["data"]
        if len(data_array.shape) != 5:
            msg = (
                f"ForecastStepDataset expects a 5D zarr "
                f"(inits, variables, ensemble, steps, gridpoints), got shape {data_array.shape}"
            )
            raise ValueError(msg)

        actual_steps = data_array.shape[3]
        if actual_steps < forecast_steps:
            msg = f"Dataset has {actual_steps} forecast steps but config requests {forecast_steps}."
            raise ValueError(msg)

        # Parse metadata from zarr attrs
        self._attrs = dict(self._zarr.attrs)
        all_variables = self._attrs.get("variables", [])

        # Apply drop list
        drop_set = set(drop)
        self._var_indices = [i for i, v in enumerate(all_variables) if v not in drop_set]
        variables = [all_variables[i] for i in self._var_indices]

        # Load and filter init dates by start/end
        base_dates = self._zarr["base_dates"][:]
        if base_dates.dtype.kind != "M":
            base_dates = base_dates.astype("datetime64[us]")
        self._init_dates, self._init_indices = self._filter_init_dates(base_dates, start, end)

        # Create the data wrapper that exposes the expected interface
        self.data = _ForecastZarrData(
            zarr_group=self._zarr,
            init_dates=self._init_dates,
            init_indices=self._init_indices,
            variables=variables,
            var_indices=self._var_indices,
            resolution=self._attrs.get("resolution", "unknown"),
            step_frequency=self._step_frequency,
            forecast_steps=self._forecast_steps,
        )

    @staticmethod
    def _filter_init_dates(
        base_dates: np.ndarray,
        start: datetime.datetime | int | None,
        end: datetime.datetime | int | None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Filter initialization dates by start/end and return (dates, indices)."""
        indices = np.arange(len(base_dates))

        if start is not None:
            start = np.datetime64(f"{start}-01-01") if isinstance(start, int) else np.datetime64(start)
            mask = base_dates >= start
            indices = indices[mask]
            base_dates = base_dates[mask]

        if end is not None:
            end = np.datetime64(f"{end}-12-31T23:59:59") if isinstance(end, int) else np.datetime64(end)
            mask = base_dates <= end
            indices = indices[mask]
            base_dates = base_dates[mask]

        return base_dates, indices

    @property
    def num_initializations(self) -> int:
        """Number of forecast initializations in the dataset."""
        return len(self._init_dates)

    @property
    def dates(self) -> np.ndarray:
        """Virtual flat dates array of length num_inits * forecast_steps."""
        step_offsets = np.array(
            [np.timedelta64(int(self._step_frequency.total_seconds() * i), "s") for i in range(self._forecast_steps)],
        )
        all_dates = self._init_dates[:, None] + step_offsets[None, :]
        return all_dates.ravel()

    @property
    def has_trajectories(self) -> bool:
        """Return whether the dataset has trajectories."""
        return True

    @property
    def trajectory_ids(self) -> np.ndarray:
        """Each virtual index maps to its initialization index."""
        return np.repeat(np.arange(self.num_initializations), self._forecast_steps)

    def get_sample(
        self,
        time_indices: TimeIndices,
        grid_shard_indices: np.ndarray | slice | None = None,
    ) -> torch.Tensor:
        """Get a sample from the 5D dataset.

        Maps virtual flat time indices to (init_index, step_indices) and
        loads from the 5D zarr.
        """
        # Convert TimeIndices to a flat array of ints
        if isinstance(time_indices, slice):
            indices = np.arange(*time_indices.indices(self.num_initializations * self._forecast_steps))
        elif isinstance(time_indices, (list, tuple)):
            indices = np.array(time_indices)
        else:
            indices = np.asarray(time_indices)

        # All indices must belong to the same initialization
        virtual_init_idx = indices[0] // self._forecast_steps
        step_indices = indices - virtual_init_idx * self._forecast_steps

        # Map virtual init index to actual zarr init index
        actual_init_idx = self._init_indices[virtual_init_idx]

        # Load from 5D zarr: (variables, ensemble, steps, gridpoints)
        if isinstance(grid_shard_indices, slice):
            x = self.data[actual_init_idx, :, :, step_indices.tolist(), grid_shard_indices]
        else:
            x = self.data[actual_init_idx, :, :, step_indices.tolist(), :]
            if grid_shard_indices is not None:
                x = x[..., grid_shard_indices]

        # Apply variable drop
        x = x[self._var_indices]

        # x shape: (variables, ensemble, num_requested_steps, gridpoints)
        x = rearrange(x, "variables ensemble steps gridpoints -> steps ensemble gridpoints variables")
        return torch.from_numpy(x)

    def tree(self, prefix: str = "") -> Tree:
        tree = Tree(prefix + " 💾 " + f"{self.__class__.__name__}")
        tree.add(f"Zarr: {self._zarr.store.path if hasattr(self._zarr.store, 'path') else 'N/A'}")
        tree.add(f"Resolution: {self.resolution}")
        tree.add(f"Num variables: {len(self.variables)}")
        tree.add(f"Forecast steps: {self._forecast_steps}")
        tree.add(f"Step frequency: {self.frequency}")
        tree.add(f"Num initializations: {self.num_initializations}")
        tree.add(f"Virtual length: {self.num_initializations * self._forecast_steps}")
        return tree


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
    forecast_steps = dataset_config.pop("forecast_steps", None)
    step_frequency = dataset_config.pop("step_frequency", None)

    if forecast_steps is not None:
        LOGGER.info("Creating ForecastStepDataset...")
        kwargs = {"forecast_steps": forecast_steps}
        if step_frequency is not None:
            kwargs["step_frequency"] = step_frequency
        return ForecastStepDataset(**dataset_config, **kwargs)

    if trajectory_config is not None and hasattr(trajectory_config, "start") and hasattr(trajectory_config, "length"):
        LOGGER.info("Creating TrajectoryDataset...")
        return TrajectoryDataset(
            **dataset_config,
            trajectory_start=trajectory_config["start"],
            trajectory_length=trajectory_config["length"],
        )

    LOGGER.info("Creating NativeGridDataset...")
    return NativeGridDataset(**dataset_config)
