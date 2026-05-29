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


def dates_to_unix_ns(dates: object) -> np.ndarray | None:
    dates_array = np.asarray(dates)
    if np.issubdtype(dates_array.dtype, np.datetime64):
        return dates_array.astype("datetime64[ns]").astype(np.int64, copy=False)

    try:
        return np.array([np.datetime64(date_value, "ns").astype(np.int64) for date_value in dates], dtype=np.int64)
    except (TypeError, ValueError):
        return None


def expand_time_indices(time_indices: slice | int | list[int]) -> list[int]:
    if isinstance(time_indices, int):
        return [int(time_indices)]
    if isinstance(time_indices, slice):
        start = 0 if time_indices.start is None else int(time_indices.start)
        stop = int(time_indices.stop)
        step = 1 if time_indices.step is None else int(time_indices.step)
        return list(range(start, stop, step))
    return [int(value) for value in time_indices]


def ensure_time_axis(sample: torch.Tensor) -> torch.Tensor:
    if sample.ndim == 3:
        return sample.unsqueeze(0)
    return sample


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


class RelativeTimeReader:
    """Reader wrapper that resolves sample indices to native dataset time indices."""

    def __init__(
        self,
        reader: BaseAnemoiReader,
        *,
        native_relative_indices: np.ndarray,
        model_relative_indices: np.ndarray | None = None,
        frequency_seconds: int | None = None,
        anchor_dates_ns: np.ndarray | None = None,
    ) -> None:
        self.reader = reader
        self.native_relative_indices = np.asarray(native_relative_indices, dtype=np.int64)
        self.model_relative_indices = (
            None if model_relative_indices is None else np.asarray(model_relative_indices, dtype=np.int64)
        )
        self.frequency_seconds = frequency_seconds
        self.anchor_dates_ns = None if anchor_dates_ns is None else np.asarray(anchor_dates_ns, dtype=np.int64)
        self._warned_nearest_time_fallback = False

    def __getattr__(self, name: str):
        return getattr(self.reader, name)

    @cached_property
    def dates_ns(self) -> np.ndarray | None:
        return dates_to_unix_ns(self.reader.dates)

    @cached_property
    def date_to_native_index(self) -> dict[int, int] | None:
        if self.dates_ns is None:
            return None
        return {int(date_ns): idx for idx, date_ns in enumerate(self.dates_ns)}

    @property
    def uses_mixed_frequency_alignment(self) -> bool:
        return (
            self.model_relative_indices is not None
            and self.frequency_seconds is not None
            and self.anchor_dates_ns is not None
            and self.dates_ns is not None
            and self.date_to_native_index is not None
        )

    def get_sample(
        self,
        sample_index: int,
        grid_shard_indices: np.ndarray | slice | None = None,
    ) -> torch.Tensor:
        if not self.uses_mixed_frequency_alignment:
            return self.reader.get_sample(self.resolve_dense_time_indices(sample_index), grid_shard_indices)
        return self.get_mixed_frequency_sample(sample_index, grid_shard_indices)

    def resolve_dense_time_indices(self, sample_index: int) -> TimeIndices:
        absolute_indices = sample_index + self.native_relative_indices
        if len(absolute_indices) == 1:
            return int(absolute_indices[0])

        diffs = np.diff(absolute_indices)
        if len(diffs) > 0 and np.all(diffs == diffs[0]):
            step = int(diffs[0])
            start = int(absolute_indices[0])
            stop = int(absolute_indices[-1] + step)
            return slice(start, stop, step)

        return absolute_indices.tolist()

    def resolve_mixed_frequency_time_indices(self, sample_index: int) -> TimeIndices:
        if not self.uses_mixed_frequency_alignment:
            dense_time_indices = self.resolve_dense_time_indices(sample_index)
            if isinstance(dense_time_indices, slice):
                return expand_time_indices(dense_time_indices)
            return dense_time_indices

        if not (0 <= sample_index < len(self.anchor_dates_ns)):
            return [-1] * len(self.model_relative_indices)

        anchor_date_ns = int(self.anchor_dates_ns[sample_index])
        offsets_ns = self.model_relative_indices.astype(np.int64, copy=False) * self.frequency_seconds * 1_000_000_000
        requested_dates_ns = anchor_date_ns + offsets_ns
        resolved_native_indices = np.array(
            [self.date_to_native_index.get(int(date_ns), -1) for date_ns in requested_dates_ns],
            dtype=np.int64,
        )
        missing_mask = resolved_native_indices < 0
        if np.any(missing_mask):
            dates_ns = self.dates_ns.astype(np.int64, copy=False)
            missing_dates_ns = requested_dates_ns[missing_mask]
            insertion_indices = np.searchsorted(dates_ns, missing_dates_ns, side="left")
            lower_indices = np.clip(insertion_indices - 1, 0, len(dates_ns) - 1)
            upper_indices = np.clip(insertion_indices, 0, len(dates_ns) - 1)

            lower_distances = np.full(insertion_indices.shape, np.iinfo(np.int64).max, dtype=np.int64)
            upper_distances = np.full(insertion_indices.shape, np.iinfo(np.int64).max, dtype=np.int64)
            valid_lower = insertion_indices > 0
            valid_upper = insertion_indices < len(dates_ns)
            lower_distances[valid_lower] = np.abs(
                dates_ns[lower_indices[valid_lower]] - missing_dates_ns[valid_lower],
            )
            upper_distances[valid_upper] = np.abs(
                dates_ns[upper_indices[valid_upper]] - missing_dates_ns[valid_upper],
            )

            use_upper = upper_distances < lower_distances
            nearest_indices = np.where(use_upper, upper_indices, lower_indices)
            nearest_distances = np.minimum(lower_distances, upper_distances)
            tolerance_ns = max(1, (frequency_to_seconds(self.reader.frequency) * 1_000_000_000) // 2)
            within_tolerance = nearest_distances <= tolerance_ns
            resolved_native_indices[missing_mask] = np.where(within_tolerance, nearest_indices, -1)
            if np.any(within_tolerance) and not self._warned_nearest_time_fallback:
                LOGGER.warning(
                    "Mixed-frequency alignment for dataset '%s' fell back to the nearest native timestamp "
                    "for %d requested "
                    "times within a %d-second tolerance. Missing exact timestamps may map to a neighbouring "
                    "analysis cycle.",
                    getattr(self.reader, "data", self.reader.__class__.__name__),
                    int(np.count_nonzero(within_tolerance)),
                    tolerance_ns // 1_000_000_000,
                )
                self._warned_nearest_time_fallback = True

        if len(resolved_native_indices) == 1:
            return int(resolved_native_indices[0])
        return resolved_native_indices.tolist()

    def get_mixed_frequency_sample(
        self,
        sample_index: int,
        grid_shard_indices: np.ndarray | slice | None = None,
    ) -> torch.Tensor:
        time_indices = self.resolve_mixed_frequency_time_indices(sample_index)
        requested_indices = expand_time_indices(time_indices)
        valid_positions = [
            pos
            for pos, native_index in enumerate(requested_indices)
            if 0 <= native_index < len(self.reader.dates) and native_index not in self.reader.missing
        ]
        if len(valid_positions) == len(requested_indices):
            return self.reader.get_sample(time_indices, grid_shard_indices)

        valid_indices = [requested_indices[pos] for pos in valid_positions]
        loaded = None
        if len(valid_indices) > 0:
            loaded = self.reader.get_sample(valid_indices, grid_shard_indices)
            loaded = ensure_time_axis(loaded)

        if loaded is None:
            probe_idx = next((idx for idx in range(len(self.reader.dates)) if idx not in self.reader.missing), None)
            if probe_idx is None:
                msg = "Reader has no available native indices."
                raise ValueError(msg)
            probe = self.reader.get_sample([probe_idx], grid_shard_indices)
            probe = ensure_time_axis(probe)
            output = torch.full(
                (len(requested_indices), *tuple(probe.shape[1:])),
                torch.nan,
                dtype=torch.float32,
                device=probe.device,
            )
        else:
            dtype = loaded.dtype if loaded.is_floating_point() else torch.float32
            output = torch.full(
                (len(requested_indices), *tuple(loaded.shape[1:])),
                torch.nan,
                dtype=dtype,
                device=loaded.device,
            )
            if not loaded.is_floating_point():
                loaded = loaded.float()

        if len(valid_positions) > 0:
            output[valid_positions] = loaded

        return output


def create_dataset(dataset_config: dict, **_kwargs) -> BaseAnemoiReader:
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
