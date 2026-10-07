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
from anemoi.models.data import TensorLayout
from anemoi.models.data.sample import GriddedSourceSample
from anemoi.models.data.sample import SourceSample
from anemoi.models.data.sample import TabularSourceSample
from anemoi.models.distributed.balanced_partition import get_balanced_partition_sizes
from anemoi.models.distributed.balanced_partition import get_partition_range
from anemoi.models.distributed.shapes import ShardSizes
from anemoi.training.data.usable_indices import get_usable_indices
from anemoi.training.utils.time_indices import TimeIndices

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


def _to_local_window_shard_data(
    data: torch.Tensor,
    coordinates: torch.Tensor,
    timedeltas: torch.Tensor,
    boundaries: list[slice],
    *,
    reader_group_rank: int,
    reader_group_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[slice], list[ShardSizes] | None]:
    """Project sparse windowed tensors to the local reader-rank shard.

    Parameters
    ----------
    data : torch.Tensor
        Full sparse payload of shape ``(N, V)``.
    coordinates : torch.Tensor
        Full coordinates of shape ``(N, 2)``.
    timedeltas : torch.Tensor
        Full per-point timedeltas of shape ``(N,)``.
    boundaries : list[slice]
        One boundary slice per logical time window over the flattened ``N`` axis.
    reader_group_rank : int
        Rank inside the reader group.
    reader_group_size : int
        Size of the reader group.

    Returns
    -------
    tuple
        ``(data_local, coordinates_local, timedeltas_local, boundaries_local, window_shard_sizes)``
        where ``window_shard_sizes`` stores the per-window balanced partition sizes, or is ``None``
        when a single reader reads everything (the payload is then not sharded).
    """
    if reader_group_size <= 1:
        return data, coordinates, timedeltas, boundaries, None

    data_parts: list[torch.Tensor] = []
    coord_parts: list[torch.Tensor] = []
    td_parts: list[torch.Tensor] = []
    boundaries_local: list[slice] = []
    window_shard_sizes_all: list[ShardSizes] = []

    offset = 0
    for boundary in boundaries:
        window_size = boundary.stop - boundary.start
        window_shard_sizes = get_balanced_partition_sizes(window_size, reader_group_size)
        start, end = get_partition_range(window_shard_sizes, reader_group_rank)
        local_slice = slice(boundary.start + start, boundary.start + end)
        local_size = end - start

        data_parts.append(data[local_slice])
        coord_parts.append(coordinates[local_slice])
        td_parts.append(timedeltas[local_slice])
        boundaries_local.append(slice(offset, offset + local_size))
        window_shard_sizes_all.append(window_shard_sizes)
        offset += local_size

    if data_parts:
        data_local = torch.cat(data_parts, dim=0)
        coordinates_local = torch.cat(coord_parts, dim=0)
        timedeltas_local = torch.cat(td_parts, dim=0)
    else:
        data_local = data[:0]
        coordinates_local = coordinates[:0]
        timedeltas_local = timedeltas[:0]

    return data_local, coordinates_local, timedeltas_local, boundaries_local, window_shard_sizes_all


class BaseAnemoiReader(ABC):
    """Generic anemoi data reader."""

    sample_type: type[SourceSample]
    has_trajectories: bool = False

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
        # start/end must sit next to window/frequency for tabular datasets
        self.data = open_dataset(source | {"start": start, "end": end})

        # lazy init reader group info (will be set by DDPGroupStrategy)
        self.reader_group_rank = 0
        self.reader_group_size = 1
        self.grid_shard_sizes = None
        self.grid_shard_slice = None

        #: Sampling config used by :meth:`compute_anchors`.
        #: ``{"stride": 1}`` keeps every valid position;
        #: ``{"stride": None}`` uses stride = window size (non-overlapping).
        self.default_sampling = {"stride": 1}

    @property
    def num_sequences(self) -> int:
        """Number of independent sequences in the dataset."""
        return 1

    def sequence_length(self, sequence: int = 0) -> int:  # noqa: ARG002
        """Return the number of positions in ``sequence``."""
        return len(self.dates)

    @property
    def missing_sequences(self) -> set[int]:
        """Return sequences that are entirely missing and must not be sampled."""
        return set()

    def missing_positions(self, sequence: int = 0) -> set[int]:  # noqa: ARG002
        """Return positions within ``sequence`` that are missing."""
        return set(self.missing)

    def compute_anchors(
        self,
        relative_indices: list[int] | np.ndarray,
        sampling: dict | None = None,
    ) -> np.ndarray:
        """Return the valid ``(sequence, position)`` anchors for a relative window.

        Parameters
        ----------
        relative_indices : list[int] | np.ndarray
            Relative offsets (in positions) requested around each anchor.
        sampling : dict | None
            Sampling configuration with key ``"stride"``.
            ``{"stride": None}`` uses stride = window size (non-overlapping);
            ``{"stride": 1}`` keeps every valid position;
            ``{"stride": 6}`` steps anchors by 6.
            Defaults to :attr:`default_sampling`.

        Returns
        -------
        np.ndarray
            Array of shape ``(n_anchors, 2)`` with ``(sequence, position)`` rows.
        """
        sampling = sampling or self.default_sampling

        rel = np.asarray(list(relative_indices), dtype=np.int64)
        window = int(rel.max()) - int(rel.min()) + 1

        # Resolve stride from sampling dict; None → window size (non-overlapping)
        raw_stride = sampling.get("stride") if isinstance(sampling, dict) else None
        stride = window if raw_stride is None else int(raw_stride)
        if stride < 1:
            msg = f"trajectory_sampling.stride must be >= 1, got {stride}."
            raise ValueError(msg)

        anchors: list[np.ndarray] = []
        for sequence in range(self.num_sequences):
            if sequence in self.missing_sequences:
                continue

            positions = get_usable_indices(
                self.missing_positions(sequence),
                self.sequence_length(sequence),
                rel,
            )

            if stride > 1 and positions.size:
                positions = positions[(positions - positions[0]) % stride == 0]

            if positions.size:
                seq_col = np.full(positions.size, sequence, dtype=np.int64)
                anchors.append(np.stack([seq_col, positions], axis=1))

        if not anchors:
            return np.empty((0, 2), dtype=np.int64)
        return np.concatenate(anchors, axis=0)

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
    def layout(self) -> TensorLayout:
        """Layout of one sample, independent of whether its coordinates change."""

    @property
    def is_tabular(self) -> bool:
        """Return whether the reader produces tabular (observation) samples."""
        return issubclass(self.sample_type, TabularSourceSample)

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

    def set_reader_group_info(self, reader_group_rank: int, reader_group_size: int) -> None:
        """Set reader communication group information (called by DDPGroupStrategy).

        Arguments
        ---------
        reader_group_rank : int
             Reader group rank.
        reader_group_size : int
             Reader group size.
        """
        self.reader_group_rank = reader_group_rank
        self.reader_group_size = reader_group_size

        assert self.reader_group_size >= 1, f"reader_group_size(={self.reader_group_size}) must be positive"

        LOGGER.info(
            "Reader group info set for %s: rank %d / %d",
            self.__class__.__name__,
            self.reader_group_rank,
            self.reader_group_size - 1,
        )

    @abstractmethod
    def get_sample(
        self,
        sequence: int,
        positions: TimeIndices,
    ) -> SourceSample:
        """Return a single per-sample payload.

        Gridded readers return a :class:`GriddedSourceSample` with data of shape
        ``(T, E, N, V)``. Observation readers return a :class:`TabularSourceSample`
        with data of shape ``(E=1, N, V)``, per-point ``timedeltas`` and the time
        windows in ``boundaries``. Coordinates are ``(N, 2)`` ``(latitude, longitude)``
        in **radians**.
        """
        msg = "Subclasses must implement get_sample() method."
        raise NotImplementedError(msg)

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


class GriddedDataReader(BaseAnemoiReader):
    """Gridded dataset reader with static grid."""

    sample_type = GriddedSourceSample

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
        """Return per-grid-point latitudes in **radians**.

        Backed by ``self.data.latitudes`` (which is stored in degrees by
        ``anemoi.datasets``); converted once and cached.
        """
        return np.deg2rad(np.asarray(self.data.latitudes, dtype=np.float32))

    @cached_property
    def longitudes(self) -> np.ndarray:
        """Return per-grid-point longitudes in **radians**."""
        return np.deg2rad(np.asarray(self.data.longitudes, dtype=np.float32))

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

    def get_coordinates(self) -> torch.Tensor:
        """Return the local shard's ``(N, 2)`` ``(latitude, longitude)`` coordinates in **radians**."""
        lats = self.latitudes
        lons = self.longitudes

        if self.grid_shard_slice is not None:
            lats = lats[self.grid_shard_slice]
            lons = lons[self.grid_shard_slice]

        coords = np.stack(
            [np.ascontiguousarray(lats), np.ascontiguousarray(lons)],
            axis=-1,
        )
        return torch.from_numpy(coords)

    def get_sample(
        self,
        sequence: int,
        positions: TimeIndices,
    ) -> GriddedSourceSample:
        """Return the per-sample payload in the unified contract."""
        return GriddedSourceSample(
            data=self.get_data(sequence, positions),
            variables=self.variables,
            layout=self.layout,
            statistics=self.statistics,
            grid_size=self.grid_size,
            coordinates=self.get_coordinates(),
            shard_sizes=self.grid_shard_sizes,
        )


class TabularDataReader(BaseAnemoiReader):
    """Observation dataset reader (e.g. from tabular zarrs).

    Each sample is built from a single round-trip ``self.data[time_indices, ...]``
    that returns an object exposing ``data``, ``latitudes``, ``longitudes``,
    ``timedeltas`` and ``boundaries``. The boundaries (``tuple[slice, ...]``)
    encode the per-time split of the flat ``N`` axis and travel through
    :attr:`Batch.metadata` rather than being moved to device.
    """

    sample_type = TabularSourceSample

    @property
    def layout(self) -> TensorLayout:
        """Return the tabular per-sample layout."""
        return TensorLayout(ensemble=0, grid=1, variables=2)

    @property
    def grid_size(self) -> None:
        """Return None — observation datasets have no static grid."""
        return None

    @property
    def supporting_arrays(self) -> dict:
        """Observations do not have supporting_arrays."""
        return {}

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
        sequence: int,
        positions: TimeIndices,
    ) -> TabularSourceSample:
        """Get a sample from the observation dataset.

        Parameters
        ----------
        sequence : int
            Sequence index; ignored, as tabular datasets have a single sequence.
        positions : TimeIndices
            Time windows and shard selection for the observation sample.

        Returns
        -------
        TabularSourceSample
            Data of shape ``(1, N, V)`` (leading size-1 ensemble axis), ``(N, 2)``
            coordinates in **radians** to match the gridded reader convention, ``(N,)``
            timedeltas and the per-window ``boundaries``.
        """
        del sequence
        # should return list(window_shard_sizes)
        x = self.data[positions]

        # the leading time axis is intentionally absent — per-time
        # structure is recoverable through ``boundaries``.
        data = torch.from_numpy(np.asarray(x.data, dtype=np.float32))
        latitudes = np.deg2rad(np.asarray(x.latitudes, dtype=np.float32))
        longitudes = np.deg2rad(np.asarray(x.longitudes, dtype=np.float32))
        coordinates = torch.from_numpy(np.stack([latitudes, longitudes], axis=-1))
        timedeltas = torch.from_numpy(np.asarray(x.timedeltas, dtype=np.float32))
        boundaries = list(x.boundaries)
        data, coordinates, timedeltas, boundaries, shard_sizes = _to_local_window_shard_data(
            data,
            coordinates,
            timedeltas,
            boundaries,
            reader_group_rank=self.reader_group_rank,
            reader_group_size=self.reader_group_size,
        )

        return TabularSourceSample(
            data=data.unsqueeze(0),  # add a leading, size-1 ensemble axis
            variables=self.variables,
            layout=self.layout,
            statistics=self.statistics,
            coordinates=coordinates,
            timedeltas=timedeltas,
            boundaries=boundaries,
            shard_sizes=shard_sizes,
        )

    def tree(self, prefix: str = "") -> Tree:
        tree = super().tree(prefix)
        if hasattr(self.data, "window"):
            tree.add(f"Window: {self.data.window}")
        return tree


class TrajectoryDataReader(GriddedDataReader):
    """Trajectory dataset with an explicit lead-step axis.

    Wraps a 5-D ``trajectories``-layout dataset opened through
    :func:`anemoi.datasets.open_dataset` (on-disk shape
    ``(base_dates, variables, ensembles, steps, cells)``).  Each base date
    (forecast initialisation) is exposed as an independent sequence and the
    forecast step is the within-sequence position, so a training sample is
    always contained within a single forecast and never crosses initialisation
    boundaries.

    Step subsetting (``steps``, ``step_start``, ``step_end``,
    ``step_frequency``) and base-date subsetting (``start``/``end`` on the
    valid-time envelope, or ``base_start``/``base_end``) are handled by
    ``open_dataset`` via the dataset configuration.
    """

    has_trajectories = True

    def __init__(
        self,
        dataset: str | dict | None = None,
        dataset_config: str | dict | None = None,
        start: datetime.datetime | int | None = None,
        end: datetime.datetime | int | None = None,
        sampling: dict | None = None,
    ) -> None:
        assert not (dataset and dataset_config), "Only one of dataset or dataset_config should be provided."
        assert dataset or dataset_config, "Either dataset or dataset_config must be provided."

        source: dict = _normalize_dataset_config(dataset_config or dataset)
        if source.get("frequency") is not None:
            msg = (
                "TrajectoryDataReader does not accept a 'frequency' in dataset_config. "
                "The step frequency is read directly from the dataset. "
                "Set data.frequency: null in your config."
            )
            raise AssertionError(msg)

        # Trajectory datasets filter by initialisation date; passing start/end
        # would trigger access to .dates, which they do not have.
        open_kwargs = {key: value for key, value in (("base_start", start), ("base_end", end)) if value is not None}
        self.data = open_dataset(source, **open_kwargs)
        self.default_sampling = sampling if sampling is not None else {"stride": None}

        # lazy init reader group info (will be set by DDPGroupStrategy)
        self.reader_group_rank = 0
        self.reader_group_size = 1
        self.grid_shard_sizes = None
        self.grid_shard_slice = None

    @property
    def num_sequences(self) -> int:
        """Number of forecast initialisations (base dates)."""
        return self.data.shape[0]

    def sequence_length(self, sequence: int = 0) -> int:  # noqa: ARG002
        """Return the number of forecast steps per initialisation."""
        return self.data.shape[-2]

    @property
    def missing_sequences(self) -> set[int]:
        """Return the base-date indices that are missing."""
        return set(self.data.missing)

    def missing_positions(self, sequence: int = 0) -> set[int]:  # noqa: ARG002
        """Forecast datasets do not track per-step missing values."""
        return set()

    @property
    def frequency(self) -> datetime.timedelta:
        """Return the step frequency (spacing between consecutive forecast steps)."""
        freq = self.data.step_frequency
        if freq is not None:
            return freq
        msg = (
            f"Cannot determine step frequency: data.step_frequency is None for dataset {self.data}. "
            "Ensure that the dataset configuration includes a valid step_frequency (e.g. '6H')."
        )
        raise ValueError(msg)

    def get_data(self, sequence: int, positions: TimeIndices) -> torch.Tensor:
        """Return steps ``positions`` of initialisation ``sequence`` as ``(steps, ensemble, grid, variables)``."""
        if isinstance(positions, slice):
            positions = list(range(*positions.indices(self.sequence_length(sequence))))
        else:
            positions = np.asarray(positions).tolist()

        # data[sequence] -> (variables, ensembles, steps, cells)
        x = self.data[sequence][:, :, positions, :]
        if self.grid_shard_slice is not None:
            x = x[..., self.grid_shard_slice]

        x = rearrange(x, "variables ensemble steps gridpoints -> steps ensemble gridpoints variables")
        return torch.from_numpy(x)

    def tree(self, prefix: str = "") -> Tree:
        tree = super().tree(prefix)
        tree.add(f"Num initialisations: {self.num_sequences}")
        tree.add(f"Steps per initialisation: {self.sequence_length()}")
        tree.add(f"Sampling: {self.default_sampling}")
        return tree


def create_dataset(dataset_config: dict, **_kwargs) -> BaseAnemoiReader:
    """Factory function to create dataset based on dataset configuration."""
    dataset_config = _normalize_reader_config(dataset_config)
    trajectory_config = _as_dict(dataset_config.pop("trajectory", None))

    if trajectory_config is not None:
        sampling = _as_dict(trajectory_config.get("sampling")) if isinstance(trajectory_config, dict) else None
        LOGGER.info("Creating TrajectoryDataReader...")
        return TrajectoryDataReader(**dataset_config, sampling=sampling)

    if "window" in dataset_config["dataset_config"] and "frequency" in dataset_config["dataset_config"]:
        LOGGER.info("Creating TabularDataReader...")
        return TabularDataReader(**dataset_config)

    LOGGER.info("Creating a GriddedDataReader...")
    return GriddedDataReader(**dataset_config)
