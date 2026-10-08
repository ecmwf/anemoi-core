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
from anemoi.models.data.sample import BaseSample
from anemoi.models.data.sample import GriddedSample
from anemoi.models.data.sample import TabularSample
from anemoi.models.data.sample import create_batched_struct
from anemoi.models.distributed.balanced_partition import get_balanced_partition_sizes
from anemoi.models.distributed.balanced_partition import get_partition_range
from anemoi.models.distributed.shapes import ShardSizes
from anemoi.training.data.usable_indices import ReaderAnchors
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
            "trajectory": {  # optional, for 5-D trajectory datasets
                "sampling": {"stride": int | None},  # optional
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
    latitudes: torch.Tensor,
    longitudes: torch.Tensor,
    timedeltas: torch.Tensor,
    boundaries: list[slice],
    *,
    reader_group_rank: int,
    reader_group_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list[slice], list[ShardSizes] | None]:
    """Project sparse windowed tensors to the local reader-rank shard.

    Parameters
    ----------
    data : torch.Tensor
        Full sparse payload of shape ``(N, V)``.
    latitudes : torch.Tensor
        Full latitudes of shape ``(N,)``.
    longitudes : torch.Tensor
        Full longitudes of shape ``(N,)``.
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
        ``(data_local, latitudes_local, longitudes_local, timedeltas_local, boundaries_local, window_shard_sizes)``
        where ``window_shard_sizes`` stores the per-window balanced partition sizes, or is ``None``
        when a single reader reads everything (the payload is then not sharded).
    """
    if reader_group_size <= 1:
        return data, latitudes, longitudes, timedeltas, boundaries, None

    data_parts: list[torch.Tensor] = []
    lat_parts: list[torch.Tensor] = []
    lon_parts: list[torch.Tensor] = []
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
        lat_parts.append(latitudes[local_slice])
        lon_parts.append(longitudes[local_slice])
        td_parts.append(timedeltas[local_slice])
        boundaries_local.append(slice(offset, offset + local_size))
        window_shard_sizes_all.append(window_shard_sizes)
        offset += local_size

    if data_parts:
        data_local = torch.cat(data_parts, dim=0)
        latitudes_local = torch.cat(lat_parts, dim=0)
        longitudes_local = torch.cat(lon_parts, dim=0)
        timedeltas_local = torch.cat(td_parts, dim=0)
    else:
        data_local = data[:0]
        latitudes_local = latitudes[:0]
        longitudes_local = longitudes[:0]
        timedeltas_local = timedeltas[:0]

    return data_local, latitudes_local, longitudes_local, timedeltas_local, boundaries_local, window_shard_sizes_all


class BaseAnemoiReader(ABC):
    """Generic anemoi data reader."""

    sample_type: type[BaseSample]
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

        #: Configured anchor sampling (``{"stride": int | None}``), or ``None`` if not configured.
        #: The readers of a MultiDataset must agree on it.
        self.sampling: dict | None = None

    def valid_anchors(self, relative_indices: list[int] | np.ndarray) -> ReaderAnchors:
        """Return the anchors at which every ``position + relative_index`` can be read.

        Single-sequence readers anchor on their own ``dates``: the anchor time is the
        date at relative offset 0.
        """
        positions = get_usable_indices(self.missing, len(self.dates), relative_indices)
        return ReaderAnchors(
            times=np.asarray(self.dates)[positions],
            sequences=np.zeros_like(positions),
            positions=positions,
        )

    @property
    def dates(self) -> np.ndarray:
        """Return dataset dates."""
        return self.data.dates

    @property
    def grid_size(self) -> int:
        """Return dataset grid size."""
        return self.data.shape[0]

    @cached_property
    def statistics(self) -> dict:
        """Return dataset statistics.

        Cached: some ``anemoi.datasets`` stores rebuild the dict (and re-read the arrays)
        on every access, and every sample must carry the same statistics object for
        :meth:`BaseSample.collate` to accept them together.
        """
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
        return issubclass(self.sample_type, TabularSample)

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
    ) -> BaseSample:
        """Return a single per-sample payload.

        Gridded readers return a :class:`GriddedSample` with data of shape
        ``(T, E, N, V)``. Observation readers return a :class:`TabularSample`
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

    sample_type = GriddedSample

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
            data_type="gridded",
            data=data,
            variables=self.variables,
            layout=self.layout,
            statistics=self.statistics,
            grid_size=self.grid_size,
            latitudes=latitudes,
            longitudes=longitudes,
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

    sample_type = TabularSample

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
    ) -> TabularSample:
        """Get a sample from the observation dataset.

        Parameters
        ----------
        sequence : int
            Sequence index; ignored, as tabular datasets have a single sequence.
        positions : TimeIndices
            Time windows and shard selection for the observation sample.

        Returns
        -------
        TabularSample
            Data of shape ``(1, N, V)`` (leading size-1 ensemble axis), ``(N,)``
            latitudes and longitudes in **radians** (converted by
            :func:`create_batched_struct`), ``(N,)``
            timedeltas and the per-window ``boundaries``.
        """
        del sequence
        # should return list(window_shard_sizes)
        x = self.data[positions]

        # the leading time axis is intentionally absent — per-time
        # structure is recoverable through ``boundaries``.
        data = torch.from_numpy(np.asarray(x.data, dtype=np.float32))
        latitudes = torch.from_numpy(np.asarray(x.latitudes, dtype=np.float32))
        longitudes = torch.from_numpy(np.asarray(x.longitudes, dtype=np.float32))

        timedeltas = torch.from_numpy(np.asarray(x.timedeltas, dtype=np.float32))
        boundaries = list(x.boundaries)
        data, latitudes, longitudes, timedeltas, boundaries, shard_sizes = _to_local_window_shard_data(
            data,
            latitudes,
            longitudes,
            timedeltas,
            boundaries,
            reader_group_rank=self.reader_group_rank,
            reader_group_size=self.reader_group_size,
        )

        return create_batched_struct(
            data_type="tabular",
            data=data.unsqueeze(0),  # add a leading, size-1 ensemble axis
            variables=self.variables,
            layout=self.layout,
            statistics=self.statistics,
            latitudes=latitudes,
            longitudes=longitudes,
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
                f"{self.__class__.__name__} does not accept a 'frequency' in dataset_config. "
                "The step frequency is read directly from the dataset. "
                "Set data.frequency: null in your config."
            )
            raise AssertionError(msg)

        # Trajectory datasets filter by initialisation date; passing start/end
        # would trigger access to .dates, which they do not have.
        open_kwargs = {key: value for key, value in (("base_start", start), ("base_end", end)) if value is not None}
        self.data = open_dataset(source, **open_kwargs)
        self.sampling = sampling if sampling is not None else {"stride": None}

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

    def valid_anchors(self, relative_indices: list[int] | np.ndarray) -> ReaderAnchors:
        """Return the ``(base date, step)`` anchors whose window fits inside one forecast.

        Missing base dates are skipped; steps are never missing. The anchor time is
        the valid time ``base_date + step`` at relative offset 0.
        """
        steps = get_usable_indices(set(), self.sequence_length(), relative_indices)
        missing = np.array(sorted(self.data.missing), dtype=np.int64)
        sequences = np.setdiff1d(np.arange(self.num_sequences, dtype=np.int64), missing)

        sequence_rows = np.repeat(sequences, len(steps))
        position_rows = np.tile(steps, len(sequences))
        base_dates = np.asarray(self.data.base_dates).astype("datetime64[s]")[sequence_rows]
        lead_times = np.asarray(self.data.steps).astype("timedelta64[s]")[position_rows]
        return ReaderAnchors(
            times=base_dates + lead_times,
            sequences=sequence_rows,
            positions=position_rows,
            base_dates=base_dates,
        )

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
        tree.add(f"Sampling: {self.sampling}")
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
