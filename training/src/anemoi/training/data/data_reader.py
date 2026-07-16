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
from functools import cached_property

import numpy as np
import torch
from einops import rearrange
from rich.console import Console
from rich.tree import Tree

from anemoi.datasets import open_dataset
from anemoi.training.data.usable_indices import get_usable_indices
from anemoi.training.utils.configs import _as_dict
from anemoi.training.utils.configs import _normalize_dataset_config
from anemoi.training.utils.configs import _normalize_reader_config
from anemoi.training.utils.time_indices import TimeIndices

LOGGER = logging.getLogger(__name__)


class BaseAnemoiReader:
    """Anemoi data reader for native grid datasets.

    A sample is addressed by a 2-D anchor ``(sequence, position)``.
    Relative time offsets are applied *within* a single sequence, so a
    sample can never span two sequences.  Analysis datasets expose a single
    sequence (the whole time series); forecast datasets expose one sequence
    per initialisation (base date), with the forecast step as the position.
    """

    def __init__(
        self,
        dataset_config: str | dict | None = None,
        start: datetime.datetime | int | None = None,
        end: datetime.datetime | int | None = None,
    ):
        """Initialize Anemoi data reader."""
        if dataset_config is None:
            msg = "Error: dataset_config must be provided."
            raise ValueError(msg)

        dataset_config = _normalize_dataset_config(dataset_config)

        self.data = open_dataset(dataset_config, start=start, end=end)

        #: Sampling config used by :meth:`compute_anchors`.
        #: ``{"stride": 1}`` keeps every valid position;
        #: ``{"stride": None}`` uses stride = window size (non-overlapping).
        self.default_sampling: dict = {"stride": 1}

    # ------------------------------------------------------------------
    # Sequence / position geometry
    # ------------------------------------------------------------------

    @property
    def num_sequences(self) -> int:
        """Number of independent sequences in the dataset."""
        return 1

    def sequence_length(self, sequence: int = 0) -> int:  # noqa: ARG002
        """Return the number of positions in ``sequence``."""
        return len(self.data.dates)

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

    # ------------------------------------------------------------------
    # Dataset properties
    # ------------------------------------------------------------------

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
        """Return dataset name-to-index mapping."""
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

    # ------------------------------------------------------------------
    # Sample loading
    # ------------------------------------------------------------------

    def get_sample(
        self,
        sequence: int,
        positions: TimeIndices,
        grid_shard_indices: np.ndarray | slice | None = None,
    ) -> torch.Tensor:
        """Get a sample from the dataset.

        For analysis datasets there is a single sequence, so ``sequence`` is
        ignored and ``positions`` index the time axis directly.
        """
        del sequence  # analysis datasets have a single sequence
        if isinstance(grid_shard_indices, slice):
            x = self.data[positions, :, :, grid_shard_indices]
        else:
            x = self.data[positions, :, :, :]
            if grid_shard_indices is not None:
                x = x[..., grid_shard_indices]

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
    """Native grid (analysis) dataset.

    A single sequence covering the whole time series; relative offsets index
    the time axis directly.  This is the default analysis behaviour.
    """

    def __init__(
        self,
        dataset: str | dict | None = None,
        dataset_config: str | dict | None = None,
        start: datetime.datetime | int | None = None,
        end: datetime.datetime | int | None = None,
        sampling: dict | None = None,
    ) -> None:
        """Initialize NativeGridDataset."""
        super().__init__(dataset=dataset, dataset_config=dataset_config, start=start, end=end)
        if sampling is not None:
            self.default_sampling = sampling


class TrajectoryDataset(BaseAnemoiReader):
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

    def __init__(
        self,
        dataset_config: str | dict | None = None,
        start: datetime.datetime | int | None = None,
        end: datetime.datetime | int | None = None,
        sampling: dict | None = None,
    ) -> None:
        if dataset_config is None:
            msg = "Error: dataset_config must be provided."
            raise ValueError(msg)

        dataset_config = _normalize_dataset_config(dataset_config)
        # Trajectory datasets derive their step frequency from the dataset itself.
        # Passing data.frequency would be misleading and is not supported.
        if isinstance(dataset_config, dict) and dataset_config.get("frequency") is not None:
            msg = (
                "TrajectoryDataset does not accept a 'frequency' in dataset_config. "
                "The step frequency is read directly from the dataset. "
                "Set data.frequency: null in your config."
            )
            raise AssertionError(msg)

        # Trajectory datasets use base_start/base_end to filter by initialisation date;
        # passing start/end would trigger access to .dates which doesn't exist on them.
        open_kwargs: dict = {}
        if start is not None:
            open_kwargs["base_start"] = start
        if end is not None:
            open_kwargs["base_end"] = end
        self.data = open_dataset(dataset_config, **open_kwargs)
        self.default_sampling = sampling if sampling is not None else {"stride": None}

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

    def statistics_tendencies(
        self,
        timestep: int | str | datetime.timedelta | None = None,  # noqa: ARG002
    ) -> dict | None:
        """Tendency statistics are not defined for forecast datasets."""
        return None

    def get_sample(
        self,
        sequence: int,
        positions: TimeIndices,
        grid_shard_indices: np.ndarray | slice | None = None,
    ) -> torch.Tensor:
        """Load forecast steps ``positions`` of initialisation ``sequence``."""
        if isinstance(positions, slice):
            positions = list(range(*positions.indices(self.sequence_length(sequence))))
        else:
            positions = np.asarray(positions).tolist()

        # data[sequence] -> (variables, ensembles, steps, cells)
        x = self.data[sequence]
        x = x[:, :, positions, :]
        if grid_shard_indices is not None:
            x = x[..., grid_shard_indices]

        x = rearrange(x, "variables ensemble steps gridpoints -> steps ensemble gridpoints variables")
        return torch.from_numpy(x)

    def tree(self, prefix: str = "") -> Tree:
        tree = Tree(prefix + " 💾 " + f"{self.__class__.__name__}")
        tree.add(f"Dataset: {self.data}")
        tree.add(f"Step frequency: {self.frequency}")
        tree.add(f"Resolution: {self.resolution}")
        tree.add(f"Num variables: {len(self.name_to_index)}")
        tree.add(f"Num initialisations: {self.num_sequences}")
        tree.add(f"Steps per initialisation: {self.sequence_length()}")
        tree.add(f"Sampling: {self.default_sampling}")
        return tree


def create_dataset(dataset_config: dict, **_kwargs) -> BaseAnemoiReader:
    """Factory function to create a data reader based on the dataset configuration."""
    dataset_config = _normalize_reader_config(dataset_config)
    trajectory_config = _as_dict(dataset_config.pop("trajectory", None))

    if trajectory_config is not None:
        sampling = trajectory_config.get("sampling") if isinstance(trajectory_config, dict) else None
        LOGGER.info("Creating TrajectoryDataset...")
        return TrajectoryDataset(**dataset_config, sampling=_as_dict(sampling))

    LOGGER.info("Creating NativeGridDataset...")
    return NativeGridDataset(**dataset_config)
