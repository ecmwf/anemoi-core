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
import re
import zlib
from functools import cached_property

import numpy as np
import torch
from einops import rearrange
from omegaconf import DictConfig
from rich.console import Console
from rich.tree import Tree

from anemoi.datasets import open_dataset
from anemoi.graphs.nodes.builders.base import BaseNodeBuilder
from anemoi.training.data.usable_indices import get_usable_indices
from anemoi.training.utils.time_indices import TimeIndices
from anemoi.utils.dates import frequency_to_string
from anemoi.utils.dates import frequency_to_timedelta

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

    allowed_keys = {"start", "end", "trajectory"}
    unknown_keys = set(normalized) - allowed_keys
    if unknown_keys:
        unknown = ", ".join(sorted(unknown_keys))
        allowed = ", ".join(sorted({"dataset_config", *allowed_keys}))
        msg = f"Unknown dataset reader option(s) [{unknown}]. Allowed top-level keys: {allowed}."
        raise ValueError(msg)

    normalized["dataset_config"] = base_dataset_config
    return normalized


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
        dataset: str | dict | None = None,
        dataset_config: str | dict | None = None,
        start: datetime.datetime | int | None = None,
        end: datetime.datetime | int | None = None,
        sampling: dict | None = None,
    ) -> None:
        source = dataset_config if dataset_config is not None else dataset
        if source is None:
            msg = "Either dataset or dataset_config must be provided."
            raise ValueError(msg)

        # Trajectory datasets derive their step frequency from the dataset itself.
        # Passing data.frequency would be misleading and is not supported.
        source_dict = _as_dict(source) if not isinstance(source, str) else {}
        if isinstance(source_dict, dict) and source_dict.get("frequency") is not None:
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
        self.data = open_dataset(_normalize_dataset_config(source), **open_kwargs)
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


# ----------------------------------------------------------------------
# Synthetic data
# ----------------------------------------------------------------------
#
# A synthetic reader lets you train without bringing a dataset. Select it with
#
#   system.input.dataset: synthetic:o800
#
# or, for more control, in dataloader.<split>.datasets.<name>.dataset_config:
#
#   dataset: synthetic:o800        # any octahedral (oNNN) grid
#   synthetic:                     # optional
#     variables: [...]             # default: the standard 101-variable AIFS set
#     ensemble: 1
#     seed: 0
#   select / drop: [...]           # applied to the variable list, like open_dataset
#
# Fields are deterministic in (seed, variable, date, member), so every worker
# and rank sees the same "climate". They are built from zonal-mean profiles
# (equator-to-pole gradients, jets, a moist tropical belt) plus eastward
# travelling planetary waves, scaled with ERA5-like statistics. Forcings
# (lat/lon, julian day, local time, insolation) are computed exactly, and the
# static fields (lsm, z, sdor, slor) come from one fixed "orography" field so
# they agree with each other.

SYNTHETIC_PREFIX = "synthetic:"

_SYNTHETIC_LEVELS = (50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000)

# name: (mean, stdev, minimum, maximum), taken from the o800 2023 ERA5 benchmarking dataset.
# fmt: off
_SYNTHETIC_STATS: dict[str, tuple[float, float, float, float]] = {
    "10u": (-0.5924, 5.397, -38.75, 33.43), "10v": (0.3026, 4.463, -34.5, 41.11),
    "2d": (284.4, 14.44, 191.1, 305.7), "2t": (289.7, 14.64, 194.0, 326.8),
    "cos_julian_day": (-0.1892, 0.66, -1.0, 1.0), "sin_julian_day": (0.1386, 0.7137, -1.0, 1.0),
    "cos_latitude": (0.8087, 0.2155, 0.0015, 1.0), "sin_latitude": (0.0, 0.5473, -1.0, 1.0),
    "cos_longitude": (0.0, 0.7071, -1.0, 1.0), "sin_longitude": (0.0, 0.7071, -1.0, 1.0),
    "cos_local_time": (0.0, 0.7071, -1.0, 1.0), "sin_local_time": (0.0, 0.7071, -1.0, 1.0),
    "insolation": (0.256, 0.3283, 0.0, 1.0), "cos_solar_zenith_angle": (0.256, 0.3283, 0.0, 1.0),
    "cp": (0.000399, 0.001299, 0.0, 0.1039), "tp": (0.0007504, 0.002556, 0.0, 0.3457),
    "lsm": (0.2802, 0.4383, 0.0, 1.0), "msl": (1.012e5, 1063.0, 9.226e4, 1.066e5),
    "sdor": (20.74, 61.9, 0.0, 1071.0), "slor": (0.003425, 0.01005, 0.0001, 0.1761),
    "skt": (290.5, 15.45, 193.8, 343.6), "sp": (9.873e4, 6537.0, 4.78e4, 1.061e5),
    "tcw": (27.05, 17.41, 0.04, 136.5), "z": (2168.0, 5965.0, -1445.0, 5.756e4),
}
# per-level (mean, stdev) of pressure-level variables, ordered like _SYNTHETIC_LEVELS
_SYNTHETIC_PL_STATS: dict[str, tuple[tuple[float, ...], tuple[float, ...]]] = {
    "q": ((2.818e-6, 2.864e-6, 7.451e-6, 3.065e-5, 9.0e-5, 1.943e-4, 5.834e-4, 1.283e-3, 2.271e-3, 3.555e-3, 6.67e-3, 8.734e-3, 1.019e-2),
          (2.499e-7, 6.849e-7, 4.886e-6, 2.898e-5, 9.483e-5, 2.161e-4, 6.639e-4, 1.366e-3, 2.105e-3, 2.921e-3, 4.258e-3, 4.982e-3, 5.761e-3)),
    "t": ((210.7, 203.9, 211.3, 219.1, 227.0, 235.1, 249.4, 260.2, 268.5, 275.5, 283.1, 286.2, 290.5),
          (6.95, 10.88, 6.942, 5.179, 7.332, 9.291, 10.61, 10.5, 10.45, 11.01, 11.76, 12.18, 12.77)),
    "u": ((5.795, 9.586, 13.7, 14.42, 13.3, 11.55, 8.276, 5.913, 4.16, 2.706, 0.6906, -0.1105, -0.6422),
          (13.59, 14.46, 18.03, 19.32, 18.79, 17.29, 14.19, 11.85, 10.24, 9.081, 8.059, 7.77, 6.057)),
    "v": ((-0.01344, 0.002616, -0.1581, -0.2393, -0.1278, -0.0235, -0.03571, -0.05574, -0.06311, -0.009114, 0.1454, 0.3071, 0.3214),
          (5.149, 6.908, 9.596, 11.73, 12.5, 11.97, 9.874, 8.127, 6.985, 6.203, 5.679, 5.998, 5.047)),
    "w": ((-9.845e-6, -3.095e-5, -8.51e-5, -1.935e-4, -3.142e-4, -3.635e-4, -3.57e-4, -3.69e-4, -2.745e-4, 3.17e-4, 3.482e-3, 5.928e-3, 7.155e-3),
          (0.01244, 0.02865, 0.06249, 0.1015, 0.1379, 0.1696, 0.2148, 0.2378, 0.2526, 0.2589, 0.2442, 0.1965, 0.1155)),
    "z": ((2.015e5, 1.605e5, 1.364e5, 1.186e5, 1.044e5, 9.227e4, 7.226e4, 5.593e4, 4.207e4, 3.001e4, 1.438e4, 7447.0, 973.7),
          (3858.0, 4083.0, 4626.0, 4703.0, 4454.0, 4064.0, 3270.0, 2622.0, 2104.0, 1672.0, 1156.0, 976.1, 862.4)),
}
# fmt: on

_SYNTHETIC_FORCINGS = (
    "cos_latitude", "sin_latitude", "cos_longitude", "sin_longitude",
    "cos_julian_day", "sin_julian_day", "cos_local_time", "sin_local_time",
    "insolation", "cos_solar_zenith_angle",
)  # fmt: skip
_CONSTANT_FORCINGS = ("cos_latitude", "sin_latitude", "cos_longitude", "sin_longitude")
_SYNTHETIC_STATIC = ("lsm", "z", "sdor", "slor")

DEFAULT_SYNTHETIC_VARIABLES = sorted(
    [k for k in _SYNTHETIC_STATS if k != "cos_solar_zenith_angle"]
    + [f"{p}_{lev}" for p in _SYNTHETIC_PL_STATS for lev in _SYNTHETIC_LEVELS],
)

# Wavenumbers of the travelling waves, and the zonal profiles used as basis functions.
_SYNTHETIC_WAVENUMBERS = (1, 2, 3, 4, 5, 6, 8, 11)
_P2, _P1, _JET, _TROPICS = range(4)
_N_ZONAL = 4

# Per-family recipe: zonal profile coefficients (as a fraction of stdev; P1 is
# multiplied by a seasonal cycle), the fraction carried by waves, and the wave
# phase offset (so e.g. v and w are shifted against z/t like in real flow).
_SYNTHETIC_FAMILIES = {
    "temperature": {"zonal": {_P2: -0.85, _P1: 0.35}, "waves": 0.45, "phase": 0.2},
    "geopotential": {"zonal": {_P2: -0.8, _P1: 0.3}, "waves": 0.55, "phase": 0.0},
    "u": {"zonal": {_JET: 0.7}, "waves": 0.7, "phase": np.pi},
    "v": {"zonal": {}, "waves": 1.0, "phase": 0.5 * np.pi},
    "w": {"zonal": {}, "waves": 1.0, "phase": -0.5 * np.pi},
    "humidity": {"zonal": {_TROPICS: 0.8, _P2: -0.3}, "waves": 0.5, "phase": 0.4},
    "precip": {"zonal": {_TROPICS: 0.5}, "waves": 0.85, "phase": -0.5 * np.pi},
    "pressure": {"zonal": {_P2: 0.2}, "waves": 0.95, "phase": 0.0},
    "generic": {"zonal": {}, "waves": 1.0, "phase": 0.0},
}
_FAMILY_OF_PARAM = {
    "t": "temperature", "2t": "temperature", "2d": "temperature", "skt": "temperature",
    "z": "geopotential", "u": "u", "10u": "u", "v": "v", "10v": "v", "w": "w",
    "q": "humidity", "tcw": "humidity", "tp": "precip", "cp": "precip",
    "msl": "pressure", "sp": "pressure",
}  # fmt: skip
_NON_NEGATIVE = ("q", "tcw", "tp", "cp")

_SYNTHETIC_EPOCH = np.datetime64("2000-01-01T00:00:00")
_SYNTHETIC_CHUNK = 1 << 20  # grid points generated per matmul chunk


def _dataset_source(dataset_config: object) -> object:
    """Return the ``dataset`` entry of a reader's dataset_config, whatever its type."""
    if hasattr(dataset_config, "model_dump"):
        dataset_config = dataset_config.model_dump()
    dataset_config = _as_dict(dataset_config)
    if isinstance(dataset_config, dict):
        return dataset_config.get("dataset")
    return dataset_config


def is_synthetic_dataset(dataset_config: object) -> bool:
    """Return True if ``dataset_config`` (or its ``dataset`` entry) requests synthetic data."""
    source = _dataset_source(dataset_config)
    return isinstance(source, str) and source.strip().lower().startswith(SYNTHETIC_PREFIX)


def _octahedral_grid(grid: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (latitudes, longitudes, area weights) of an octahedral reduced Gaussian grid.

    Points are ordered north to south, each row starting at longitude 0, matching
    the ordering of ECMWF/anemoi datasets.
    """
    match = re.fullmatch(r"[oO](\d+)", grid.strip())
    if match is None:
        msg = f"Synthetic datasets only support octahedral grids (e.g. 'o96', 'o800'), got '{grid}'."
        raise ValueError(msg)
    n = int(match.group(1))

    # Gaussian latitudes: roots of the Legendre polynomial P_{2n}, by vectorised Newton iteration.
    nlat = 2 * n
    i = np.arange(1, n + 1)
    x = np.cos(np.pi * (i - 0.25) / (nlat + 0.5))
    for _ in range(100):
        p0, p1 = np.ones_like(x), x
        for k in range(2, nlat + 1):
            p0, p1 = p1, ((2 * k - 1) * x * p1 - (k - 1) * p0) / k
        dp = nlat * (x * p1 - p0) / (x * x - 1)
        dx = p1 / dp
        x -= dx
        if np.max(np.abs(dx)) < 1e-14:
            break
    weights = 2.0 / ((1 - x * x) * dp * dp)
    lat_north = np.degrees(np.arcsin(x))  # north hemisphere, pole to equator

    nlon_north = 4 * i + 16
    nlons = np.concatenate([nlon_north, nlon_north[::-1]])
    row_lat = np.concatenate([lat_north, -lat_north[::-1]])
    row_w = np.concatenate([weights, weights[::-1]]) / nlons

    lats = np.repeat(row_lat, nlons)
    row_start = np.repeat(np.cumsum(nlons) - nlons, nlons)
    lons = (np.arange(lats.size) - row_start) * np.repeat(360.0 / nlons, nlons)
    area = np.repeat(row_w, nlons)
    return lats, lons, area


def _parse_synthetic_date(value: object, frequency: datetime.timedelta, *, last: bool) -> np.datetime64:
    """Interpret start/end like open_dataset: years/months/days are inclusive ranges."""
    if isinstance(value, np.datetime64):
        return value.astype("datetime64[s]")
    if isinstance(value, datetime.datetime):
        return np.datetime64(value.replace(tzinfo=None), "s")
    if isinstance(value, datetime.date):
        value = value.isoformat()

    text = str(value).strip()
    digits = text.replace("-", "")
    if re.fullmatch(r"\d{4}-?\d{2}(-?\d{2})?|\d{4}", text):
        unit = {4: "Y", 6: "M", 8: "D"}[len(digits)]
        period = np.datetime64("-".join(filter(None, (digits[:4], digits[4:6], digits[6:8]))), unit)
    else:
        return np.datetime64(datetime.datetime.fromisoformat(text).replace(tzinfo=None), "s")
    first, stop = period.astype("datetime64[s]"), (period + 1).astype("datetime64[s]")
    if not last:
        return first
    return stop - np.timedelta64(int(frequency.total_seconds()), "s")


class SyntheticDataset(BaseAnemoiReader):
    """Reader generating random but physically plausible fields, without any dataset on disk.

    Exposes the same interface as :class:`NativeGridDataset` (analysis layout,
    single sequence) so it can be used anywhere a real reader is used.
    """

    def __init__(
        self,
        dataset_config: str | dict,
        start: datetime.datetime | int | str | None = None,
        end: datetime.datetime | int | str | None = None,
        sampling: dict | None = None,
    ) -> None:
        if hasattr(dataset_config, "model_dump"):
            dataset_config = dataset_config.model_dump()
        dataset_config = _as_dict(dataset_config)
        if isinstance(dataset_config, str):
            dataset_config = {"dataset": dataset_config}
        dataset_config = {k: _as_dict(v) for k, v in dataset_config.items()}

        self.grid = dataset_config["dataset"].strip()[len(SYNTHETIC_PREFIX) :].strip().lower()
        options = dict(dataset_config.get("synthetic") or {})
        self.ensemble = int(options.get("ensemble", 1))
        self.seed = int(options.get("seed", 0))

        variables = list(options.get("variables") or DEFAULT_SYNTHETIC_VARIABLES)
        if dataset_config.get("select"):
            select = list(dataset_config["select"])
            missing = set(select) - set(variables)
            if missing:
                msg = f"Synthetic dataset: cannot select unknown variables {sorted(missing)}."
                raise ValueError(msg)
            variables = [v for v in variables if v in select]
        if dataset_config.get("drop"):
            variables = [v for v in variables if v not in set(dataset_config["drop"])]
        self._variables = variables

        frequency = dataset_config.get("frequency") or "6h"
        frequency = getattr(frequency, "as_timedelta", frequency)
        self._frequency = frequency_to_timedelta(frequency)

        first = _parse_synthetic_date(start if start is not None else 2023, self._frequency, last=False)
        if end is None:
            end = start if start is not None else 2023
        last = _parse_synthetic_date(end, self._frequency, last=True)
        step = np.timedelta64(int(self._frequency.total_seconds()), "s")
        self._dates = np.arange(first, last + step, step)
        if self._dates.size == 0:
            msg = f"Synthetic dataset: no dates between start={start} and end={end}."
            raise ValueError(msg)

        self.data = self  # nothing on disk; callers using reader.data see the same interface
        self.default_sampling = sampling if sampling is not None else {"stride": 1}
        self._build_recipes()

    # ------------------------------------------------------------------
    # Grid & per-variable recipes
    # ------------------------------------------------------------------

    @cached_property
    def _grid(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return _octahedral_grid(self.grid)

    @property
    def latitudes(self) -> np.ndarray:
        return self._grid[0]

    @property
    def longitudes(self) -> np.ndarray:
        return self._grid[1]

    @staticmethod
    def _split_name(name: str) -> tuple[str, int | None]:
        match = re.fullmatch(r"(.+)_(\d+)", name)
        if match and name not in _SYNTHETIC_FORCINGS:
            return match.group(1), int(match.group(2))
        return name, None

    def _variable_stats(self, name: str) -> tuple[float, float, float, float]:
        if name in _SYNTHETIC_STATS:
            return _SYNTHETIC_STATS[name]
        param, level = self._split_name(name)
        if param in _SYNTHETIC_PL_STATS and level is not None:
            means, stds = _SYNTHETIC_PL_STATS[param]
            levels = np.log(_SYNTHETIC_LEVELS)
            mean = float(np.interp(np.log(level), levels, means))
            std = float(np.interp(np.log(level), levels, stds))
            lower = 0.0 if param in _NON_NEGATIVE else mean - 6 * std
            return mean, std, lower, mean + 6 * std
        LOGGER.warning("Synthetic dataset: no climatology for '%s', using N(0, 1).", name)
        return 0.0, 1.0, -6.0, 6.0

    def _build_recipes(self) -> None:
        """Pre-compute per-variable statistics, families and wave amplitudes/phases."""
        names = self._variables
        stats = np.array([self._variable_stats(v) for v in names], dtype=np.float64)
        self._stats = {"mean": stats[:, 0], "stdev": stats[:, 1], "minimum": stats[:, 2], "maximum": stats[:, 3]}

        n_waves = len(_SYNTHETIC_WAVENUMBERS)
        # Phases shared by all variables give coherent "weather systems" across variables/levels.
        shared = np.random.default_rng([self.seed, 0])
        base_phase = shared.uniform(0, 2 * np.pi, size=(n_waves, 2))
        self._mod_freq = shared.uniform(2 * np.pi / (30 * 24), 2 * np.pi / (8 * 24), size=(n_waves, 2))
        self._mod_phase = shared.uniform(0, 2 * np.pi, size=(n_waves, 2))
        # eastward phase speed ~10 m/s at the equator (rad/hour)
        self._omega = np.array(_SYNTHETIC_WAVENUMBERS, dtype=np.float64) * 10.0 * 3600 / 6.371e6

        n = len(names)
        self._zonal = np.zeros((n, _N_ZONAL))
        self._seasonal = np.zeros(n, dtype=bool)
        self._wave_amp = np.zeros((n, n_waves, 2))
        self._wave_phase = np.zeros((n, n_waves, 2))
        self._kind = []
        spectrum = np.array(_SYNTHETIC_WAVENUMBERS, dtype=np.float64) ** -0.8
        for j, name in enumerate(names):
            param, level = self._split_name(name)
            if name in _SYNTHETIC_FORCINGS:
                self._kind.append("forcing")
                continue
            if name in _SYNTHETIC_STATIC:
                self._kind.append("static")
                continue
            self._kind.append(param)
            family = _SYNTHETIC_FAMILIES[_FAMILY_OF_PARAM.get(param, "generic")]
            for k, coef in family["zonal"].items():
                self._zonal[j, k] = coef
            self._seasonal[j] = _P1 in family["zonal"]

            rng = np.random.default_rng([self.seed, zlib.crc32(name.encode())])
            amp = spectrum[:, None] * rng.rayleigh(1.0, size=(n_waves, 2))
            self._wave_amp[j] = family["waves"] * amp / np.sqrt((amp**2).sum())
            tilt = 0.15 * np.log(level / 500.0) if level else 0.0
            self._wave_phase[j] = base_phase + family["phase"] + tilt + rng.normal(0, 0.1, size=(n_waves, 2))

        self._stdev = self._stats["stdev"].copy()
        self._gen_mean = self._stats["mean"].copy()
        precip = np.isin(self._kind, ("tp", "cp"))
        self._gen_mean[precip] -= 0.5 * self._stdev[precip]  # shift so relu leaves dry areas
        self._non_negative = np.isin(self._kind, _NON_NEGATIVE)

        # Normalise basis functions to unit RMS over the sphere (1-D Gauss quadrature).
        mu, w = np.polynomial.legendre.leggauss(512)
        zonal = self._zonal_basis(mu)
        self._zonal_mean = np.average(zonal, axis=0, weights=w)
        self._basis_norm = 1.0 / np.sqrt(np.average((zonal - self._zonal_mean) ** 2, axis=0, weights=w))
        env = self._wave_envelopes(mu)
        self._wave_norm = 1.0 / np.sqrt(0.5 * np.average(env**2, axis=0, weights=w))

    @staticmethod
    def _zonal_basis(sinlat: np.ndarray) -> np.ndarray:
        lat = np.degrees(np.arcsin(sinlat))
        jet = np.exp(-(((np.abs(lat) - 45.0) / 12.0) ** 2))
        tropics = np.exp(-((lat / 15.0) ** 2))
        return np.stack([1.5 * sinlat**2 - 0.5, sinlat, jet, tropics], axis=-1)

    @staticmethod
    def _wave_envelopes(sinlat: np.ndarray) -> np.ndarray:
        """Latitudinal envelopes (symmetric, antisymmetric) for each wavenumber."""
        coslat = np.sqrt(np.clip(1 - sinlat**2, 0, None))
        env = [coslat ** min(m, 3) for m in _SYNTHETIC_WAVENUMBERS]
        return np.stack([e for m in env for e in (m, m * sinlat)], axis=-1)

    # ------------------------------------------------------------------
    # Field generation
    # ------------------------------------------------------------------

    def _coefficients(self, hours: float, member: int) -> np.ndarray:
        """Return the (K, V) matrix mapping basis functions to variables at a given time."""
        n_waves = len(_SYNTHETIC_WAVENUMBERS)
        doy = (hours / 24.0) % 365.25
        season = np.cos(2 * np.pi * (doy - 200.0) / 365.25)  # NH summer peak late July

        zonal = self._zonal.copy()
        zonal[:, _P1] *= np.where(self._seasonal, season, 0.0)
        zonal *= self._basis_norm

        phase = self._wave_phase + self._omega[None, :, None] * hours
        if member:
            phase = phase + np.random.default_rng([self.seed, 1, member]).normal(0, 0.3, size=(n_waves, 2))
        amp = self._wave_amp * (1 + 0.3 * np.sin(self._mod_freq * hours + self._mod_phase))
        amp = amp * self._wave_norm.reshape(n_waves, 2)
        # a cos(m lon - phase) = a cos(phase) cos(m lon) + a sin(phase) sin(m lon)
        waves = np.stack([amp * np.cos(phase), amp * np.sin(phase)], axis=-1)  # (V, waves, sym/anti, cos/sin)

        coeffs = np.concatenate([zonal, waves.reshape(len(self._variables), -1)], axis=1)
        return (coeffs * self._stdev[:, None]).T.astype(np.float32)

    def _basis(self, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
        """Return the (points, K) basis evaluated on a set of grid points."""
        sinlat = np.sin(np.radians(lat))
        zonal = self._zonal_basis(sinlat) - self._zonal_mean
        env = self._wave_envelopes(sinlat)
        lon_r = np.radians(lon)
        cols = [zonal]
        for i, m in enumerate(_SYNTHETIC_WAVENUMBERS):
            c, s = np.cos(m * lon_r), np.sin(m * lon_r)
            sym, anti = env[:, 2 * i], env[:, 2 * i + 1]
            cols.append(np.stack([sym * c, sym * s, anti * c, anti * s], axis=-1))
        return np.concatenate(cols, axis=1).astype(np.float32)

    def _orography(self, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
        """Time-independent field (unit RMS) defining continents and mountains."""
        rng = np.random.default_rng([self.seed, 2])
        basis = self._basis(lat, lon)[:, _N_ZONAL:]
        coeffs = rng.normal(size=basis.shape[1]) * np.repeat(np.array(_SYNTHETIC_WAVENUMBERS) ** -0.5, 4)
        coeffs *= np.repeat(self._wave_norm, 2) / np.sqrt((coeffs**2).sum())  # unit RMS
        field = basis @ coeffs.astype(np.float32)
        return field + 2.0 * (lat < -70)  # Antarctica

    def _fill_static_and_forcings(self, out: np.ndarray, lat: np.ndarray, lon: np.ndarray, dates: np.ndarray) -> None:
        """Overwrite static and forcing columns of ``out`` (dates, ens, points, vars)."""
        idx = {name: j for j, name in enumerate(self._variables)}
        lat_r, lon_r = np.radians(lat), np.radians(lon)

        if any(name in idx for name in _SYNTHETIC_STATIC):
            relief = self._orography(lat, lon) - 0.6
            land = relief > 0
            height = np.where(land, relief, 0.0)
            static = {
                "lsm": land.astype(np.float32),
                "z": 9.81 * 1500.0 * height,
                "sdor": 120.0 * height,
                "slor": 1e-4 + 0.02 * height,
            }
            for name, value in static.items():
                if name in idx:
                    out[..., idx[name]] = value
            if "sp" in idx:
                # surface pressure drops with orography (scale height of ~7.5 km)
                out[..., idx["sp"]] *= np.exp(-1500.0 * height / 7500.0)

        constant = {
            "cos_latitude": np.cos(lat_r), "sin_latitude": np.sin(lat_r),
            "cos_longitude": np.cos(lon_r), "sin_longitude": np.sin(lon_r),
        }  # fmt: skip
        for name, value in constant.items():
            if name in idx:
                out[..., idx[name]] = value

        for t, date in enumerate(dates):
            hours = (date - _SYNTHETIC_EPOCH) / np.timedelta64(1, "h")
            doy = (date - date.astype("datetime64[Y]")) / np.timedelta64(1, "D")
            year_frac = 2 * np.pi * doy / 365.25
            utc = hours % 24
            local = 2 * np.pi * ((utc + lon / 15.0) % 24) / 24.0
            decl = np.radians(23.44) * np.sin(2 * np.pi * (doy - 81) / 365.0)
            cosz = np.sin(lat_r) * np.sin(decl) + np.cos(lat_r) * np.cos(decl) * np.cos(local - np.pi)
            cosz = np.clip(cosz, 0.0, 1.0)
            timed = {
                "cos_julian_day": np.cos(year_frac), "sin_julian_day": np.sin(year_frac),
                "cos_local_time": np.cos(local), "sin_local_time": np.sin(local),
                "insolation": cosz, "cos_solar_zenith_angle": cosz,
            }  # fmt: skip
            for name, value in timed.items():
                if name in idx:
                    out[t, ..., idx[name]] = value

    def get_sample(
        self,
        sequence: int,
        positions: TimeIndices,
        grid_shard_indices: np.ndarray | slice | None = None,
    ) -> torch.Tensor:
        """Generate a sample of shape (dates, ensemble, gridpoints, variables)."""
        del sequence
        dates = self._dates[positions]
        dates = np.atleast_1d(dates)
        points = np.arange(self.grid_size)
        if grid_shard_indices is not None:
            points = points[grid_shard_indices]

        # Generated with numpy rather than torch: torch CPU matmuls (oneDNN/OpenMP)
        # can deadlock inside forked dataloader workers.
        n_vars = len(self._variables)
        out = np.empty((len(dates), self.ensemble, points.size, n_vars), dtype=np.float32)
        mean = self._gen_mean.astype(np.float32)
        coeffs = {
            (t, e): self._coefficients((d - _SYNTHETIC_EPOCH) / np.timedelta64(1, "h"), e)
            for t, d in enumerate(dates)
            for e in range(self.ensemble)
        }
        non_negative = np.flatnonzero(self._non_negative)

        for lo in range(0, points.size, _SYNTHETIC_CHUNK):
            chunk = points[lo : lo + _SYNTHETIC_CHUNK]
            lat, lon = self.latitudes[chunk], self.longitudes[chunk]
            basis = self._basis(lat, lon)
            for (t, e), c in coeffs.items():
                view = out[t, e, lo : lo + chunk.size]
                np.matmul(basis, c, out=view)
                view += mean
                view[:, non_negative] = np.maximum(view[:, non_negative], 0.0)
            self._fill_static_and_forcings(out[:, :, lo : lo + chunk.size], lat, lon, dates)

        idx = {name: j for j, name in enumerate(self._variables)}
        if "tp" in idx and "cp" in idx:
            np.minimum(out[..., idx["cp"]], out[..., idx["tp"]], out=out[..., idx["cp"]])
        return torch.from_numpy(out)

    # ------------------------------------------------------------------
    # Dataset properties
    # ------------------------------------------------------------------

    def sequence_length(self, sequence: int = 0) -> int:  # noqa: ARG002
        return len(self._dates)

    @property
    def dates(self) -> np.ndarray:
        return self._dates

    @property
    def grid_size(self) -> int:
        return int(self.latitudes.size)

    @property
    def statistics(self) -> dict:
        return {k: v.copy() for k, v in self._stats.items()}

    def statistics_tendencies(self, timestep: int | str | datetime.timedelta | None = None) -> dict | None:  # noqa: ARG002
        return None

    @property
    def variables(self) -> list[str]:
        return list(self._variables)

    @property
    def missing(self) -> set[int]:
        return set()

    @property
    def frequency(self) -> datetime.timedelta:
        return self._frequency

    @property
    def name_to_index(self) -> dict[str, int]:
        return {name: i for i, name in enumerate(self._variables)}

    @property
    def resolution(self) -> str:
        return self.grid.upper()

    @property
    def supporting_arrays(self) -> dict:
        return {"latitudes": self.latitudes, "longitudes": self.longitudes}

    @cached_property
    def cutout_mask(self) -> np.ndarray:
        err_msg = "Synthetic datasets do not have a cutout grid."
        raise ValueError(err_msg)

    @property
    def metadata(self) -> dict:
        variables_metadata = {}
        for name in self._variables:
            param, level = self._split_name(name)
            if name in _SYNTHETIC_FORCINGS:
                variables_metadata[name] = {
                    "computed_forcing": True,
                    "constant_in_time": name in _CONSTANT_FORCINGS,
                    "units": "dimensionless",
                }
            elif level is not None:
                variables_metadata[name] = {"mars": {"levtype": "pl", "param": param, "levelist": level}}
            else:
                variables_metadata[name] = {"mars": {"levtype": "sfc", "param": param}}
                if name in _SYNTHETIC_STATIC:
                    variables_metadata[name]["constant_in_time"] = True
            if param in ("tp", "cp"):
                variables_metadata[name].update({"process": "accumulation", "period": ["0h", frequency_to_string(self._frequency)]})

        # Like the data_request of a MARS dataset: anemoi-inference reads the grid from it (e.g. for GRIB output).
        pressure_levels = [list(self._split_name(n)) for n in self._variables if self._split_name(n)[1] is not None]
        surface = [n for n in self._variables if self._split_name(n)[1] is None and n not in _SYNTHETIC_FORCINGS]
        step = int(self._frequency.total_seconds() // 3600)
        data_request = {
            "grid": self.resolution,
            "area": [
                round(float(self.latitudes.max()), 3),
                round(float(self.longitudes.min()), 3),
                round(float(self.latitudes.min()), 3),
                round(float(self.longitudes.max()), 3),
            ],
            "param_level": {"pl": pressure_levels, "sfc": surface},
            "param_step": {"sfc": [[n, step] for n in surface if n in ("tp", "cp")]},
        }

        return {
            "version": "synthetic",
            "data_request": data_request,
            "name": f"synthetic-{self.grid}",
            "arguments": {"synthetic": self.grid, "seed": self.seed, "ensemble": self.ensemble},
            "frequency": frequency_to_string(self._frequency),
            "resolution": self.resolution,
            "variables": list(self._variables),
            "variables_metadata": variables_metadata,
            "shape": [len(self._dates), len(self._variables), self.ensemble, self.grid_size],
            "dtype": "float32",
            "start_date": str(self._dates[0]),
            "end_date": str(self._dates[-1]),
            "supporting_arrays": ["latitudes", "longitudes"],
        }

    def tree(self, prefix: str = "") -> Tree:
        tree = Tree(prefix + " 🎲 " + f"{self.__class__.__name__}")
        tree.add(f"Grid: {self.resolution} ({self.grid_size} points)")
        tree.add(f"Dates: {self._dates[0]} to {self._dates[-1]} ({len(self._dates)})")
        tree.add(f"Frequency: {self.frequency}")
        tree.add(f"Num variables: {len(self._variables)}")
        tree.add(f"Ensemble members: {self.ensemble}, seed: {self.seed}")
        return tree


class SyntheticGridNodes(BaseNodeBuilder):
    """Graph data nodes on the grid of a synthetic dataset (same signature as AnemoiDatasetNodes)."""

    def __init__(self, dataset: str | dict | DictConfig, name: str) -> None:
        source = _dataset_source(dataset)
        self.grid = source.strip()[len(SYNTHETIC_PREFIX) :].strip().lower()
        super().__init__(name)

    def get_coordinates(self) -> torch.Tensor:
        lats, lons, _ = _octahedral_grid(self.grid)
        return self.reshape_coords(lats, lons)


def create_dataset(dataset_config: dict, **_kwargs) -> BaseAnemoiReader:
    """Factory function to create a data reader based on the dataset configuration."""
    dataset_config = _normalize_reader_config(dataset_config)

    if is_synthetic_dataset(dataset_config["dataset_config"]):
        if dataset_config.get("trajectory") is not None:
            msg = "Synthetic datasets do not support the 'trajectory' option."
            raise ValueError(msg)
        LOGGER.info("Creating SyntheticDataset...")
        return SyntheticDataset(
            dataset_config=dataset_config["dataset_config"],
            start=dataset_config.get("start"),
            end=dataset_config.get("end"),
        )

    trajectory_config = _as_dict(dataset_config.pop("trajectory", None))

    if trajectory_config is not None:
        sampling = trajectory_config.get("sampling") if isinstance(trajectory_config, dict) else None
        LOGGER.info("Creating TrajectoryDataset...")
        return TrajectoryDataset(**dataset_config, sampling=_as_dict(sampling))

    LOGGER.info("Creating NativeGridDataset...")
    return NativeGridDataset(**dataset_config)
