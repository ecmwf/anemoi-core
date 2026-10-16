# (C) Copyright 2024-2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import datetime
import logging

import numpy as np
import torch
from einops import rearrange
from rich.tree import Tree

from anemoi.training.data.readers.gridded import GriddedDataReader
from anemoi.training.data.usable_indices import ReaderAnchors
from anemoi.training.data.usable_indices import get_usable_indices
from anemoi.training.utils.time_indices import TimeIndices
from anemoi.training.utils.time_indices import normalize_time_indices

LOGGER = logging.getLogger(__name__)


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

    def __init__(
        self,
        dataset_config: str | dict,
        start: datetime.datetime | int | None = None,
        end: datetime.datetime | int | None = None,
        sampling: dict | None = None,
    ) -> None:
        super().__init__(dataset_config, start=start, end=end)
        self.sampling = sampling if sampling is not None else {"stride": None}

    def _open_options(
        self,
        source: dict,
        start: datetime.datetime | int | None,
        end: datetime.datetime | int | None,
    ) -> dict:
        """Return the ``open_dataset`` options, filtering by initialisation date."""
        if source.get("frequency") is not None:
            msg = (
                f"{self.__class__.__name__} does not accept a 'frequency' in dataset_config. "
                "The step frequency is read directly from the dataset. "
                "Set data.frequency: null in your config."
            )
            raise ValueError(msg)

        # Trajectory datasets filter by initialisation date; passing start/end
        # would trigger access to .dates, which they do not have.
        return source | {"base_start": start, "base_end": end}

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
        if freq is None:
            msg = (
                f"Cannot determine step frequency: data.step_frequency is None for dataset {self.data}. "
                "Ensure that the dataset configuration includes a valid step_frequency (e.g. '6H')."
            )
            raise ValueError(msg)
        return freq

    def get_data(self, sequence: int, positions: TimeIndices) -> torch.Tensor:
        """Return steps ``positions`` of initialisation ``sequence`` as ``(steps, ensemble, grid, variables)``."""
        if isinstance(positions, slice):
            positions = range(*positions.indices(self.sequence_length(sequence)))
        positions = normalize_time_indices(np.asarray(positions).tolist())

        # data[sequence] -> (variables, ensembles, steps, cells)
        if isinstance(positions, slice):
            # Read only the requested steps instead of the whole forecast; with per-step chunking
            # this avoids reading and decompressing unused steps. Only slices take this path: a list
            # index on the step axis is not applied correctly by anemoi-datasets.
            x = self.data[sequence, :, :, positions, :]
        else:
            x = self.data[sequence]
            x = x[:, :, positions, :]

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
