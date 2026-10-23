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
from abc import ABC
from abc import abstractmethod
from functools import cached_property
from typing import ClassVar

import numpy as np
from rich.console import Console
from rich.tree import Tree

from anemoi.datasets import open_dataset
from anemoi.models.data import TensorLayout
from anemoi.models.data.sample import BaseSample
from anemoi.training.data.readers.utils import _normalize_dataset_config
from anemoi.training.data.usable_indices import ReaderAnchors
from anemoi.training.data.usable_indices import get_usable_indices
from anemoi.training.utils.time_indices import TimeIndices

LOGGER = logging.getLogger(__name__)


class BaseAnemoiReader(ABC):
    """Generic anemoi data reader."""

    #: Kind of sample the reader produces, as registered in ``anemoi.models.data.sample.sample_registry``.
    data_type: ClassVar[str]

    #: Configured anchor sampling (``{"stride": int | None}``), or ``None`` if not configured.
    sampling: dict | None = None

    def __init__(
        self,
        dataset_config: str | dict,
        start: datetime.datetime | int | None = None,
        end: datetime.datetime | int | None = None,
    ) -> None:
        """Initialize Anemoi data reader."""
        source = _normalize_dataset_config(dataset_config)
        self.data = open_dataset(self._open_options(source, start, end))

        # lazy init reader group info (will be set by DDPGroupStrategy)
        self.reader_group_rank = 0
        self.reader_group_size = 1

    def _open_options(
        self,
        source: dict,
        start: datetime.datetime | int | None,
        end: datetime.datetime | int | None,
    ) -> dict:
        """Return the ``open_dataset`` options for ``source`` restricted to ``[start, end]``."""
        # start/end must sit next to window/frequency for tabular datasets
        return source | {"start": start, "end": end}

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
    @abstractmethod
    def grid_size(self) -> int | None:
        """Return the number of grid points, or ``None`` when the reader has no static grid."""

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
        timestep: int | str | datetime.timedelta,
    ) -> dict | None:
        """Return dataset tendency statistics at ``timestep``, or ``None`` if the dataset has none."""
        try:
            return self.data.statistics_tendencies(timestep)
        except (KeyError, AttributeError, TypeError):
            return None

    @property
    @abstractmethod
    def layout(self) -> TensorLayout:
        """Layout of one sample, independent of whether its coordinates change."""

    @property
    @abstractmethod
    def supporting_arrays(self) -> dict:
        """Return dataset supporting arrays."""

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
        """Return the mapping from variable name to its index in the data."""
        return self.data.name_to_index

    @property
    def resolution(self) -> str:
        """Return dataset resolution."""
        return self.data.resolution

    def set_reader_group_info(self, reader_group_rank: int, reader_group_size: int) -> None:
        """Set reader communication group information (called by DDPGroupStrategy).

        Parameters
        ----------
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

    def measure_from_reference(self, sample: BaseSample, reference: int, positions: TimeIndices) -> BaseSample:
        """Return ``sample`` with its point times measured from the sample's reference time.

        ``reference`` is the position of the reference time (relative offset 0) and ``positions``
        are the positions the sample was read from. Gridded samples carry no point times.
        """
        del reference, positions
        return sample

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
