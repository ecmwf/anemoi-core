# (C) Copyright 2024-2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import datetime
from typing import Protocol

import numpy as np

from anemoi.models.data.sample import BaseSample
from anemoi.training.data.usable_indices import ReaderAnchors
from anemoi.training.utils.time_indices import TimeIndices


class DataReader(Protocol):
    """What the data pipeline needs from a data reader.

    :class:`~anemoi.training.data.readers.base.BaseAnemoiReader` implements it for
    ``anemoi.datasets`` stores. ``MultiDataset``, anchor computation,
    the datamodule and the training methods depend on this :class:`DataReader` only.
    """

    #: Configured anchor sampling (``{"stride": int | None}``), or ``None`` if not configured.
    sampling: dict | None

    #: Kind of sample the reader produces (``"gridded"``, ``"tabular"``).
    data_type: str

    @property
    def frequency(self) -> datetime.timedelta:
        """Spacing between consecutive positions of a sequence."""

    @property
    def variables(self) -> list[str]:
        """Variable names, in data order."""

    @property
    def name_to_index(self) -> dict[str, int]:
        """Mapping from variable name to its index in the data."""

    @property
    def resolution(self) -> str:
        """Dataset resolution."""

    @property
    def statistics(self) -> dict:
        """Per-variable statistics."""

    def statistics_tendencies(self, timestep: int | str | datetime.timedelta) -> dict | None:
        """Per-variable tendency statistics at ``timestep``, or ``None`` if unavailable."""

    @property
    def metadata(self) -> dict:
        """Dataset metadata."""

    @property
    def supporting_arrays(self) -> dict:
        """Supporting arrays stored with the dataset."""

    def valid_anchors(self, relative_indices: list[int] | np.ndarray) -> ReaderAnchors:
        """Return the anchors at which every ``position + relative_index`` can be read."""

    def set_reader_group_info(self, reader_group_rank: int, reader_group_size: int) -> None:
        """Set reader communication group information (called by DDPGroupStrategy)."""

    def get_sample(self, sequence: int, positions: TimeIndices) -> BaseSample:
        """Return the sample at ``positions`` of ``sequence``."""

    def measure_from_reference(self, sample: BaseSample, reference: int, positions: TimeIndices) -> BaseSample:
        """Return ``sample`` with its point times measured from the reference position."""
