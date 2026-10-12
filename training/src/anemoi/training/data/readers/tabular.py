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
from rich.tree import Tree

from anemoi.models.data import TensorLayout
from anemoi.models.data.sample import TabularSample
from anemoi.models.data.sample import create_batched_struct
from anemoi.training.data.readers.base import BaseAnemoiReader
from anemoi.training.data.readers.utils import _to_local_window_shard_data
from anemoi.training.utils.time_indices import TimeIndices

LOGGER = logging.getLogger(__name__)


class TabularDataReader(BaseAnemoiReader):
    """Observation dataset reader (e.g. from tabular zarrs).

    Each sample is built from a single round-trip ``self.data[time_indices, ...]``
    that returns an object exposing ``data``, ``latitudes``, ``longitudes``,
    ``timedeltas`` and ``boundaries``. The boundaries (``tuple[slice, ...]``)
    encode the per-time split of the flat ``N`` axis and travel through
    :attr:`Batch.metadata` rather than being moved to device.
    """

    data_type = "tabular"

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

    def statistics_tendencies(self, timestep: int | str | datetime.timedelta) -> None:
        """Observation datasets do not have tendency statistics."""
        del timestep
        return

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
            data_type=self.data_type,
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

    def measure_from_reference(self, sample: TabularSample, reference: int, positions: TimeIndices) -> TabularSample:
        """Return ``sample`` with its timedeltas measured from the reference time, not each window's date.

        Each window's timedeltas are offsets from that window's own date. Adding the window's
        offset from the reference date puts all windows on one time axis, so a point in an
        earlier window has a more negative timedelta than one in a later window.
        """
        dates = np.asarray(self.dates)
        window_dates = dates[np.atleast_1d(np.arange(len(dates))[positions])]
        window_offsets = (window_dates - dates[reference]) / np.timedelta64(1, "s")
        return sample.with_time_offsets(window_offsets.tolist())

    def tree(self, prefix: str = "") -> Tree:
        tree = super().tree(prefix)
        if hasattr(self.data, "window"):
            tree.add(f"Window: {self.data.window}")
        return tree
