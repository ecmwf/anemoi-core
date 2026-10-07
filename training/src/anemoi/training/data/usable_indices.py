# (C) Copyright 2024-2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import logging
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from anemoi.training.data.data_reader import BaseAnemoiReader

LOGGER = logging.getLogger(__name__)

# Anchors are matched on (base date, valid time) in seconds; base is 0 for single-sequence readers.
_KEY = np.dtype([("base", np.int64), ("time", np.int64)])


@dataclass(frozen=True)
class ReaderAnchors:
    """Valid anchors of one reader, keyed by time.

    Parameters
    ----------
    times : np.ndarray
        ``(n,)`` valid time of the anchor (the time at relative offset 0).
    sequences : np.ndarray
        ``(n,)`` reader sequence of each anchor.
    positions : np.ndarray
        ``(n,)`` position within the sequence of each anchor.
    base_dates : np.ndarray, optional
        ``(n,)`` forecast initialisation of each anchor, for trajectory readers.
        Anchors of readers without base dates match on ``times`` only.
    """

    times: np.ndarray
    sequences: np.ndarray
    positions: np.ndarray
    base_dates: np.ndarray | None = None

    def __len__(self) -> int:
        return len(self.times)

    def keys(self) -> np.ndarray:
        return _keys(self.times, self.base_dates)


@dataclass(frozen=True)
class Anchors:
    """Anchors shared by all readers, sorted by base date then valid time.

    ``rows[name][i]`` is the ``(sequence, position)`` reader ``name`` loads for sample ``i``.
    """

    times: np.ndarray
    base_dates: np.ndarray | None
    rows: dict[str, np.ndarray]

    def __len__(self) -> int:
        return len(self.times)


def _keys(times: np.ndarray, base_dates: np.ndarray | None = None) -> np.ndarray:
    keys = np.empty(len(times), dtype=_KEY)
    keys["time"] = np.asarray(times).astype("datetime64[s]").astype(np.int64)
    keys["base"] = 0 if base_dates is None else np.asarray(base_dates).astype("datetime64[s]").astype(np.int64)
    return keys


def _lookup(table: np.ndarray, query: np.ndarray) -> np.ndarray:
    """Return the index of each ``query`` key in ``table``, or -1 where it is absent."""
    order = np.argsort(table, order=("base", "time"))
    sorted_table = table[order]
    pos = np.searchsorted(sorted_table, query)
    clipped = np.minimum(pos, len(sorted_table) - 1)
    found = (pos < len(sorted_table)) & (sorted_table[clipped] == query)
    return np.where(found, order[clipped], -1)


def _query_for(table: ReaderAnchors, keys: np.ndarray) -> np.ndarray:
    if table.base_dates is not None:
        return keys
    query = keys.copy()
    query["base"] = 0
    return query


def _stride_steps(
    stride: int | None,
    data_readers: dict[str, "BaseAnemoiReader"],
    relative_date_indices: dict[str, np.ndarray | list[int]],
) -> tuple[int, int]:
    """Return ``(stride, grid)``: the stride in steps of the shared anchor grid of ``grid`` seconds."""
    seconds = {name: int(reader.frequency.total_seconds()) for name, reader in data_readers.items()}
    grid = math.lcm(*seconds.values())
    if stride is None:
        # non-overlapping: the next window starts after the widest window of any reader
        span = max(
            (int(np.max(relative_date_indices[name])) - int(np.min(relative_date_indices[name]))) * seconds[name]
            for name in data_readers
        )
        return span // grid + 1, grid
    return stride, grid


def _apply_stride(keys: np.ndarray, stride: int, grid: int) -> np.ndarray:
    """Keep every ``stride``-th grid step per base date, counted from its first anchor."""
    group_start = np.r_[True, keys["base"][1:] != keys["base"][:-1]]
    first = np.maximum.accumulate(np.where(group_start, np.arange(len(keys)), 0))
    return keys[(keys["time"] - keys["time"][first]) % (stride * grid) == 0]


def compute_valid_anchors(
    data_readers: dict[str, "BaseAnemoiReader"],
    relative_date_indices: dict[str, np.ndarray | list[int]],
    stride: int | None = 1,
) -> Anchors:
    """Return the anchors every reader can sample, aligned by time.

    Each reader lists its valid anchors as times (plus base dates for trajectory
    readers). An anchor is kept when every reader has it: readers with base dates
    must agree on ``(base date, time)``, the others on ``time``. Readers may have
    different frequencies and date ranges; each one gets its own
    ``(sequence, position)`` for every shared anchor.

    Parameters
    ----------
    data_readers : dict[str, BaseAnemoiReader]
        Mapping of dataset name to data reader.
    relative_date_indices : dict[str, np.ndarray | list[int]]
        Relative offsets (in each reader's own positions) requested for each reader.
    stride : int | None
        Keep every ``stride``-th step of the shared anchor grid per base date;
        ``None`` keeps non-overlapping windows.

    Returns
    -------
    Anchors
        The shared anchors and each reader's ``(sequence, position)`` rows.
    """
    if stride is not None and stride < 1:
        msg = f"sampling stride must be >= 1, got {stride}."
        raise ValueError(msg)

    tables: dict[str, ReaderAnchors] = {}
    for name, reader in data_readers.items():
        table = reader.valid_anchors(relative_date_indices[name])
        if len(table) == 0:
            msg = f"No valid anchors found for data reader '{name}': {reader}"
            raise ValueError(msg)
        LOGGER.info("Data reader '%s' has %d valid anchors", name, len(table))
        tables[name] = table

    with_base = [name for name, table in tables.items() if table.base_dates is not None]
    reference = tables[with_base[0] if with_base else next(iter(tables))]
    keys = np.sort(reference.keys(), order=("base", "time"))
    for table in tables.values():
        keys = keys[_lookup(table.keys(), _query_for(table, keys)) >= 0]

    if len(keys) == 0:
        msg = "No valid anchors found after intersection across all datasets."
        raise ValueError(msg)

    if stride != 1:
        keys = _apply_stride(keys, *_stride_steps(stride, data_readers, relative_date_indices))

    rows = {}
    for name, table in tables.items():
        index = _lookup(table.keys(), _query_for(table, keys))
        rows[name] = np.stack([table.sequences[index], table.positions[index]], axis=1).astype(np.int64)

    LOGGER.info("MultiDataset has %d valid anchors after intersection.", len(keys))
    return Anchors(
        times=keys["time"].astype("datetime64[s]"),
        base_dates=keys["base"].astype("datetime64[s]") if with_base else None,
        rows=rows,
    )


def get_usable_indices(
    missing_indices: set[int],
    series_length: int,
    relative_indices: np.ndarray | list[int],
) -> np.ndarray:
    """Get the usable indices of a series with missing indices.

    Parameters
    ----------
    missing_indices : set[int]
        Set of missing indices in the series.
    series_length : int
        Length of the series.
    relative_indices: np.ndarray | list[int]
        Array of relative indices requested at each index i.

    Returns
    -------
    usable_indices : np.array
        Indices ``i`` such that every ``i + relative_index`` is in range and not missing.
    """
    relative_indices = np.asarray(relative_indices, dtype=np.int64)
    start = max(0, -int(relative_indices.min()))
    stop = min(series_length, series_length - int(relative_indices.max()))
    usable_indices = np.arange(start, max(start, stop), dtype=np.int64)

    missing = [i for i in missing_indices if 0 <= i < series_length]
    if missing and usable_indices.size:
        is_missing = np.zeros(series_length, dtype=bool)
        is_missing[missing] = True
        reads_missing = is_missing[usable_indices[:, None] + relative_indices[None, :]].any(axis=1)
        usable_indices = usable_indices[~reads_missing]

    return usable_indices
