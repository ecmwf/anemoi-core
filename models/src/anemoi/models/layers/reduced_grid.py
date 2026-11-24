# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Reduced grids (latitude rows of different lengths) and their neighbourhood attention rule."""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from dataclasses import field
from functools import cached_property

import numpy as np
import torch
from torch import Tensor

# Coordinates are compared with this tolerance, in radians (about 0.006 degrees).
_COORD_ATOL = 1e-4


@dataclass(frozen=True)
class ReducedGrid:
    """A grid of latitude rows, each with its own number of equally spaced points.

    Rows run north to south, points eastwards. Point ``j`` of a row of ``n`` points is at longitude
    ``360 * (j + s / 2) / n`` degrees, with shift ``s`` 0 or 1. O and N grids have no shifts; HEALPix
    in ring ordering shifts some rings.

    ``row_latitudes`` (degrees) are only used to match the rows of two grids in cross attention.
    ``rows_beyond`` is the number of rows of the whole grid north and south of this one: ``(0, 0)``
    for a whole grid, more for a band of rows (see :meth:`rows`).
    """

    row_lengths: tuple[int, ...]
    row_latitudes: tuple[float, ...] | None = field(default=None, compare=False)
    row_shifts: tuple[int, ...] | None = None
    rows_beyond: tuple[int, int] = (0, 0)

    @classmethod
    def octahedral(cls, n: int) -> ReducedGrid:
        """The octahedral reduced Gaussian grid O``n``: rows of 20, 24, 28, ... points from each pole, at the Gaussian latitudes."""
        north = [4 * i + 20 for i in range(n)]
        zeros, _ = np.polynomial.legendre.leggauss(2 * n)
        latitudes = np.degrees(np.arcsin(zeros))[::-1]
        return cls(tuple(north + north[::-1]), tuple(float(x) for x in latitudes))

    @classmethod
    def healpix(cls, nside: int) -> ReducedGrid:
        """The HEALPix grid ``nside`` in ring ordering: ``4 * nside - 1`` rings, ``12 * nside**2`` points."""
        lengths, shifts, heights = [], [], []
        for ring in range(1, 4 * nside):
            if ring < nside:
                lengths.append(4 * ring)
                shifts.append(1)
                heights.append(1 - ring**2 / (3 * nside**2))
            elif ring <= 3 * nside:
                lengths.append(4 * nside)
                shifts.append(1 if (ring - nside) % 2 == 0 else 0)
                heights.append(4 / 3 - 2 * ring / (3 * nside))
            else:
                mirror = 4 * nside - ring
                lengths.append(4 * mirror)
                shifts.append(1)
                heights.append(mirror**2 / (3 * nside**2) - 1)
        latitudes = np.degrees(np.arcsin(np.clip(heights, -1.0, 1.0)))
        return cls(tuple(lengths), tuple(float(x) for x in latitudes), tuple(shifts))

    @classmethod
    def from_coords(cls, coords: Tensor) -> ReducedGrid:
        """Work out the grid from point coordinates in grid order.

        Parameters
        ----------
        coords : Tensor
            Latitude and longitude in radians, ``(num_points, 2)``.

        Returns
        -------
        ReducedGrid
            The grid.

        Raises
        ------
        ValueError
            If the points are not stored row by row as described in :class:`ReducedGrid`.
        """
        coords = coords.detach().to(dtype=torch.float64, device="cpu")
        lat, lon = coords[:, 0], torch.remainder(coords[:, 1], 2 * math.pi)
        new_row = torch.ones_like(lat, dtype=torch.bool)
        new_row[1:] = (lat[1:] - lat[:-1]).abs() > _COORD_ATOL
        starts = torch.nonzero(new_row).flatten().tolist() + [lat.numel()]
        row_lengths = tuple(b - a for a, b in zip(starts[:-1], starts[1:]))

        row_lat = lat[starts[:-1]]
        if len(row_lengths) > 1 and not bool((row_lat[1:] < row_lat[:-1]).all()):
            raise ValueError("Rows must run from north to south, each row at a single latitude.")
        shifts = []
        for start, n in zip(starts[:-1], row_lengths):
            spacing = 2 * math.pi / n
            shift = round(2 * float(lon[start]) / spacing)
            expected = (torch.arange(n, dtype=torch.float64) + shift / 2) * spacing
            if shift not in (0, 1) or not torch.allclose(lon[start : start + n], expected, atol=_COORD_ATOL):
                raise ValueError(
                    f"The {n} points of the row starting at point {start} are not equally spaced eastwards from "
                    "longitude 0 or from half a spacing."
                )
            shifts.append(shift)
        return cls(
            row_lengths,
            tuple(float(x) for x in torch.rad2deg(row_lat)),
            tuple(shifts) if any(shifts) else None,
        )

    @classmethod
    def from_unordered_coords(cls, coords: Tensor) -> tuple[ReducedGrid, Tensor]:
        """Like :meth:`from_coords` for points in any order, e.g. HEALPix in nested ordering.

        Parameters
        ----------
        coords : Tensor
            Latitude and longitude in radians, ``(num_points, 2)``.

        Returns
        -------
        tuple[ReducedGrid, Tensor]
            The grid, and ``order`` such that grid point ``i`` is ``coords[order[i]]``.
        """
        coords = coords.detach().to(dtype=torch.float64, device="cpu")
        lat_key = torch.round(coords[:, 0] / _COORD_ATOL)
        lon = torch.remainder(coords[:, 1], 2 * math.pi)
        order = torch.argsort(lon, stable=True)
        order = order[torch.argsort(-lat_key[order], stable=True)]
        return cls.from_coords(coords[order]), order

    @property
    def num_rows(self) -> int:
        return len(self.row_lengths)

    @property
    def num_points(self) -> int:
        return sum(self.row_lengths)

    @property
    def shifts(self) -> tuple[int, ...]:
        """Shift of every row in half spacings, 0 or 1."""
        return self.row_shifts if self.row_shifts is not None else (0,) * self.num_rows

    @property
    def is_shifted(self) -> bool:
        return any(self.shifts)

    @cached_property
    def row_starts(self) -> Tensor:
        """Index of the first point of each row, plus the total number of points at the end."""
        return torch.tensor([0, *self.row_lengths]).cumsum(0)

    @cached_property
    def rows_and_positions(self) -> tuple[Tensor, Tensor]:
        """Row of every point and its position within the row."""
        lengths = torch.tensor(self.row_lengths)
        rows = torch.repeat_interleave(torch.arange(self.num_rows), lengths)
        positions = torch.arange(self.num_points) - self.row_starts[rows]
        return rows, positions

    @cached_property
    def coords(self) -> Tensor:
        """Latitude and longitude of every point in radians, ``(num_points, 2)``."""
        if self.row_latitudes is None:
            raise ValueError("The coordinates of a grid need the latitudes of its rows.")
        rows, positions = self.rows_and_positions
        lat = torch.deg2rad(torch.tensor(self.row_latitudes, dtype=torch.float64))[rows]
        lengths = torch.tensor(self.row_lengths, dtype=torch.float64)[rows]
        shifts = torch.tensor(self.shifts, dtype=torch.float64)[rows]
        lon = 2 * math.pi * (positions + shifts / 2) / lengths
        return torch.stack([lat, lon], dim=1)

    def rows(self, start: int, stop: int) -> ReducedGrid:
        """The grid of rows ``start`` to ``stop - 1``."""
        north, south = self.rows_beyond
        return ReducedGrid(
            self.row_lengths[start:stop],
            None if self.row_latitudes is None else self.row_latitudes[start:stop],
            None if self.row_shifts is None else self.row_shifts[start:stop],
            (north + start, south + self.num_rows - stop),
        )

    def nearest_rows(self, other: ReducedGrid) -> Tensor:
        """For each row of this grid, the row of ``other`` nearest in latitude."""
        if self.row_latitudes is None or other.row_latitudes is None:
            raise ValueError("Matching the rows of two grids needs the latitudes of both.")
        mine = torch.tensor(self.row_latitudes, dtype=torch.float64)
        theirs = torch.tensor(other.row_latitudes, dtype=torch.float64)
        return (mine[:, None] - theirs[None, :]).abs().argmin(dim=1)


def matching_position(position, length, other_length, shift=0, other_shift=0):
    """Position in a row of ``other_length`` points nearest in longitude to ``position`` in a row of ``length``.

    Shifts are in half spacings. Works on ints and integer tensors; halfway cases round up.
    """
    return ((2 * position + shift) * other_length - other_shift * length + length) // (2 * length)


def over_the_pole(centre_row, row, num_rows: int, rows_beyond: tuple[int, int], radius: int):
    """Whether ``row`` is within ``radius`` rows of ``centre_row`` going over the nearer pole.

    Rows ``a`` and ``b`` rows from a pole are about ``a + b + 1`` row spacings apart over it. A row in
    reach is reached all the way round. Works on ints and integer tensors.
    """
    north, south = rows_beyond
    from_north = 2 * north + centre_row + row
    from_south = 2 * (south + num_rows - 1) - centre_row - row
    return (from_north < radius) | (from_south < radius)


class ReducedGridCrossNeighbourhoodMask:
    """Which keys of one reduced grid a query on another reduced grid attends to.

    A query is centred on the key row nearest its latitude. It attends to the key rows at most
    ``kernel_size[0] // 2`` rows away and, in each, to the points within ``kernel_size[1] // 2`` of
    the nearest longitude, going round the globe. A row no longer than ``kernel_size[1]``, or reached
    over the pole, is taken whole. Called with broadcasting query and key indices, it returns the mask.
    """

    def __init__(
        self,
        query_grid: ReducedGrid,
        key_grid: ReducedGrid,
        kernel_size: tuple[int, int],
        row_map: Tensor | None = None,
    ) -> None:
        kernel_h, kernel_w = kernel_size
        if kernel_h % 2 == 0 or kernel_w % 2 == 0 or kernel_h <= 0 or kernel_w <= 0:
            raise ValueError(f"kernel_size entries must be positive and odd, got {kernel_size}.")
        self.kernel_size = (kernel_h, kernel_w)
        self.num_k_rows = key_grid.num_rows
        self.rows_beyond = key_grid.rows_beyond
        self.row_map = query_grid.nearest_rows(key_grid) if row_map is None else row_map
        self.q_rows, self.q_positions = query_grid.rows_and_positions
        self.q_lengths = torch.tensor(query_grid.row_lengths)
        self.q_shifts = torch.tensor(query_grid.shifts)
        self.k_rows, self.k_positions = key_grid.rows_and_positions
        self.k_lengths = torch.tensor(key_grid.row_lengths)
        self.k_shifts = torch.tensor(key_grid.shifts)

    def to(self, device: torch.device) -> ReducedGridCrossNeighbourhoodMask:
        """Copy with the lookup tables on ``device``."""
        moved = copy.copy(self)
        for name in (
            "row_map",
            "q_rows",
            "q_positions",
            "q_lengths",
            "q_shifts",
            "k_rows",
            "k_positions",
            "k_lengths",
            "k_shifts",
        ):
            setattr(moved, name, getattr(self, name).to(device))
        return moved

    def _whole_ring(self, centre_row, k_row, k_length):
        """Whether key row ``k_row`` is taken whole by queries centred on ``centre_row``."""
        kernel_h, kernel_w = self.kernel_size
        short = kernel_w // 2 >= k_length // 2
        return short | over_the_pole(centre_row, k_row, self.num_k_rows, self.rows_beyond, kernel_h // 2)

    def __call__(self, q_idx: Tensor, kv_idx: Tensor) -> Tensor:
        kernel_h, kernel_w = self.kernel_size
        q_row, k_row = self.q_rows[q_idx], self.k_rows[kv_idx]
        centre_row = self.row_map[q_row]
        in_rows = (k_row - centre_row).abs() <= kernel_h // 2

        k_length = self.k_lengths[k_row]
        centre = matching_position(
            self.q_positions[q_idx], self.q_lengths[q_row], k_length, self.q_shifts[q_row], self.k_shifts[k_row]
        )
        offset = torch.remainder(self.k_positions[kv_idx] - centre, k_length)
        in_cols = torch.minimum(offset, k_length - offset) <= kernel_w // 2
        in_cols = in_cols | over_the_pole(centre_row, k_row, self.num_k_rows, self.rows_beyond, kernel_h // 2)
        return in_rows & in_cols

    def _key_rows(self, centre_rows: Tensor) -> tuple[Tensor, Tensor]:
        """The ``kernel_size[0]`` rows around each centre row, clamped to the grid, and which are inside it."""
        radius = self.kernel_size[0] // 2
        k_rows = centre_rows[..., None] + torch.arange(-radius, radius + 1)
        inside = (k_rows >= 0) & (k_rows < self.num_k_rows)
        return k_rows.clamp(0, self.num_k_rows - 1), inside

    def keys_per_query(self) -> Tensor:
        """Number of keys each query attends to, in query grid order."""
        kernel_w = self.kernel_size[1]
        k_rows, inside = self._key_rows(self.row_map)
        k_length = self.k_lengths[k_rows]
        per_row = torch.where(self._whole_ring(self.row_map[:, None], k_rows, k_length), k_length, kernel_w)
        return torch.repeat_interleave((per_row * inside).sum(1), self.q_lengths)

    def times_attended(self) -> Tensor:
        """Number of queries attending to each key, in key grid order, without building the mask.

        Each query covers one run of points per key row. A run adds one at its start and takes one away
        after its end; a running sum then gives the counts.
        """
        kernel_w = self.kernel_size[1]
        half_w = kernel_w // 2
        k_starts = torch.cat([torch.zeros(1, dtype=torch.long), self.k_lengths.cumsum(0)])
        changes = torch.zeros(int(k_starts[-1]) + 1, dtype=torch.long)
        for q_row in range(len(self.q_lengths)):
            centre_row = self.row_map[q_row]
            k_rows, inside = self._key_rows(centre_row)
            k_rows = k_rows[inside][:, None]
            k_length = self.k_lengths[k_rows]
            row_start = k_starts[k_rows]
            q_length = self.q_lengths[q_row]
            centre = matching_position(
                torch.arange(int(q_length))[None, :], q_length, k_length, self.q_shifts[q_row], self.k_shifts[k_rows]
            )
            # A run starts at `first` and ends just before `first + kernel_w`, wrapping past the row's end.
            whole_row = self._whole_ring(centre_row, k_rows, k_length)
            first = torch.where(whole_row, 0, torch.remainder(centre - half_w, k_length))
            end = torch.where(whole_row, k_length, first + kernel_w)
            end_in_row = torch.minimum(end, k_length)
            wrapped = end > k_length
            changes.index_add_(0, (row_start + first).flatten(), torch.ones(first.numel(), dtype=torch.long))
            changes.index_add_(
                0, (row_start + end_in_row).flatten(), torch.full((first.numel(),), -1, dtype=torch.long)
            )
            wrap_start = row_start.expand_as(wrapped)[wrapped]
            changes.index_add_(0, wrap_start, torch.ones(wrap_start.numel(), dtype=torch.long))
            changes.index_add_(
                0, (row_start + end - k_length)[wrapped], torch.full((wrap_start.numel(),), -1, dtype=torch.long)
            )
        return changes.cumsum(0)[:-1]


class ReducedGridNeighbourhoodMask(ReducedGridCrossNeighbourhoodMask):
    """Which keys a query attends to on one :class:`ReducedGrid`: the cross-grid rule with each row centred on itself."""

    def __init__(self, grid: ReducedGrid, kernel_size: tuple[int, int]) -> None:
        super().__init__(grid, grid, kernel_size, row_map=torch.arange(grid.num_rows))
