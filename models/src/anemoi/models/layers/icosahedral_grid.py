# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Neighbourhood attention on the refined icosahedron: every point attends to the points within ``radius`` hops.

The points are stored in diamond order (see :mod:`anemoi.graphs.generate.icosahedral_diamonds`):
ten diamonds, each an ``n`` by ``n`` lattice in which every point has six neighbours. Within a
diamond the points within ``radius`` hops of a point ``(i, j)`` are the lattice points
``(i + di, j + dj)`` with ``max(|di|, |dj|, |di + dj|) <= radius``, a hexagon of
``3 * radius * (radius + 1) + 1`` points.

Near the edge of a diamond that hexagon reaches into the neighbouring diamonds. Each diamond
therefore gets a lookup table, ``radius`` points wider on every side, that names the point at
every lattice position around it: the neighbouring diamonds are unfolded flat around it, the way
a paper net of the icosahedron is laid out. The attention kernels read keys only through these
tables and never need a list of edges; a table holds one number per lattice position.

At the twelve vertices of the original icosahedron five triangles meet instead of six, so the
unfolded neighbours do not fill the plane: around each corner of a diamond a wedge stays empty,
or points meet twice along a seam. For the few points near those vertices (and for the two
vertices that belong to no diamond) the hexagon read from the table differs from the true set of
points within ``radius`` hops. These points are the exceptions; their neighbourhoods are kept
as explicit lists and handled outside the kernels, so every point attends to exactly the points
within ``radius`` hops.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from functools import cached_property

import numpy as np
import scipy.ndimage
import scipy.sparse
import scipy.sparse.csgraph
import torch
from torch import Tensor

from anemoi.graphs.generate.icosahedral_diamonds import NUM_DIAMONDS
from anemoi.graphs.generate.icosahedral_diamonds import IcosahedralDiamonds


def hexagon_offsets(radius: int) -> np.ndarray:
    """The lattice offsets ``(di, dj)`` within ``radius`` hops, shape ``(3 * radius * (radius + 1) + 1, 2)``."""
    d = np.arange(-radius, radius + 1)
    di, dj = np.meshgrid(d, d, indexing="ij")
    keep = np.abs(di + dj) <= radius
    return np.stack([di[keep], dj[keep]], axis=1)


class IcosahedralNeighbourhood:
    """The lookup tables and exceptions of neighbourhood attention on one refined icosahedron.

    Lattice positions are counted in the padded frame of each diamond: the point ``(i, j)`` of
    diamond ``d`` (``1 <= i <= n``, ``0 <= j < n`` for its own points) sits at row
    ``i - 1 + radius`` and column ``j + radius`` of the diamond's table, which has
    ``n + 2 * radius`` rows and columns.

    Parameters
    ----------
    resolution : int
        Refinement level of the icosahedron; ``n = 2**resolution`` points along a diamond edge.
    radius : int
        Each point attends to the points within this many hops.
    """

    def __init__(self, resolution: int, radius: int) -> None:
        if radius < 1:
            raise ValueError(f"The radius must be at least 1, got {radius}.")
        self.layout = IcosahedralDiamonds(resolution)
        self.resolution = resolution
        self.radius = radius
        self.n = self.layout.n
        self.size = self.n + 2 * radius

    @property
    def num_points(self) -> int:
        return self.layout.num_points

    @property
    def window_size(self) -> int:
        """Number of points within ``radius`` hops of a point away from the twelve icosahedron vertices."""
        return 3 * self.radius * (self.radius + 1) + 1

    def cells(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Diamond, table row and table column of points owned by a diamond (not the first or last point)."""
        n, rest = self.n, np.asarray(points) - 1
        d, rest = rest // n**2, rest % n**2
        return d, rest // n + self.radius, rest % n + self.radius

    @cached_property
    def adjacency(self) -> scipy.sparse.csr_matrix:
        """Which points are joined by an edge of the mesh."""
        triangles = self.layout.triangles()
        a = np.concatenate([triangles[:, 0], triangles[:, 1], triangles[:, 2]])
        b = np.concatenate([triangles[:, 1], triangles[:, 2], triangles[:, 0]])
        ones = np.ones(2 * a.size, dtype=bool)
        n = self.num_points
        return scipy.sparse.csr_matrix((ones, (np.concatenate([a, b]), np.concatenate([b, a]))), shape=(n, n))

    @cached_property
    def table(self) -> np.ndarray:
        """The point at every lattice position around every diamond, or -1, shape ``(10, size, size)``."""
        tables = np.full((NUM_DIAMONDS, self.size, self.size), -1, dtype=np.int64)
        across = _triangles_across_edges(self.layout.triangles(), self.num_points)
        for d in range(NUM_DIAMONDS):
            self._unfold_around(d, tables[d], across)
        return tables

    def _unfold_around(self, d: int, table: np.ndarray, across: tuple[np.ndarray, np.ndarray]) -> None:
        """Fill the table of diamond ``d`` by laying the triangles around it flat, nearest first.

        Positions are kept in the diamond's own coordinates ``(i, j)``. A triangle with corners at
        ``A``, ``B`` and ``C`` has its neighbour across the edge ``A - B`` with the third corner at
        ``A + B - C``. Each triangle of the mesh is laid down once; a position reached by two
        different points keeps the first.
        """
        n, radius = self.n, self.radius
        triangles = self.layout.triangles()
        neighbour, third = across
        grid = self.layout.closed_grids[d]
        table[radius - 1 : radius + n, radius : radius + n + 1] = grid

        # Unfold a little past the table so that positions only reached round its corner are found.
        lo_i, hi_i, lo_j, hi_j = -radius - 1, n + radius + 1, -radius - 1, n + radius

        placed = np.zeros(len(triangles), dtype=bool)
        queue = deque()
        cell = np.arange(n)
        ii, jj = np.meshgrid(cell, cell, indexing="ij")
        for half, corners in ((0, ((0, 0), (1, 0), (0, 1))), (1, ((1, 0), (1, 1), (0, 1)))):
            ids = half * NUM_DIAMONDS * n**2 + d * n**2 + ii * n + jj
            placed[ids.ravel()] = True
            border = (ii == 0) | (jj == 0) | (ii == n - 1) | (jj == n - 1)
            for t, i, j in zip(ids[border], ii[border], jj[border]):
                queue.append((int(t), tuple((int(i + ci), int(j + cj)) for ci, cj in corners)))

        while queue:
            t, at = queue.popleft()
            for e in range(3):
                other = neighbour[t, e]
                if placed[other]:
                    continue
                a, b, c = at[e], at[(e + 1) % 3], at[(e + 2) % 3]
                x = (a[0] + b[0] - c[0], a[1] + b[1] - c[1])
                if not (lo_i <= x[0] <= hi_i and lo_j <= x[1] <= hi_j):
                    continue
                placed[other] = True
                row, col = x[0] - 1 + radius, x[1] + radius
                if 0 <= row < self.size and 0 <= col < self.size and table[row, col] < 0:
                    table[row, col] = third[t, e]
                # Corners of the new triangle in its own order, so its edges are found the same way.
                where = {int(triangles[t, e]): a, int(triangles[t, (e + 1) % 3]): b, int(third[t, e]): x}
                queue.append((int(other), tuple(where[int(p)] for p in triangles[other])))

    def windows(self, points: np.ndarray) -> np.ndarray:
        """The hexagon read from the tables for each of ``points`` (owned by a diamond), -1 where empty."""
        d, row, col = self.cells(points)
        offsets = hexagon_offsets(self.radius)
        rows = np.clip(row[:, None] + offsets[None, :, 0], 0, self.size - 1)
        cols = np.clip(col[:, None] + offsets[None, :, 1], 0, self.size - 1)
        return self.table[d[:, None], rows, cols]

    def within_radius(self, points: np.ndarray) -> scipy.sparse.csr_matrix:
        """Rows of the points within ``radius`` hops of each of ``points``."""
        step = self.adjacency + scipy.sparse.identity(self.num_points, dtype=bool, format="csr")
        reach = scipy.sparse.csr_matrix(
            (np.ones(len(points), dtype=bool), (np.arange(len(points)), points)),
            shape=(len(points), self.num_points),
        )
        for _ in range(self.radius):
            reach = (reach @ step).astype(bool)
        reach.sort_indices()
        return reach

    @cached_property
    def nearest_vertex(self) -> np.ndarray:
        """For every point within ``3 * radius`` hops of one of the twelve icosahedron vertices, which of them is nearest; -1 elsewhere."""
        degree = np.diff(self.adjacency.indptr)
        vertices = np.flatnonzero(degree == 5)
        hops = scipy.sparse.csgraph.dijkstra(
            self.adjacency, unweighted=True, indices=vertices, limit=3 * self.radius + 1
        )
        return np.where(np.isfinite(hops).any(axis=0), np.argmin(hops, axis=0), -1)

    @cached_property
    def exceptions(self) -> np.ndarray:
        """The points whose hexagon read from the tables differs from the points within ``radius`` hops.

        Away from the twelve icosahedron vertices the unfolding is exact, so only points close to
        them are checked: a seam starts at a vertex and stays within the ``radius`` wide border of
        the table, so a hexagon can only touch it when its centre is within ``3 * radius`` hops of
        the vertex.
        """
        near = np.flatnonzero(self.nearest_vertex >= 0)
        candidates = near[(near != self.layout.north) & (near != self.layout.south)]

        windows = self.windows(candidates)
        true = self.within_radius(candidates)
        wrong = [self.layout.north, self.layout.south]
        for k, point in enumerate(candidates):
            window = windows[k][windows[k] >= 0]
            expected = true.indices[true.indptr[k] : true.indptr[k + 1]]
            if len(window) != len(expected) or not np.array_equal(np.sort(window), expected):
                wrong.append(int(point))
        return np.sort(np.array(wrong, dtype=np.int64))

    def mask(self, device: torch.device | None = None) -> Tensor:
        """The full ``points x points`` attention mask, True where a query attends to a key. For small grids and tests."""
        reach = self.within_radius(np.arange(self.num_points))
        return torch.as_tensor(reach.toarray(), device=device)

    def shard(self, own: slice | None = None) -> IcosahedralShard:
        """What the GPU that owns the query points ``own`` needs (see :class:`IcosahedralShard`); all points by default."""
        own = slice(0, self.num_points) if own is None else own
        if not 0 <= own.start < own.stop <= self.num_points:
            raise ValueError(f"The own points {own} are not a non-empty run of the {self.num_points} points.")
        points = np.arange(own.start, own.stop)
        is_exception = np.zeros(self.num_points, dtype=bool)
        is_exception[self.exceptions] = True
        own_exceptions = points[is_exception[points]]

        # Keys: every point a hexagon of an own point reaches in the tables, and the neighbours of the
        # own exceptions. A point's own diamond holds it at a single position, so marking those
        # positions and growing them by the hexagon finds the reach of all own points at once.
        in_diamond = points[(points > self.layout.north) & (points < self.layout.south)]
        d, row, col = self.cells(in_diamond)
        marked = np.zeros(self.table.shape, dtype=bool)
        marked[d, row, col] = True
        hexagon = np.zeros((2 * self.radius + 1,) * 2, dtype=bool)
        offsets = hexagon_offsets(self.radius)
        hexagon[offsets[:, 0] + self.radius, offsets[:, 1] + self.radius] = True
        reached = np.stack([scipy.ndimage.binary_dilation(m, structure=hexagon) for m in marked])
        exception_lists = self.within_radius(own_exceptions)
        wanted = np.union1d(self.table[reached & (self.table >= 0)], exception_lists.indices)
        halo = wanted[(wanted < own.start) | (wanted >= own.stop)]

        # Position of every point in this GPU's keys: its own points first, then the halo.
        key_position = np.full(self.num_points + 1, -1, dtype=np.int64)
        key_position[points] = np.arange(len(points))
        key_position[halo] = len(points) + np.arange(len(halo))
        query_position = np.full(self.num_points + 1, -1, dtype=np.int64)
        query_position[points] = np.arange(len(points))
        query_position[own_exceptions] = -1
        # Index -1 (an empty table position) stays -1.
        keys = key_position[self.table].astype(np.int32)
        queries = query_position[self.table].astype(np.int32)

        # Keys whose gradient is put together outside the tiled kernels: the exceptions among the
        # keys take it from all their own queries, keys next to an own exception query take that
        # query's part.
        key_points = np.concatenate([points, halo])
        key_exceptions = key_points[is_exception[key_points]]
        corrected = np.union1d(key_exceptions, exception_lists.indices)
        replaced = is_exception[corrected]
        lists = self.within_radius(corrected).tocoo()
        keep = (lists.col >= own.start) & (lists.col < own.stop)
        keep &= replaced[lists.row] | is_exception[lists.col]
        query_lists = scipy.sparse.csr_matrix(
            (np.ones(int(keep.sum()), dtype=bool), (lists.row[keep], lists.col[keep] - own.start)),
            shape=(len(corrected), max(len(points), 1)),
        )
        query_lists.sort_indices()

        return IcosahedralShard(
            radius=self.radius,
            n=self.n,
            own=own,
            halo=halo,
            keys=keys,
            queries=queries,
            exception_queries=own_exceptions - own.start,
            exception_clusters=self.nearest_vertex[own_exceptions],
            exception_key_starts=exception_lists.indptr.astype(np.int64),
            exception_key_lists=key_position[exception_lists.indices],
            corrected_keys=key_position[corrected],
            corrected_clusters=self.nearest_vertex[corrected],
            replaced=replaced,
            corrected_query_starts=query_lists.indptr.astype(np.int64),
            corrected_query_lists=query_lists.indices.astype(np.int64),
        )


@dataclass(frozen=True, eq=False)
class IcosahedralShard:
    """The tables and lists one GPU needs for icosahedral neighbourhood attention.

    The GPU owns the query points ``own``, a run of consecutive points, and holds as keys its own
    points followed by the ``halo``, the points of other GPUs that its queries reach, in
    increasing order. Query positions count from ``own.start``, key positions from the first of
    these keys. All arrays are on the host; -1 marks an empty entry.
    """

    radius: int
    n: int
    own: slice
    halo: np.ndarray
    keys: np.ndarray
    "Key position of the point at every lattice position around every diamond, shape ``(10, n + 2 * radius, n + 2 * radius)``."
    queries: np.ndarray
    "Query position of the point at every lattice position, if it is an own query and not an exception."
    exception_queries: np.ndarray
    "Query positions of the own exceptions, in increasing order."
    exception_clusters: np.ndarray
    "The icosahedron vertex nearest to each own exception."
    exception_key_starts: np.ndarray
    "Where the key list of each own exception starts in ``exception_key_lists``, and where the last ends."
    exception_key_lists: np.ndarray
    "Key positions of the points within ``radius`` hops of each own exception, one list after the other."
    corrected_keys: np.ndarray
    "Key positions of the keys that get gradient from queries outside the tiled kernels."
    corrected_clusters: np.ndarray
    "The icosahedron vertex nearest to each corrected key."
    replaced: np.ndarray
    "Which corrected keys are exceptions, whose gradient from the tiled kernels is replaced; the others' is added to."
    corrected_query_starts: np.ndarray
    "Where the query list of each corrected key starts in ``corrected_query_lists``, and where the last ends."
    corrected_query_lists: np.ndarray
    "For each corrected key, the positions of the own queries whose part is put together outside the tiled kernels."

    @property
    def num_queries(self) -> int:
        return self.own.stop - self.own.start

    @property
    def num_keys(self) -> int:
        return self.num_queries + len(self.halo)


def _triangles_across_edges(triangles: np.ndarray, num_points: int) -> tuple[np.ndarray, np.ndarray]:
    """For each triangle and each edge ``e`` (corners ``e`` and ``e + 1``): the triangle across it and that triangle's third corner."""
    num = len(triangles)
    a = triangles[:, [0, 1, 2]].ravel()
    b = triangles[:, [1, 2, 0]].ravel()
    keys = np.minimum(a, b) * num_points + np.maximum(a, b)
    order = np.argsort(keys, kind="stable")
    # Every edge is shared by exactly two triangles, so the sorted keys come in pairs.
    first, second = order[0::2], order[1::2]
    partner = np.empty(3 * num, dtype=np.int64)
    partner[first], partner[second] = second, first
    neighbour = (partner // 3).reshape(num, 3)
    # The corner of the triangle across that is on neither end of the shared edge.
    other = triangles[neighbour]
    edge_ends = np.stack([triangles, np.roll(triangles, -1, axis=1)], axis=-1)
    on_edge = (other[:, :, :, None] == edge_ends[:, :, None, :]).any(axis=-1)
    third = other[~on_edge].reshape(num, 3)
    return neighbour, third
