# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""The refined icosahedron laid out as ten diamonds of a regular triangular lattice.

The twenty faces of the icosahedron pair up into ten diamonds. Pick one vertex as the north
vertex ``N``; its five neighbours, going anticlockwise seen from outside, are the upper ring
``U_0 .. U_4``; the vertex shared by ``U_c`` and ``U_{c+1}`` (other than ``N``) is ``L_c``; the
vertex opposite ``N`` is the south vertex ``S``. Counting ``c`` round modulo 5:

* northern diamond ``c`` (diamond ``c``) has the corners ``N``, ``U_c``, ``L_c``, ``U_{c+1}``;
* southern diamond ``c`` (diamond ``5 + c``) has the corners ``U_{c+1}``, ``L_c``, ``S``, ``L_{c+1}``.

Each diamond is refined into an ``n`` by ``n`` lattice, ``n = 2**resolution``. In diamond
coordinates ``(i, j)``, ``0 <= i, j <= n``, the corners are ``t = (0, 0)``, ``l = (n, 0)``,
``r = (0, n)`` and ``b = (n, n)``, in the order listed above (``t, l, b, r``); the upper triangle
is ``t, l, r`` and the lower one ``l, b, r``. Every point has the six lattice neighbours
``(i +- 1, j)``, ``(i, j +- 1)``, ``(i + 1, j - 1)`` and ``(i - 1, j + 1)``, and the lattice
distance between two points ``(di, dj)`` apart is ``max(|di|, |dj|, |di + dj|)``.

A diamond owns its points with ``1 <= i <= n`` and ``0 <= j < n``; the points of its first row
``i = 0`` and last column ``j = n`` belong to its neighbours. Every vertex of the mesh belongs to
exactly one diamond except ``N`` and ``S``. Points are numbered ``N`` first, then the ten
diamonds one after the other, each row by row, then ``S``:

    index(d, i, j) = 1 + d * n**2 + (i - 1) * n + j

This is the order in which :class:`anemoi.graphs.nodes.TriNodes` stores its nodes with
``ordering: diamonds``. The vertices themselves are those of ``trimesh.creation.icosphere``:
each refinement puts a vertex at the middle of every edge and moves it out onto the sphere.
"""

from __future__ import annotations

from functools import cached_property

import numpy as np

NUM_DIAMONDS = 10


class IcosahedralDiamonds:
    """Point numbering and lattice of the refined icosahedron at one resolution (see module docstring)."""

    def __init__(self, resolution: int) -> None:
        if resolution < 0:
            raise ValueError(f"The resolution must not be negative, got {resolution}.")
        self.resolution = resolution
        self.n = 2**resolution

    @classmethod
    def from_num_points(cls, num_points: int) -> IcosahedralDiamonds:
        """The layout with ``num_points = 10 * 4**resolution + 2`` points."""
        resolution = 0
        while NUM_DIAMONDS * 4**resolution + 2 < num_points:
            resolution += 1
        if NUM_DIAMONDS * 4**resolution + 2 != num_points:
            raise ValueError(f"{num_points} points do not form a refined icosahedron (10 * 4**r + 2 points).")
        return cls(resolution)

    @property
    def num_points(self) -> int:
        return NUM_DIAMONDS * self.n**2 + 2

    @property
    def north(self) -> int:
        return 0

    @property
    def south(self) -> int:
        return self.num_points - 1

    def index(self, d, i, j):
        """Index of the point owned by diamond ``d`` at ``(i, j)``, ``1 <= i <= n``, ``0 <= j < n``."""
        return 1 + d * self.n**2 + (i - 1) * self.n + j

    @cached_property
    def closed_grids(self) -> np.ndarray:
        """Index of every point ``(i, j)``, ``0 <= i, j <= n``, of every diamond, shape ``(10, n + 1, n + 1)``.

        The first row and last column of a diamond hold points of its neighbours, found along the
        edges the diamonds share:

        * northern ``c``: row 0 is the edge ``N - U_{c+1}``, column 0 of northern ``c + 1``; column
          ``n`` is the edge ``U_{c+1} - L_c``, column 0 of southern ``c``;
        * southern ``c``: row 0 is the edge ``U_{c+1} - L_{c+1}``, row ``n`` of northern ``c + 1``;
          column ``n`` is the edge ``L_{c+1} - S``, row ``n`` of southern ``c + 1``.
        """
        n = self.n
        grids = np.full((NUM_DIAMONDS, n + 1, n + 1), -1, dtype=np.int64)
        i, j = np.meshgrid(np.arange(1, n + 1), np.arange(n), indexing="ij")
        for d in range(NUM_DIAMONDS):
            grids[d, 1:, :n] = self.index(d, i, j)
        steps = np.arange(1, n + 1)
        for c in range(5):
            north, south = c, 5 + c
            north_next, south_next = (c + 1) % 5, 5 + (c + 1) % 5
            grids[north, 0, 0] = self.north
            grids[north, 0, 1:] = grids[north_next, steps, 0]
            grids[north, 1:, n] = grids[south, steps, 0]
            grids[south, 0, :n] = grids[north_next, n, :n]
            # Corner r of southern c is L_{c+1}, corner l of southern c + 1.
            grids[south, 0, n] = grids[south_next, n, 0]
            grids[south, 1:n, n] = grids[south_next, n, 1:n]
            grids[south, n, n] = self.south
        return grids

    def triangles(self) -> np.ndarray:
        """The ``20 * n**2`` triangles of the mesh as point indices, shape ``(20 * n**2, 3)``, anticlockwise from outside."""
        g = self.closed_grids
        upper = np.stack([g[:, :-1, :-1], g[:, 1:, :-1], g[:, :-1, 1:]], axis=-1)
        lower = np.stack([g[:, 1:, :-1], g[:, 1:, 1:], g[:, :-1, 1:]], axis=-1)
        return np.concatenate([upper.reshape(-1, 3), lower.reshape(-1, 3)])

    def positions(self, corners: np.ndarray) -> np.ndarray:
        """Unit vectors of all points, refining from the twelve corner vectors ``N, U_0..U_4, L_0..L_4, S``.

        Each refinement puts a point at the middle of every lattice edge and moves it out onto the
        unit sphere, as ``trimesh.creation.icosphere`` does.
        """
        north, upper, lower, south = corners[0], corners[1:6], corners[6:11], corners[11]
        diamonds = []
        for c in range(5):
            diamonds.append((north, upper[c], lower[c], upper[(c + 1) % 5]))
        for c in range(5):
            diamonds.append((upper[(c + 1) % 5], lower[c], south, lower[(c + 1) % 5]))

        out = np.empty((self.num_points, 3), dtype=np.float64)
        grids = self.closed_grids
        for d, (top, left, bottom, right) in enumerate(diamonds):
            grid = np.array([[top, right], [left, bottom]], dtype=np.float64)
            for _ in range(self.resolution):
                grid = _refine(grid)
            out[grids[d].ravel()] = grid.reshape(-1, 3)
        return out


def _unit(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def _refine(grid: np.ndarray) -> np.ndarray:
    """One refinement of a closed diamond lattice of unit vectors, shape ``(m + 1, m + 1, 3)`` to ``(2m + 1, 2m + 1, 3)``."""
    m = grid.shape[0] - 1
    fine = np.empty((2 * m + 1, 2 * m + 1, 3), dtype=grid.dtype)
    fine[::2, ::2] = grid
    fine[1::2, ::2] = _unit((grid[:-1] + grid[1:]) / 2)
    fine[::2, 1::2] = _unit((grid[:, :-1] + grid[:, 1:]) / 2)
    fine[1::2, 1::2] = _unit((grid[:-1, 1:] + grid[1:, :-1]) / 2)
    return fine


def label_icosahedron(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """The vertices of an icosahedron in the order ``N, U_0..U_4, L_0..L_4, S`` (see module docstring).

    ``N`` is the vertex furthest along z (the first one on a tie), and ``U_0`` its neighbour of
    lowest index.
    """
    vertices = _unit(np.asarray(vertices, dtype=np.float64))
    neighbours = [set() for _ in range(len(vertices))]
    for a, b, c in faces:
        neighbours[a] |= {b, c}
        neighbours[b] |= {a, c}
        neighbours[c] |= {a, b}

    north = int(np.argmax(vertices[:, 2]))
    ring = sorted(neighbours[north])
    axis = vertices[north]
    e1 = _unit(vertices[ring[0]] - np.dot(vertices[ring[0]], axis) * axis)
    e2 = np.cross(axis, e1)
    angles = [np.arctan2(np.dot(vertices[v], e2), np.dot(vertices[v], e1)) % (2 * np.pi) for v in ring]
    upper = [ring[k] for k in np.argsort(angles)]
    lower = []
    for c in range(5):
        (shared,) = (neighbours[upper[c]] & neighbours[upper[(c + 1) % 5]]) - {north}
        lower.append(shared)
    (south,) = set(range(len(vertices))) - {north, *upper, *lower}
    return np.array([north, *upper, *lower, south])


def diamond_ordering(vertices: np.ndarray, base_vertices: np.ndarray, base_faces: np.ndarray) -> np.ndarray:
    """The order that puts the vertices of a refined icosahedron into diamond order.

    Parameters
    ----------
    vertices : np.ndarray
        Unit vectors of the refined mesh, shape ``(10 * 4**r + 2, 3)``, in any order.
    base_vertices, base_faces : np.ndarray
        The icosahedron it was refined from.

    Returns
    -------
    np.ndarray
        ``order`` such that ``vertices[order]`` is in diamond order.
    """
    from scipy.spatial import cKDTree

    layout = IcosahedralDiamonds.from_num_points(len(vertices))
    corners = _unit(np.asarray(base_vertices, dtype=np.float64))[label_icosahedron(base_vertices, base_faces)]
    expected = layout.positions(corners)
    distances, order = cKDTree(_unit(vertices)).query(expected)
    spacing = np.pi / (5 * layout.n)
    if distances.max() > 1e-3 * spacing or len(np.unique(order)) != len(order):
        raise ValueError("The vertices are not those of the refined icosahedron.")
    return order
