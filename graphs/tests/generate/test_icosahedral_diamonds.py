# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import numpy as np
import pytest
import trimesh

from anemoi.graphs.generate.icosahedral_diamonds import IcosahedralDiamonds
from anemoi.graphs.generate.icosahedral_diamonds import diamond_ordering


@pytest.mark.parametrize("resolution", [0, 1, 3])
def test_layout_gives_the_triangles_of_the_icosphere(resolution: int) -> None:
    """The diamond lattice, put back in trimesh's order, has exactly the faces of trimesh's icosphere."""
    base = trimesh.creation.icosahedron()
    sphere = trimesh.creation.icosphere(subdivisions=resolution)
    order = diamond_ordering(sphere.vertices, base.vertices, base.faces)
    layout = IcosahedralDiamonds(resolution)

    assert sorted(order.tolist()) == list(range(layout.num_points))
    in_diamond_order = np.empty_like(order)
    in_diamond_order[order] = np.arange(len(order))
    faces = {tuple(sorted(face)) for face in in_diamond_order[sphere.faces].tolist()}
    assert faces == {tuple(sorted(face)) for face in layout.triangles().tolist()}


def test_triangles_face_outwards() -> None:
    base = trimesh.creation.icosahedron()
    sphere = trimesh.creation.icosphere(subdivisions=2)
    vertices = sphere.vertices[diamond_ordering(sphere.vertices, base.vertices, base.faces)]
    triangles = IcosahedralDiamonds(2).triangles()
    a, b, c = (vertices[triangles[:, k]] for k in range(3))
    assert (np.einsum("ij,ij->i", np.cross(b - a, c - a), a + b + c) > 0).all()


def test_every_point_but_the_poles_is_owned_by_one_diamond() -> None:
    layout = IcosahedralDiamonds(2)
    owned = layout.closed_grids[:, 1:, :-1].ravel()
    assert sorted(owned.tolist()) == list(range(1, layout.num_points - 1))
    assert set(layout.closed_grids.ravel().tolist()) == set(range(layout.num_points))


@pytest.mark.parametrize("num_points", [12, 42, 162, 10242])
def test_from_num_points(num_points: int) -> None:
    assert IcosahedralDiamonds.from_num_points(num_points).num_points == num_points


def test_from_num_points_rejects_other_sizes() -> None:
    with pytest.raises(ValueError, match="refined icosahedron"):
        IcosahedralDiamonds.from_num_points(100)
