# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import numpy as np

from anemoi.graphs.generate.tri_icosahedron import get_latlon_coords_icosphere
from anemoi.graphs.generate.utils import get_coordinates_ordering


def test_coordinates_ordering_breaks_latitude_ties_by_longitude() -> None:
    """Nodes sharing a latitude are ordered by descending longitude, not by sort implementation."""
    coords = get_latlon_coords_icosphere(4)
    assert len(np.unique(coords[:, 0])) < len(coords), "Test needs latitude ties."

    node_ordering = get_coordinates_ordering(coords)

    expected = np.lexsort((coords[:, 1], coords[:, 0]))[::-1]
    np.testing.assert_array_equal(node_ordering, expected)


def test_ordered_coordinates_do_not_depend_on_input_order() -> None:
    coords = get_latlon_coords_icosphere(4)
    perm = np.random.default_rng(0).permutation(len(coords))

    ordered = coords[get_coordinates_ordering(coords)]
    ordered_from_permuted = coords[perm][get_coordinates_ordering(coords[perm])]

    np.testing.assert_array_equal(ordered, ordered_from_permuted)
