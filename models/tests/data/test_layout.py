# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import pytest

from anemoi.models.data import TensorLayout

EXPLICIT = TensorLayout(time=0, ensemble=1, grid=2, variables=3)


@pytest.mark.parametrize(
    "layout",
    [
        TensorLayout(time=0, ensemble=1),  # grid and variables default to the last two axes
        TensorLayout(time=-4, ensemble=-3, grid=-2, variables=-1),
        TensorLayout(time=0, ensemble=1, grid=-2, variables=3),
        TensorLayout.from_tuple("time", "ensemble", "grid", "variables"),
    ],
)
def test_negative_positions_are_resolved_on_construction(layout):
    assert layout == EXPLICIT
    assert hash(layout) == hash(EXPLICIT)
    assert (layout.time, layout.ensemble, layout.grid, layout.variables) == (0, 1, 2, 3)


def test_pattern_follows_physical_order_with_default_positions():
    # Sorting unresolved positions put the negative defaults first: "grid variables time ensemble".
    layout = TensorLayout(time=0, ensemble=1)
    assert layout.pattern == "time ensemble grid variables"
    assert layout.axis_names == ("time", "ensemble", "grid", "variables")
    assert layout.with_batch_dim().pattern == "batch time ensemble grid variables"


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"time": 0, "grid": 1, "variables": 5}, "variables=5 is out of range for a layout of 3 axes"),
        ({"time": -4, "grid": 1, "variables": 2}, "time=-4 is out of range for a layout of 3 axes"),
        ({"time": 0, "ensemble": 1, "grid": 2, "variables": -2}, "several axes at the same position"),
    ],
)
def test_invalid_positions_are_rejected(kwargs, match):
    with pytest.raises(ValueError, match=match):
        TensorLayout(**kwargs)


def test_batch_dim_round_trip():
    batched = EXPLICIT.with_batch_dim()
    assert batched == TensorLayout(batch=0, time=1, ensemble=2, grid=3, variables=4)
    assert batched.with_batch_dim() is batched
    assert batched.without_batch_dim() == EXPLICIT
    assert EXPLICIT.without_batch_dim() is EXPLICIT


def test_without_batch_dim_shifts_only_the_axes_after_the_batch():
    layout = TensorLayout(time=0, ensemble=1, batch=2, grid=3, variables=4)
    assert layout.without_batch_dim() == EXPLICIT


def test_tabular_layout_with_batch_dim():
    assert TensorLayout(ensemble=0).with_batch_dim() == TensorLayout(batch=0, ensemble=1, grid=2, variables=3)


@pytest.mark.parametrize(
    ("names", "permutation"),
    [
        (("time", "ensemble", "grid", "variables"), (0, 1, 2, 3)),
        (("grid", "time", "ensemble", "variables"), (1, 2, 0, 3)),
        (("variables", "grid", "ensemble", "time"), (3, 2, 1, 0)),
        (("grid", "ensemble", "variables"), (1, 0, 2)),
        (("variables", "grid"), (1, 0)),
    ],
)
def test_canonical_reorders_the_axes(names, permutation):
    layout = TensorLayout.from_tuple(*names)
    assert layout.canonical_permutation == permutation
    assert layout.is_canonical == (permutation == tuple(range(len(names))))
    canonical = layout.canonical()
    assert canonical.is_canonical
    assert canonical.axis_names == tuple(name for name in TensorLayout.AXES if name in names)
    # the permutation takes a tensor in this layout to the canonical one
    assert tuple(names[i] for i in permutation) == canonical.axis_names


def test_canonical_layout_is_returned_unchanged():
    assert EXPLICIT.canonical() is EXPLICIT
