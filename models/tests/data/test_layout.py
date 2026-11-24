# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import pytest
import torch

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


# --------------------------------------------------------------------------------------
# conform_to: optional axes and axis order between a caller's layout and the model's
# --------------------------------------------------------------------------------------

THREE_D = TensorLayout.from_tuple("time", "grid", "variables")


def test_conform_to_inserts_a_missing_axis_with_size_one():
    values = torch.arange(24.0).reshape(2, 4, 3)

    conformed = THREE_D.conform_to(values, EXPLICIT)

    assert conformed.shape == (2, 1, 4, 3)
    assert torch.equal(conformed[:, 0], values)


def test_conform_to_drops_a_size_one_axis():
    values = torch.arange(24.0).reshape(2, 1, 4, 3)

    conformed = EXPLICIT.conform_to(values, THREE_D)

    assert conformed.shape == (2, 4, 3)
    assert torch.equal(conformed, values[:, 0])


def test_conform_to_refuses_to_drop_several_members():
    with pytest.raises(ValueError, match="Cannot drop the 'ensemble' axis of size 3"):
        EXPLICIT.conform_to(torch.zeros(2, 3, 4, 5), THREE_D)


def test_conform_to_inserts_and_reorders_together():
    """(grid, variables, time) in, (time, ensemble, grid, variables) out, and back again."""
    source = TensorLayout.from_tuple("grid", "variables", "time")
    values = torch.arange(24.0).reshape(4, 3, 2)

    conformed = source.conform_to(values, EXPLICIT)

    assert conformed.shape == (2, 1, 4, 3)
    assert torch.equal(conformed[:, 0], values.permute(2, 0, 1))
    assert torch.equal(EXPLICIT.conform_to(conformed, source), values)


def test_conform_to_returns_the_tensor_itself_when_nothing_changes():
    values = torch.zeros(2, 1, 4, 3)
    assert EXPLICIT.conform_to(values, EXPLICIT) is values


def test_conform_to_needs_the_layout_rank():
    with pytest.raises(ValueError, match="does not match the layout"):
        EXPLICIT.conform_to(torch.zeros(2, 4, 3), THREE_D)


def test_conform_to_drops_the_batch_axis_of_a_single_sample():
    values = torch.arange(24.0).reshape(1, 2, 1, 4, 3)

    assert torch.equal(EXPLICIT.with_batch_dim().conform_to(values, THREE_D), values[0, :, 0])
    with pytest.raises(ValueError, match="Cannot drop the 'batch' axis of size 2"):
        EXPLICIT.with_batch_dim().conform_to(torch.zeros(2, 2, 1, 4, 3), EXPLICIT)


@pytest.mark.parametrize(
    ("names", "match"),
    [
        (("time", "member", "grid", "variables"), "Unknown axis names \\['member'\\]"),
        (("time", "grid", "grid", "variables"), "names the axes \\['grid'\\] more than once"),
    ],
)
def test_from_tuple_rejects_unknown_and_repeated_names(names, match):
    with pytest.raises(ValueError, match=match):
        TensorLayout.from_tuple(*names)
