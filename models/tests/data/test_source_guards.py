# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Construction guards, flattening and allgather edge cases of the concrete sources.

Complements ``test_source.py``, whose ``TestAllgather`` covers the single-rank (``group=None``) paths.
"""

import numpy as np
import pytest
import torch

from anemoi.models.data import FlatSource
from anemoi.models.data import TensorLayout
from anemoi.models.data.sources import GriddedSource
from anemoi.models.data.sources import TabularSource

GRID = 4
GRIDDED_LAYOUT = TensorLayout(batch=0, time=1, ensemble=2, grid=3, variables=4)
TABULAR_LAYOUT = TensorLayout(grid=0, variables=1)
_GUARD_COMMON = {"name": "d", "variables": ["a", "b", "c"], "statistics": {"mean": np.zeros(3)}}


class _FakeGroup:
    """Minimal ProcessGroup stand-in; the sharding guards only call ``size()``."""

    def __init__(self, size: int) -> None:
        self._size = size

    def size(self) -> int:
        return self._size


def _tabular_source(timedeltas: list[torch.Tensor], shard_sizes: list | None = None) -> TabularSource:
    return TabularSource(
        name="obs",
        data=[torch.zeros(2, 1), torch.zeros(3, 1)],
        variables=["value"],
        statistics={},
        coordinates=[torch.zeros(2, 2), torch.ones(3, 2)],
        timedeltas=timedeltas,
        layout=TABULAR_LAYOUT,
        boundaries=[(slice(0, 2),), (slice(0, 3),)],
        shard_sizes=shard_sizes,
    )


def _gridded_source(shard_sizes: list[int] | None = None, grid: int = GRID) -> GriddedSource:
    return GriddedSource(
        name="era5",
        data=torch.arange(float(grid * 2)).reshape(1, 1, 1, grid, 2),
        variables=["a", "b"],
        statistics={},
        coordinates=torch.zeros(grid, 2),
        layout=GRIDDED_LAYOUT,
        shard_sizes=shard_sizes,
    )


def test_tabular_flatten_preserves_timedelta_order() -> None:
    source = _tabular_source(
        [
            torch.tensor([-3600.0, 0.0]),
            torch.tensor([1800.0, 3600.0, 7200.0]),
        ],
    )

    flattened = source.flatten()

    assert flattened.data.shape == (5, 1)
    assert flattened.coordinates.shape == (5, 2)
    assert torch.equal(flattened.timedeltas, torch.tensor([-3600.0, 0.0, 1800.0, 3600.0, 7200.0]))
    assert flattened.batch_sizes == (2, 3)


def test_tabular_source_rejects_misaligned_timedeltas() -> None:
    with pytest.raises(ValueError, match="same number of nodes"):
        _tabular_source([torch.zeros(2), torch.zeros(2)])


def test_flat_source_preserves_legacy_constructor_order() -> None:
    flat = FlatSource(torch.zeros(1, 1), torch.zeros(1, 2), torch.device("cpu"), None, (1,))

    assert flat.batch_sizes == (1,)
    assert flat.timedeltas is None


@pytest.mark.parametrize("group_size", [1, 2])
def test_replicated_gridded_allgather_over_a_group_is_a_noop(group_size: int) -> None:
    """A replicated source must no-op rather than attempt a collective.

    Regression test for ``TypeError: 'NoneType' object is not iterable`` raised from
    ``primitives._gather``: the batch is all-gathered once in ``on_after_batch_transfer``
    (which clears ``shard_sizes``) and then again by the validation plot callbacks over the
    model communication group, so ``allgather`` has to be idempotent.
    """
    source = _gridded_source(shard_sizes=None)

    assert source.allgather(_FakeGroup(group_size)) is source


def test_gridded_allgather_with_a_single_rank_group_drops_shard_sizes() -> None:
    source = _gridded_source(shard_sizes=[GRID])

    gathered = source.allgather(_FakeGroup(1))

    assert gathered.shard_sizes is None
    assert gathered.data is source.data  # nothing to communicate, no copy


def test_gridded_allgather_rejects_coordinates_on_another_device() -> None:
    """Coordinates are gathered alongside the data, so they must already share its device.

    ``Batch.to`` is what puts them there; allgather does not transfer them behind the
    caller's back. The "meta" device stands in for a GPU here.
    """
    source = _gridded_source(shard_sizes=[2, 2])
    source = source.clone(coordinates=source.coordinates.to("meta"))

    with pytest.raises(ValueError, match="coordinates on meta"):
        source.allgather(_FakeGroup(2))


@pytest.mark.parametrize("group_size", [1, 2])
def test_replicated_tabular_allgather_over_a_group_is_a_noop(group_size: int) -> None:
    source = _tabular_source([torch.zeros(2), torch.zeros(3)], shard_sizes=None)

    assert source.allgather(_FakeGroup(group_size)) is source


class TestSourceGuards:
    """The ``__post_init__`` checks reject a payload built for the other kind of source.

    Without them a mis-built source fails later with an opaque error -- a gridded payload
    wrongly routed to a tabular source, for instance, surfaces as
    ``cat() received an invalid combination of arguments``.
    """

    def test_tabular_source_rejects_a_gridded_layout(self) -> None:
        with pytest.raises(ValueError, match="requires a layout without time and batch axes"):
            TabularSource(
                data=[torch.zeros(2, 2, 1, GRID, 3)],
                coordinates=[torch.zeros(GRID, 2)],
                timedeltas=[torch.zeros(GRID)],
                boundaries=[(slice(0, GRID),)],
                layout=GRIDDED_LAYOUT,
                **_GUARD_COMMON,
            )

    def test_tabular_source_rejects_a_single_tensor(self) -> None:
        """Tabular payloads are one tensor per sample, held in a list."""
        with pytest.raises(TypeError, match="must be a list of tensors"):
            TabularSource(
                data=torch.zeros(GRID, 3),
                coordinates=[torch.zeros(GRID, 2)],
                timedeltas=[torch.zeros(GRID)],
                boundaries=[(slice(0, GRID),)],
                layout=TABULAR_LAYOUT,
                **_GUARD_COMMON,
            )

    def test_gridded_source_rejects_a_layout_without_a_time_axis(self) -> None:
        """Gridded sources carry an explicit time axis."""
        with pytest.raises(ValueError, match="requires a layout with a time axis"):
            GriddedSource(
                data=torch.zeros(GRID, 3),
                coordinates=torch.zeros(GRID, 2),
                layout=TABULAR_LAYOUT,
                **_GUARD_COMMON,
            )

    def test_gridded_source_rejects_a_list_payload(self) -> None:
        """Gridded payloads are a single stacked tensor."""
        with pytest.raises(TypeError, match="must be a single tensor"):
            GriddedSource(
                data=[torch.zeros(2, 1, GRID, 3)],
                coordinates=torch.zeros(GRID, 2),
                layout=GRIDDED_LAYOUT,
                **_GUARD_COMMON,
            )

    def test_valid_sources_index_their_variables_and_flatten(self) -> None:
        gridded = GriddedSource(
            data=torch.zeros(2, 2, 1, GRID, 3),
            coordinates=torch.zeros(GRID, 2),
            layout=GRIDDED_LAYOUT,
            **_GUARD_COMMON,
        )
        assert gridded.name_to_index == {"a": 0, "b": 1, "c": 2}
        assert tuple(gridded.flatten().data.shape) == (2 * GRID, 2 * 3)

        tabular = TabularSource(
            data=[torch.zeros(GRID, 3)],
            coordinates=[torch.zeros(GRID, 2)],
            timedeltas=[torch.zeros(GRID)],
            boundaries=[(slice(0, GRID),)],
            layout=TABULAR_LAYOUT,
            **_GUARD_COMMON,
        )
        assert tabular.name_to_index == {"a": 0, "b": 1, "c": 2}
        assert tuple(tabular.flatten().data.shape) == (GRID, 3)
