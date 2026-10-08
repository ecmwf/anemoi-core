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

from anemoi.models.data import Batch
from anemoi.models.data import GriddedSample
from anemoi.models.data import TabularSample
from anemoi.models.data import TensorLayout
from anemoi.models.data import create_batched_struct
from anemoi.models.data.sample import sample_registry

LATS = [0.0, 45.0, 90.0]
LONS = [0.0, 90.0, 180.0]


def _gridded(**overrides):
    kwargs = {
        "data_type": "gridded",
        "data": torch.zeros(2, 1, 3, 2),
        "variables": ["a", "b"],
        "layout": ("time", "ensemble", "grid", "variables"),
        "latitudes": LATS,
        "longitudes": LONS,
    }
    return create_batched_struct(**(kwargs | overrides))


def _tabular(**overrides):
    kwargs = {
        "data_type": "tabular",
        "data": torch.zeros(1, 3, 1),
        "variables": ["a"],
        "layout": ("ensemble", "grid", "variables"),
        "latitudes": LATS,
        "longitudes": LONS,
        "timedeltas": [0.0, 0.0, 3600.0],
        "boundaries": [(0, 2), (2, 3)],
    }
    return create_batched_struct(**(kwargs | overrides))


def test_gridded_converts_layout_and_degrees():
    sample = _gridded()
    assert isinstance(sample, GriddedSample)
    assert sample.layout == TensorLayout(time=0, ensemble=1, grid=2, variables=3)
    assert sample.grid_size == 3
    torch.testing.assert_close(sample.coordinates[:, 0], torch.deg2rad(torch.tensor(LATS)))
    torch.testing.assert_close(sample.coordinates[:, 1], torch.deg2rad(torch.tensor(LONS)))


def test_tabular_converts_boundaries_and_timedeltas():
    sample = _tabular()
    assert isinstance(sample, TabularSample)
    assert sample.boundaries == (slice(0, 2), slice(2, 3))
    assert sample.timedeltas.dtype == torch.float32


def test_layout_object_and_slices_are_accepted():
    sample = _tabular(layout=TensorLayout(ensemble=0, grid=1, variables=2), boundaries=(slice(0, 3),))
    assert sample.boundaries == (slice(0, 3),)


@pytest.mark.parametrize("build", [_gridded, _tabular])
def test_data_is_optional(build):
    sample = build(data=None, variables=[])
    assert sample.data is None
    with pytest.raises(ValueError, match="without data"):
        Batch.collate({"ds": sample})


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"data_type": "unknown"}, "Cannot find 'unknown'"),
        ({"longitudes": [0.0, 1.0]}, "same points"),
        ({"data": torch.zeros(1, 3, 2)}, "does not match the layout"),
        ({"variables": ["a"]}, "2 variables but 1 names"),
        ({"data": torch.zeros(2, 1, 4, 2)}, "4 points but 3 coordinates"),
    ],
)
def test_gridded_validation(overrides, match):
    with pytest.raises(ValueError, match=match):
        _gridded(**overrides)


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"timedeltas": None}, "require timedeltas and boundaries"),
        ({"timedeltas": [0.0]}, "1 timedeltas were given for 3 points"),
        ({"boundaries": [(0, 4)]}, "extend past"),
    ],
)
def test_tabular_validation(overrides, match):
    with pytest.raises(ValueError, match=match):
        _tabular(**overrides)


def test_registry_holds_both_kinds():
    assert sample_registry.lookup("gridded") is GriddedSample
    assert sample_registry.lookup("tabular") is TabularSample


@pytest.mark.parametrize(
    ("build", "overrides"),
    [(_gridded, {"boundaries": [(0, 3)]}), (_tabular, {"grid_size": 3})],
)
def test_arguments_of_the_other_kind_are_rejected(build, overrides):
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        build(**overrides)


def test_sharded_grid_size_defaults_to_full_grid():
    sample = _gridded(shard_sizes=[3, 2])
    assert sample.grid_size == 5
    assert _gridded(shard_sizes=[3, 2], grid_size=5).grid_size == 5


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"shard_sizes": [3, 2], "grid_size": 4}, "does not match the full grid of 5"),
        ({"shard_sizes": [2, 2]}, "do not match any of the shard sizes"),
        ({"grid_size": 4}, "does not match the full grid of 3"),
    ],
)
def test_grid_size_is_checked_against_sharding(overrides, match):
    with pytest.raises(ValueError, match=match):
        _gridded(**overrides)


def test_collate_matches_direct_construction():
    batch = Batch.collate([{"ds": _gridded()}, {"ds": _gridded()}])
    assert batch["ds"].data.shape == (2, 2, 1, 3, 2)
    torch.testing.assert_close(batch["ds"].coordinates[:, 0], torch.deg2rad(torch.tensor(LATS)))


def test_tabular_collate_keeps_one_entry_per_sample():
    source = Batch.collate([{"obs": _tabular()}, {"obs": _tabular(shard_sizes=None)}])["obs"]
    assert len(source.data) == len(source.coordinates) == len(source.timedeltas) == 2
    assert source.boundaries[0] == (slice(0, 2), slice(2, 3))
    assert source.shard_sizes is None


def test_tabular_collate_rejects_mixed_sharding():
    sharded = _tabular(shard_sizes=[[2], [1]])
    with pytest.raises(ValueError, match="mixes sharded and unsharded"):
        Batch.collate([{"obs": sharded}, {"obs": _tabular()}])


@pytest.mark.parametrize("build", [_gridded, _tabular])
def test_collate_rejects_mixed_data_presence(build):
    with pytest.raises(ValueError, match="mixes samples with and without data"):
        Batch.collate([{"ds": build()}, {"ds": build(data=None)}])


def test_collate_accepts_samples_sharing_statistics():
    statistics = {"mean": torch.zeros(2)}
    source = Batch.collate([{"ds": _gridded(statistics=statistics)}, {"ds": _gridded(statistics=statistics)}])["ds"]
    assert source.statistics is statistics


def test_tabular_layout_without_batch_axis():
    source = Batch.collate([{"obs": _tabular()}, {"obs": _tabular()}])["obs"]
    assert source.layout == TensorLayout(ensemble=0, grid=1, variables=2)
