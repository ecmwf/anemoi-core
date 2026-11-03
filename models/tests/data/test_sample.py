# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import dataclasses

import pytest
import torch

from anemoi.models.data import Batch
from anemoi.models.data import GriddedSample
from anemoi.models.data import TabularSample
from anemoi.models.data import TensorLayout
from anemoi.models.data.sample import sample_registry
from anemoi.models.data.sources import GriddedTemplate
from anemoi.models.data.sources import TabularTemplate
from tests.sample_builders import gridded_sample
from tests.sample_builders import latitudes
from tests.sample_builders import longitudes
from tests.sample_builders import tabular_sample

# The validation tests below match messages that name these sizes ("3 coordinates",
# "2 variables", "extend past the 3 points"), so the samples here pin them.
N_POINTS = 3
LATS = latitudes(N_POINTS)
LONS = longitudes(N_POINTS)


def _gridded(**overrides):
    return gridded_sample(n_points=N_POINTS, n_variables=2, **overrides)


def _tabular(**overrides):
    return tabular_sample(n_points=N_POINTS, n_variables=1, **overrides)


def test_gridded_converts_layout_and_degrees():
    sample = _gridded()
    assert isinstance(sample, GriddedSample)
    assert sample.layout == TensorLayout(time=0, ensemble=1, grid=2, variables=3)
    assert sample.grid_size == 3
    torch.testing.assert_close(sample.coordinates[:, 0], torch.deg2rad(torch.tensor(LATS)))
    torch.testing.assert_close(sample.coordinates[:, 1], torch.deg2rad(torch.tensor(LONS)))


@pytest.mark.parametrize("build", [_gridded, _tabular])
def test_degrees_are_converted_in_float64_and_rounded_once(build):
    # Rounding the degrees to float32 before the conversion changes the last bit of some radians.
    lats = torch.tensor([12.3456789, -45.6789012, 67.8901234], dtype=torch.float64)
    lons = torch.tensor([123.456789, 234.567890, 345.678901], dtype=torch.float64)
    sample = build(latitudes=lats.numpy(), longitudes=lons.numpy())
    coordinates = sample.coordinates[0] if isinstance(sample.coordinates, list) else sample.coordinates
    assert coordinates.dtype == torch.float32
    assert torch.equal(coordinates[:, 0], torch.deg2rad(lats).to(torch.float32))
    assert torch.equal(coordinates[:, 1], torch.deg2rad(lons).to(torch.float32))


def test_tabular_converts_boundaries_and_timedeltas():
    sample = _tabular()
    assert isinstance(sample, TabularSample)
    assert sample.boundaries == (slice(0, 2), slice(2, 3))
    assert sample.timedeltas.dtype == torch.float32


def test_layout_object_and_slices_are_accepted():
    sample = _tabular(layout=TensorLayout(ensemble=0, grid=1, variables=2), boundaries=(slice(0, 3),))
    assert sample.boundaries == (slice(0, 3),)


#: What the interface passes, besides the payload, to build a template.
TEMPLATE_SIZES = {"name": "ds", "ensemble_size": 1, "batch_size": 1, "time_size": 2}


def test_gridded_payload_without_data_builds_a_template():
    template = _gridded(data=None, **TEMPLATE_SIZES)
    assert isinstance(template, GriddedTemplate)
    assert template.name == "ds"
    assert (template.batch_size, template.ensemble_size, template.time_size) == (1, 1, 2)
    torch.testing.assert_close(template.coordinates[:, 0], torch.deg2rad(torch.tensor(LATS)))


def test_tabular_payload_without_data_builds_a_template():
    template = _tabular(data=None, **(TEMPLATE_SIZES | {"batch_size": 2, "ensemble_size": 3}))
    assert isinstance(template, TabularTemplate)
    assert template.name == "ds"
    # one entry per sample, each holding the payload's geometry
    assert (template.batch_size, template.ensemble_size) == (2, 3)
    assert template.time_size == 2, "the time windows come from the boundaries"
    assert template.boundaries == [(slice(0, 2), slice(2, 3))] * 2
    assert template.node_counts == [3, 3]
    for coordinates, timedeltas in zip(template.coordinates, template.timedeltas, strict=True):
        torch.testing.assert_close(coordinates[:, 0], torch.deg2rad(torch.tensor(LATS)))
        assert timedeltas.dtype == torch.float32


def test_tabular_template_checks_its_time_windows():
    with pytest.raises(ValueError, match="extend past the 3 points"):
        _tabular(data=None, boundaries=[(0, 4)], **TEMPLATE_SIZES)


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


def test_collate_rejects_mixing_samples_and_templates():
    with pytest.raises(TypeError, match="single BaseSample subclass"):
        Batch.collate([{"ds": _gridded()}, {"ds": _gridded(data=None, **TEMPLATE_SIZES)}])


def test_collate_accepts_samples_sharing_statistics():
    statistics = {"mean": torch.zeros(2)}
    source = Batch.collate([{"ds": _gridded(statistics=statistics)}, {"ds": _gridded(statistics=statistics)}])["ds"]
    assert source.statistics is statistics


def test_tabular_layout_without_batch_axis():
    source = Batch.collate([{"obs": _tabular()}, {"obs": _tabular()}])["obs"]
    assert source.layout == TensorLayout(ensemble=0, grid=1, variables=2)


def _assert_same_field(actual, expected, path):
    if isinstance(expected, torch.Tensor):
        assert isinstance(actual, torch.Tensor), f"{path}: expected a tensor, got {type(actual).__name__}"
        assert actual.dtype == expected.dtype, f"{path}: dtype {actual.dtype} != {expected.dtype}"
        assert actual.device == expected.device, f"{path}: device {actual.device} != {expected.device}"
        assert torch.equal(actual, expected), f"{path}: values differ"
    elif isinstance(expected, (list, tuple)):
        assert type(actual) is type(expected), f"{path}: {type(actual).__name__} != {type(expected).__name__}"
        assert len(actual) == len(expected), f"{path}: length {len(actual)} != {len(expected)}"
        for i, (a, e) in enumerate(zip(actual, expected, strict=True)):
            _assert_same_field(a, e, f"{path}[{i}]")
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys(), f"{path}: keys {actual.keys()} != {expected.keys()}"
        for key in expected:
            _assert_same_field(actual[key], expected[key], f"{path}[{key!r}]")
    else:
        assert actual == expected, f"{path}: {actual!r} != {expected!r}"


def _assert_same_template(actual, expected):
    assert type(actual) is type(expected)
    for template_field in dataclasses.fields(expected):
        name = template_field.name
        _assert_same_field(getattr(actual, name), getattr(expected, name), name)


# A template is derived from a collated source in training (Source.template) and built
# from a payload without data in inference (create_batched_struct(data=None)). Both must
# describe the same batch when given the same geometry.
@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize(
    ("build", "sharding", "sizes"),
    [
        pytest.param(_gridded, {}, {"ensemble_size": 1, "time_size": 2}, id="gridded"),
        pytest.param(_gridded, {"shard_sizes": [3, 2]}, {"ensemble_size": 1, "time_size": 2}, id="gridded-sharded"),
        pytest.param(_tabular, {}, {"ensemble_size": 1}, id="tabular"),
        pytest.param(_tabular, {"shard_sizes": [[1, 1], [1, 0]]}, {"ensemble_size": 1}, id="tabular-sharded"),
    ],
)
def test_template_from_source_matches_template_from_payload(build, sharding, sizes, batch_size):
    statistics = {"mean": torch.arange(len(build().variables), dtype=torch.float32)}
    payload = sharding | {"statistics": statistics}

    source = Batch.collate([{"ds": build(**payload)} for _ in range(batch_size)])["ds"]
    from_source = source.template()
    from_payload = build(data=None, name="ds", batch_size=batch_size, **sizes, **payload)

    _assert_same_template(from_payload, from_source)
