# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""The time information (``time_offsets``, ``reference_date``) of samples, sources and templates."""

import datetime

import numpy as np
import pytest
import torch
from rich.console import Console

from anemoi.models.data import Batch
from tests.sample_builders import gridded_sample
from tests.sample_builders import tabular_sample

H = datetime.timedelta(hours=1)
REF = datetime.datetime(2024, 1, 1, 12)
OFFSETS = (-6 * H, 0 * H)


def _gridded(n_times: int = 2, **overrides):
    return gridded_sample(n_points=3, n_variables=2, n_times=n_times, **overrides)


def _tabular(**overrides):
    # Two windows: points [0, 2) and [2, 3).
    return tabular_sample(n_points=3, n_variables=1, **overrides)


def _timed(build, reference=REF, offsets=OFFSETS):
    return build().with_time_reference(reference, offsets)


def _collate(*samples):
    return Batch.collate([{"ds": s} for s in samples])["ds"]


def test_sample_defaults_to_unknown_times():
    sample = _gridded()
    assert sample.time_offsets is None
    assert sample.reference_date is None
    assert _collate(sample, _gridded()).reference_date is None


def test_numpy_times_are_converted():
    sample = _gridded().with_time_reference(
        np.datetime64("2024-01-01T12:00:00", "ns"),
        np.array([-6, 0], dtype="timedelta64[h]"),
    )
    assert sample.reference_date == REF
    assert sample.time_offsets == OFFSETS


@pytest.mark.parametrize("build", [_gridded, _tabular])
def test_offsets_must_match_the_time_size(build):
    with pytest.raises(ValueError, match="has 2 time steps but 3 time offsets"):
        build().with_time_reference(REF, (0 * H, H, 2 * H))


def test_tabular_reference_shifts_point_timedeltas_per_window():
    base = _tabular()
    sample = base.with_time_reference(REF, OFFSETS)
    expected = base.timedeltas + torch.tensor([-6 * 3600.0, -6 * 3600.0, 0.0])
    torch.testing.assert_close(sample.timedeltas, expected)


@pytest.mark.parametrize("build", [_gridded, _tabular])
def test_collate_shares_offsets_and_keeps_reference_dates_per_sample(build):
    later = REF + 6 * H
    source = _collate(_timed(build), _timed(build, reference=later))
    assert source.time_offsets == OFFSETS
    assert source.reference_date == [REF, later]


@pytest.mark.parametrize("build", [_gridded, _tabular])
def test_collate_rejects_different_offsets(build):
    with pytest.raises(ValueError, match="different 'time_offsets'"):
        _collate(_timed(build), _timed(build, offsets=(-3 * H, 0 * H)))


def test_collate_rejects_mixed_known_and_unknown_reference_dates():
    with pytest.raises(ValueError, match="with and without 'reference_date'"):
        _collate(_timed(_gridded), _gridded().with_time_reference(None, OFFSETS))


@pytest.mark.parametrize("indices", [1, [1], slice(1, 2)])
@pytest.mark.parametrize("build", [_gridded, _tabular])
def test_select_time_subsets_offsets_without_re_referencing(build, indices):
    source = _collate(_timed(build))
    selected = source.select_time(indices)
    assert selected.time_offsets == (0 * H,)
    assert selected.reference_date == [REF]


def test_tabular_concat_time_joins_offsets():
    source = _collate(_timed(_tabular))
    joined = source.select_time([1]).concat_time(source.select_time([0]))
    assert joined.time_offsets == (0 * H, -6 * H)


def test_tabular_concat_time_rejects_different_reference_dates():
    a = _collate(_timed(_tabular))
    b = _collate(_timed(_tabular, reference=REF + H))
    with pytest.raises(ValueError, match="reference dates"):
        a.concat_time(b)


@pytest.mark.parametrize("build", [_gridded, _tabular])
def test_shift_time_reference_keeps_absolute_times(build):
    source = _collate(_timed(build))
    shifted = Batch(sources={"ds": source}).shift_time_reference(6 * H)["ds"]

    assert shifted.reference_date == [REF + 6 * H]
    assert shifted.time_offsets == (-12 * H, -6 * H)
    before = [source.reference_date[0] + o for o in source.time_offsets]
    after = [shifted.reference_date[0] + o for o in shifted.time_offsets]
    assert before == after
    if source.is_tabular:
        torch.testing.assert_close(shifted.timedeltas[0], source.timedeltas[0] - 6 * 3600.0)


@pytest.mark.parametrize("build", [_gridded, _tabular])
def test_template_round_trip_keeps_times(build):
    source = _collate(_timed(build))
    template = source.template()
    assert template.time_offsets == OFFSETS
    assert template.reference_date == [REF]

    rebuilt = template.unflatten(source.flatten().values)
    assert rebuilt.time_offsets == OFFSETS
    assert rebuilt.reference_date == [REF]

    shifted = template.shift_time_reference(H)
    assert shifted.time_offsets == (-7 * H, -H)
    assert shifted.reference_date == [REF + H]


def test_gridded_source_rejects_offsets_for_another_time_size():
    source = _collate(_timed(_gridded))
    with pytest.raises(ValueError, match="has 2 time steps but 1 time offsets"):
        source.clone(time_offsets=(0 * H,))


def test_template_payload_repeats_the_reference_date_per_sample():
    template = _gridded(
        values=None,
        name="ds",
        ensemble_size=1,
        batch_size=2,
        time_size=2,
        time_offsets=OFFSETS,
        reference_date=REF,
    )
    assert template.reference_date == [REF, REF]
    assert template.time_offsets == OFFSETS


def test_tree_shows_times():
    console = Console(record=True, width=120)
    with console.capture() as capture:
        console.print(_collate(_timed(_gridded)).tree())
    text = capture.get()
    assert "Time offsets: [-6h, 0h]" in text
    assert "2024-01-01T12:00:00" in text
