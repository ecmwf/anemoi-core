# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import datetime

import numpy as np
from pytest_mock import MockFixture

from anemoi.training.data.loading import CrossDatasetLoadingPolicy
from anemoi.training.data.relative_time_indices import compute_relative_date_indices
from anemoi.training.data.usable_indices import ReaderAnchors
from anemoi.training.tasks import TemporalDownscaler


def test_temporal_downscaler_interior_offsets_only() -> None:
    """No boundaries: only interior interpolation steps are produced."""
    task = TemporalDownscaler(
        input_timestep="6h",
        output_timestep="2h",
        output_left_boundary=False,
        output_right_boundary=False,
    )
    expected = [datetime.timedelta(hours=2), datetime.timedelta(hours=4)]
    assert task._output_offsets == expected


def test_temporal_downscaler_left_boundary_included() -> None:
    """output_left_boundary=True adds t=0 to the output offsets."""
    task = TemporalDownscaler(
        input_timestep="6h",
        output_timestep="2h",
        output_left_boundary=True,
        output_right_boundary=False,
    )
    expected = [datetime.timedelta(hours=0), datetime.timedelta(hours=2), datetime.timedelta(hours=4)]
    assert task._output_offsets == expected


def test_temporal_downscaler_right_boundary_included() -> None:
    """output_right_boundary=True adds t=input_timestep to the output offsets."""
    task = TemporalDownscaler(
        input_timestep="6h",
        output_timestep="2h",
        output_left_boundary=False,
        output_right_boundary=True,
    )
    expected = [datetime.timedelta(hours=2), datetime.timedelta(hours=4), datetime.timedelta(hours=6)]
    assert task._output_offsets == expected


def test_temporal_downscaler_both_boundaries_included() -> None:
    """Both boundaries: offsets span the full [0h, input_timestep] range."""
    task = TemporalDownscaler(
        input_timestep="6h",
        output_timestep="2h",
        output_left_boundary=True,
        output_right_boundary=True,
    )
    expected = [
        datetime.timedelta(hours=0),
        datetime.timedelta(hours=2),
        datetime.timedelta(hours=4),
        datetime.timedelta(hours=6),
    ]
    assert task._output_offsets == expected


def test_temporal_downscaler_num_output_timesteps_matches_offsets() -> None:
    """num_output_timesteps equals the length of output_offsets."""
    task = TemporalDownscaler(
        input_timestep="6h",
        output_timestep="2h",
        output_left_boundary=True,
        output_right_boundary=True,
    )
    assert task.num_output_timesteps == len(task._output_offsets) == 4


def test_temporal_downscaler_input_offsets_are_boundary_pair() -> None:
    """Input offsets are always [0h, input_timestep] regardless of output settings."""
    task = TemporalDownscaler(input_timestep="6h", output_timestep="2h")
    assert task._input_offsets == [datetime.timedelta(0), datetime.timedelta(hours=6)]


def test_temporal_downscaler_offsets_are_loaded_from_one_domain(mocker: MockFixture) -> None:
    task = TemporalDownscaler(input_timestep="6h", output_timestep="2h")
    readers = {}
    for name, frequency in (("data", datetime.timedelta(hours=1)), ("data_2", datetime.timedelta(hours=2))):
        reader = mocker.MagicMock()
        reader.frequency = frequency
        reader.sampling = None
        reader.valid_anchors.return_value = ReaderAnchors(
            np.array([0], dtype="datetime64[s]"),
            np.array([0]),
            np.array([0]),
        )
        reader.measure_from_reference.side_effect = lambda sample, *_args: sample
        readers[name] = reader

    relative_date_indices = compute_relative_date_indices(task, readers)
    assert relative_date_indices == {"data": [0, 2, 4, 6], "data_2": [0, 1, 2, 3]}

    dataset = CrossDatasetLoadingPolicy().build_dataset(readers, relative_date_indices)
    sample = dataset[0]

    assert set(sample) == {"data"}
    readers["data"].get_sample.assert_called_once_with(0, slice(0, 8, 2))
    readers["data_2"].get_sample.assert_not_called()
