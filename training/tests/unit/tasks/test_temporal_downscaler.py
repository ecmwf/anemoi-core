# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import datetime

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
    assert task.offsets.output == expected


def test_temporal_downscaler_left_boundary_included() -> None:
    """output_left_boundary=True adds t=0 to the output offsets."""
    task = TemporalDownscaler(
        input_timestep="6h",
        output_timestep="2h",
        output_left_boundary=True,
        output_right_boundary=False,
    )
    expected = [datetime.timedelta(hours=0), datetime.timedelta(hours=2), datetime.timedelta(hours=4)]
    assert task.offsets.output == expected


def test_temporal_downscaler_right_boundary_included() -> None:
    """output_right_boundary=True adds t=input_timestep to the output offsets."""
    task = TemporalDownscaler(
        input_timestep="6h",
        output_timestep="2h",
        output_left_boundary=False,
        output_right_boundary=True,
    )
    expected = [datetime.timedelta(hours=2), datetime.timedelta(hours=4), datetime.timedelta(hours=6)]
    assert task.offsets.output == expected


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
    assert task.offsets.output == expected


def test_temporal_downscaler_num_output_timesteps_matches_offsets() -> None:
    """num_output_timesteps equals the length of output_offsets."""
    task = TemporalDownscaler(
        input_timestep="6h",
        output_timestep="2h",
        output_left_boundary=True,
        output_right_boundary=True,
    )
    assert task.num_output_timesteps == len(task.offsets.output) == 4


def test_temporal_downscaler_input_offsets_are_boundary_pair() -> None:
    """Input offsets are always [0h, input_timestep] regardless of output settings."""
    task = TemporalDownscaler(input_timestep="6h", output_timestep="2h")
    assert task.offsets.input == [datetime.timedelta(0), datetime.timedelta(hours=6)]
