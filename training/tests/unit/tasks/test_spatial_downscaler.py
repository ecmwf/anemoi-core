# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import datetime

import pytest

from anemoi.training.tasks import SpatialDownscaler


def test_default_is_one_snapshot_at_zero() -> None:
    task = SpatialDownscaler()

    assert task._input_offsets == [datetime.timedelta(0)]
    assert task._output_offsets == [datetime.timedelta(0)]
    assert list(task.steps("training")) == [{}]


def test_inputs_and_outputs_share_the_sorted_offsets() -> None:
    task = SpatialDownscaler(offsets=["12H", "0H", "6H"])

    expected = [datetime.timedelta(hours=hours) for hours in (0, 6, 12)]
    assert task._input_offsets == expected
    assert task._output_offsets == expected
    assert task.get_batch_input_indices() == [0, 1, 2]
    assert task.get_batch_output_indices() == [0, 1, 2]


@pytest.mark.parametrize(
    ("offsets", "message"),
    [(["0H", "6H", "18H"], "evenly spaced"), (["0H", "6H", "6H"], "distinct")],
)
def test_offsets_must_be_distinct_and_evenly_spaced(offsets: list[str], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        SpatialDownscaler(offsets=offsets)


def test_fill_metadata_records_the_offsets_and_no_feedback() -> None:
    """With ``timestep: 0H`` inference would derive every offset as zero, so they are written out."""
    task = SpatialDownscaler(offsets=["0H", "6H"])
    md_dict = {"metadata_inference": {"dataset_names": ["lres", "hres"], "lres": {}, "hres": {}}}

    task.fill_metadata(md_dict)

    assert md_dict["metadata_inference"]["task"] == "spatial_downscaler"
    for name in ("lres", "hres"):
        timesteps = md_dict["metadata_inference"][name]["timesteps"]
        assert timesteps["timestep"] == "0H"
        assert timesteps["input_offsets"] == ["0h", "6h"]
        assert timesteps["output_offsets"] == ["0h", "6h"]
        assert timesteps["advance_map"] == {"inin": [], "outin": []}
        assert "rollout_shift" not in timesteps
