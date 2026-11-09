# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import datetime
from types import SimpleNamespace

import pytest
import torch

from anemoi.training.tasks.spatial_downscaler import SpatialDownscaler

# ---------------------------------------------------------------------------
# Offset configuration
# ---------------------------------------------------------------------------


def test_default_offsets_are_single_zero() -> None:
    """Without explicit offsets, both input and output default to [timedelta(0)]."""
    task = SpatialDownscaler(
        input_datasets=["in_lres"],
        target_datasets=["out_hres"],
    )
    assert task._input_offsets == [datetime.timedelta(0)]
    assert task._output_offsets == [datetime.timedelta(0)]
    assert task._offsets == [datetime.timedelta(0)]


def test_multiple_offsets_parsed_correctly() -> None:
    """Three offsets are parsed and sorted for both input and output."""
    task = SpatialDownscaler(
        input_datasets=["in_lres"],
        target_datasets=["out_hres"],
        input_offsets=["12H", "0H", "6H"],
        output_offsets=["0H", "6H", "12H"],
    )
    expected = [
        datetime.timedelta(hours=0),
        datetime.timedelta(hours=6),
        datetime.timedelta(hours=12),
    ]
    assert task._input_offsets == expected
    assert task._output_offsets == expected
    assert task._offsets == expected


# ---------------------------------------------------------------------------
# Batch index helpers
# ---------------------------------------------------------------------------


def test_batch_indices_are_the_positions_of_the_shared_offsets() -> None:
    task = SpatialDownscaler(
        input_datasets=["in_lres"],
        target_datasets=["out_hres"],
        input_offsets=["0H", "6H", "12H"],
        output_offsets=["0H", "6H", "12H"],
    )
    assert task.get_batch_input_indices() == [0, 1, 2]
    assert task.get_batch_output_indices() == [0, 1, 2]


# ---------------------------------------------------------------------------
# Dataset roles must agree with the model
# ---------------------------------------------------------------------------


def _roles_task() -> SpatialDownscaler:
    return SpatialDownscaler(input_datasets=["in_lres", "in_hres"], target_datasets=["out_hres"])


def test_validate_dataset_roles_accepts_the_model_roles_in_any_order() -> None:
    _roles_task().validate_dataset_roles(input_datasets=["in_hres", "in_lres"], target_datasets=["out_hres"])


def test_validate_dataset_roles_rejects_inputs_the_model_does_not_read() -> None:
    with pytest.raises(ValueError, match=r"input_datasets.*\['in_hres', 'in_lres'\].*\['in_lres'\]"):
        _roles_task().validate_dataset_roles(input_datasets=["in_lres"], target_datasets=["out_hres"])


def test_validate_dataset_roles_rejects_targets_the_model_does_not_predict() -> None:
    with pytest.raises(ValueError, match=r"target_datasets.*\['out_hres'\].*\['out_hres', 'out_lres'\]"):
        _roles_task().validate_dataset_roles(
            input_datasets=["in_lres", "in_hres"],
            target_datasets=["out_hres", "out_lres"],
        )


# ---------------------------------------------------------------------------
# get_inputs / get_targets behaviour
# ---------------------------------------------------------------------------


def test_get_inputs_and_get_targets_split_the_batch_by_role_and_keep_every_offset() -> None:
    n_offsets, nvar = 3, 4
    task = SpatialDownscaler(
        input_datasets=["in_lres", "in_hres"],
        target_datasets=["out_hres"],
        input_offsets=["0H", "6H", "12H"],
        output_offsets=["0H", "6H", "12H"],
    )
    batch = {name: torch.randn(1, n_offsets, 1, 10, nvar) for name in ("in_lres", "in_hres", "out_hres")}
    indices = SimpleNamespace(data=SimpleNamespace(input=SimpleNamespace(full=torch.arange(nvar))))
    data_indices = dict.fromkeys(batch, indices)

    x = task.get_inputs(batch, data_indices=data_indices)
    y = task.get_targets(batch)

    assert set(x) == {"in_lres", "in_hres"}
    assert set(y) == {"out_hres"}
    assert x["in_lres"].shape[1] == n_offsets
    assert y["out_hres"].shape[1] == n_offsets


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------


def _make_metadata_dict(dataset_names: list[str]) -> dict:
    """Skeleton of the metadata the trainer builds before the task fills it in."""
    return {
        "task": None,
        "metadata_inference": {
            "task": None,
            "dataset_names": dataset_names,
            **{name: {} for name in dataset_names},
        },
    }


def test_fill_metadata_records_explicit_offsets_no_feedback_and_no_rollout_shift() -> None:
    """With ``timestep: 0H`` inference would derive all offsets as zero, so they must be explicit.

    Downscaling is not autoregressive (empty advance map), and the stride between
    windows is an inference choice, so no ``rollout_shift`` is written.
    """
    task = SpatialDownscaler(
        input_datasets=["in_lres"],
        target_datasets=["out_hres"],
        input_offsets=["0H", "6H"],
        output_offsets=["6H"],
    )
    md_dict = _make_metadata_dict(["in_lres", "out_hres"])

    task.fill_metadata(md_dict)

    timesteps = md_dict["metadata_inference"]["out_hres"]["timesteps"]
    assert timesteps["timestep"] == "0H"
    assert timesteps["input_offsets"] == ["0h", "6h"]
    assert timesteps["output_offsets"] == ["6h"]
    assert "rollout_shift" not in timesteps
    assert md_dict["metadata_inference"]["in_lres"]["timesteps"]["advance_map"] == {"inin": [], "outin": []}
