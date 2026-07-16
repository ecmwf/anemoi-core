# (C) Copyright 2026 Anemoi contributors.

import datetime
from types import SimpleNamespace

import pytest
import torch

from anemoi.training.data.relative_time_indices import compute_relative_date_indices
from anemoi.training.tasks import SpatialDownscaler


def _task(input_offsets: list[int], output_offsets: list[int], **kwargs) -> SpatialDownscaler:
    defaults = {
        "input_datasets": ["input"],
        "output_datasets": ["output"],
        "frequency": "6h",
    }
    defaults.update(kwargs)
    return SpatialDownscaler(input_offsets=input_offsets, output_offsets=output_offsets, **defaults)


def _index_collection_for(names: tuple[str, ...]) -> dict[str, SimpleNamespace]:
    return {
        name: SimpleNamespace(data=SimpleNamespace(input=SimpleNamespace(full=torch.tensor([0])))) for name in names
    }


# ── Construction: the four supported offset forms ─────────────────────────────


OFFSET_CASES = [
    # (input_offsets, output_offsets, expected_input_positions, expected_output_positions, expected_output_to_input)
    ([0], [0], [0], [0], [0]),
    ([0, 1], [0, 1], [0, 1], [0, 1], [0, 1]),
    ([-1, 0, 1, 2], [0, 1], [0, 1, 2, 3], [1, 2], [1, 2]),
    ([-1, 0, 1], [0], [0, 1, 2], [1], [1]),
]


@pytest.mark.parametrize(
    ("input_offsets", "output_offsets", "expected_input_positions", "expected_output_positions", "expected_o2i"),
    OFFSET_CASES,
)
def test_construction_and_batch_positions(
    input_offsets: list[int],
    output_offsets: list[int],
    expected_input_positions: list[int],
    expected_output_positions: list[int],
    expected_o2i: list[int],
) -> None:
    task = _task(input_offsets, output_offsets)

    assert task.get_batch_input_indices() == expected_input_positions
    assert task.get_batch_output_indices() == expected_output_positions
    assert task.output_to_input_positions() == expected_o2i


def test_offsets_are_frequency_multiples() -> None:
    """Integer offsets become physical offsets at construction: offset x frequency."""
    task = _task([-1, 0, 1], [0], frequency="6h")

    assert task.get_input_offsets() == [
        datetime.timedelta(hours=-6),
        datetime.timedelta(hours=0),
        datetime.timedelta(hours=6),
    ]
    assert task.get_output_offsets() == [datetime.timedelta(hours=0)]
    assert task._get_timestep_for_metadata() == "6h"


def test_relative_date_indices_equal_the_configured_integers() -> None:
    """The round trip: integers x frequency // frequency == the same integers.

    ``compute_relative_date_indices`` is the existing helper that turns task offsets into
    dataset row positions; with this task the positions ARE the configured integers.
    """
    task = _task([-1, 0, 1, 2], [0, 1])
    readers = {
        "input": SimpleNamespace(frequency=datetime.timedelta(hours=6)),
        "output": SimpleNamespace(frequency=datetime.timedelta(hours=6)),
    }

    relative_date_indices = compute_relative_date_indices(task, readers)

    expected = [-1, 0, 1, 2]
    assert relative_date_indices == {"input": expected, "output": expected}


def test_incompatible_frequency_fails_in_existing_helper() -> None:
    """A task frequency the data cannot represent fails loudly in the existing helper."""
    task = _task([-1, 0, 1], [0], frequency="6h")
    readers = {"input": SimpleNamespace(frequency=datetime.timedelta(hours=4))}

    with pytest.raises(ValueError, match="not compatible"):
        compute_relative_date_indices(task, readers)


# ── Invariant violations ───────────────────────────────────────────────────────


def test_output_offset_not_in_inputs_raises() -> None:
    with pytest.raises(ValueError, match="output_offsets"):
        _task([0, 1], [2])


# Types, non-emptiness and uniqueness are enforced by the pydantic task schema (SpatialDownscalerSchema),
# not re-checked in __init__. The subset invariant is the one check kept in the task itself — see
# ``test_output_offset_not_in_inputs_raises``.


# ── Role selection ──────────────────────────────────────────────────────────


def test_get_inputs_and_get_targets_select_named_roles_in_order() -> None:
    task = _task([-1, 0, 1, 2], [0, 1], input_datasets=["input", "output"])
    n_time = 4  # len of the offset union [-1, 0, 1, 2]
    batch = {
        "input": torch.arange(n_time).reshape(1, n_time, 1, 1).float(),
        "output": torch.arange(n_time, 2 * n_time).reshape(1, n_time, 1, 1).float(),
    }
    data_indices = _index_collection_for(("input", "output"))

    inputs = task.get_inputs(batch, data_indices)
    targets = task.get_targets(batch)

    assert list(inputs) == ["input", "output"]
    assert list(targets) == ["output"]
    # 4 input positions (all of them), 2 output positions (indices 1,2 within the union)
    assert inputs["input"].shape[1] == 4
    assert targets["output"].shape[1] == 2


def test_get_inputs_missing_dataset_raises_key_error() -> None:
    task = _task([0], [0], input_datasets=["input", "missing"])
    batch = {
        "input": torch.zeros(1, 1, 1, 1),
        "output": torch.zeros(1, 1, 1, 1),
    }
    data_indices = _index_collection_for(("input", "output"))

    with pytest.raises(KeyError, match="missing"):
        task.get_inputs(batch, data_indices)


def test_get_targets_missing_dataset_raises_key_error() -> None:
    task = _task([0], [0], output_datasets=["missing"])
    batch = {"input": torch.zeros(1, 1, 1, 1)}

    with pytest.raises(KeyError, match="missing"):
        task.get_targets(batch)


# ── fill_metadata ────────────────────────────────────────────────────────────


def test_fill_metadata_records_integer_offsets_and_roles() -> None:
    task = _task([-1, 0, 1, 2], [0, 1], input_datasets=["input", "output"])

    md_dict = {
        "metadata_inference": {
            "dataset_names": ["input", "output"],
            "input": {},
            "output": {},
        },
    }

    task.fill_metadata(md_dict)

    assert md_dict["task"] == "spatial-downscaler"
    assert md_dict["input_datasets"] == ["input", "output"]
    assert md_dict["output_datasets"] == ["output"]
    for dataset_name in ("input", "output"):
        timesteps = md_dict["metadata_inference"][dataset_name]["timesteps"]
        assert timesteps["input_offsets"] == [-1, 0, 1, 2]
        assert timesteps["output_offsets"] == [0, 1]
        assert timesteps["timestep"] == "6h"


def test_schema_rejects_empty_lists() -> None:
    """Non-emptiness is owned by the config-time schema, not the task."""
    from pydantic import ValidationError

    from anemoi.training.schemas.tasks import SpatialDownscalerSchema

    base = {
        "_target_": "anemoi.training.tasks.SpatialDownscaler",
        "input_datasets": ["input"],
        "output_datasets": ["output"],
        "input_offsets": [0],
        "output_offsets": [0],
        "frequency": "6h",
    }
    for field in ("input_datasets", "output_datasets", "input_offsets", "output_offsets"):
        with pytest.raises(ValidationError):
            SpatialDownscalerSchema(**{**base, field: []})
