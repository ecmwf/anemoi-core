# (C) Copyright 2026 Anemoi contributors.

import datetime
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from anemoi.training.data.relative_time_indices import compute_relative_date_indices
from anemoi.training.tasks import SpatialDownscaler
from anemoi.training.tasks.spatial_downscaler import bind_task_frequency


def _index_collection_for(names: tuple[str, ...]) -> dict[str, SimpleNamespace]:
    return {
        name: SimpleNamespace(data=SimpleNamespace(input=SimpleNamespace(full=torch.tensor([0]))))
        for name in names
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
    task = SpatialDownscaler(
        input_datasets=["input"],
        output_datasets=["output"],
        input_offsets=input_offsets,
        output_offsets=output_offsets,
    )

    assert task.get_batch_input_indices() == expected_input_positions
    assert task.get_batch_output_indices() == expected_output_positions
    assert task.output_to_input_positions() == expected_o2i


# ── Invariant violations ───────────────────────────────────────────────────────


def test_output_offset_not_in_inputs_raises() -> None:
    with pytest.raises(ValueError, match="output_offsets"):
        SpatialDownscaler(
            input_datasets=["input"],
            output_datasets=["output"],
            input_offsets=[0, 1],
            output_offsets=[2],
        )


def test_duplicate_input_offsets_raise() -> None:
    with pytest.raises(ValueError, match="duplicates"):
        SpatialDownscaler(
            input_datasets=["input"],
            output_datasets=["output"],
            input_offsets=[0, 0, 1],
            output_offsets=[0],
        )


def test_duplicate_output_offsets_raise() -> None:
    with pytest.raises(ValueError, match="duplicates"):
        SpatialDownscaler(
            input_datasets=["input"],
            output_datasets=["output"],
            input_offsets=[0, 1],
            output_offsets=[0, 0],
        )


def test_empty_input_offsets_raise() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        SpatialDownscaler(
            input_datasets=["input"],
            output_datasets=["output"],
            input_offsets=[],
            output_offsets=[0],
        )


def test_empty_output_offsets_raise() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        SpatialDownscaler(
            input_datasets=["input"],
            output_datasets=["output"],
            input_offsets=[0],
            output_offsets=[],
        )


def test_bool_offset_is_rejected() -> None:
    with pytest.raises(ValueError, match="integers"):
        SpatialDownscaler(
            input_datasets=["input"],
            output_datasets=["output"],
            input_offsets=[0, True],
            output_offsets=[0],
        )


def test_empty_datasets_raise() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        SpatialDownscaler(input_datasets=[], output_datasets=["output"], input_offsets=[0], output_offsets=[0])


def test_duplicate_datasets_raise() -> None:
    with pytest.raises(ValueError, match="duplicates"):
        SpatialDownscaler(
            input_datasets=["input", "input"],
            output_datasets=["output"],
            input_offsets=[0],
            output_offsets=[0],
        )


# ── Role selection ──────────────────────────────────────────────────────────


def test_get_inputs_and_get_targets_select_named_roles_in_order() -> None:
    task = SpatialDownscaler(
        input_datasets=["input", "output"],
        output_datasets=["output"],
        input_offsets=[-1, 0, 1, 2],
        output_offsets=[0, 1],
    )
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
    task = SpatialDownscaler(
        input_datasets=["input", "missing"],
        output_datasets=["output"],
        input_offsets=[0],
        output_offsets=[0],
    )
    batch = {
        "input": torch.zeros(1, 1, 1, 1),
        "output": torch.zeros(1, 1, 1, 1),
    }
    data_indices = _index_collection_for(("input", "output"))

    with pytest.raises(KeyError, match="missing"):
        task.get_inputs(batch, data_indices)


def test_get_targets_missing_dataset_raises_key_error() -> None:
    task = SpatialDownscaler(
        input_datasets=["input"],
        output_datasets=["missing"],
        input_offsets=[0],
        output_offsets=[0],
    )
    batch = {"input": torch.zeros(1, 1, 1, 1)}

    with pytest.raises(KeyError, match="missing"):
        task.get_targets(batch)


# ── Frequency binding ──────────────────────────────────────────────────────


def test_bind_data_frequency_converts_offsets_to_timedeltas() -> None:
    task = SpatialDownscaler(
        input_datasets=["input"],
        output_datasets=["output"],
        input_offsets=[-1, 0, 1, 2],
        output_offsets=[0, 1],
    )
    task.bind_data_frequency(datetime.timedelta(hours=6))

    assert task.get_offsets() == [
        datetime.timedelta(hours=-6),
        datetime.timedelta(hours=0),
        datetime.timedelta(hours=6),
        datetime.timedelta(hours=12),
    ]
    assert task.get_input_offsets() == [
        datetime.timedelta(hours=-6),
        datetime.timedelta(hours=0),
        datetime.timedelta(hours=6),
        datetime.timedelta(hours=12),
    ]
    assert task.get_output_offsets() == [datetime.timedelta(hours=0), datetime.timedelta(hours=6)]


def test_positional_indices_identical_before_and_after_binding() -> None:
    task = SpatialDownscaler(
        input_datasets=["input"],
        output_datasets=["output"],
        input_offsets=[-1, 0, 1, 2],
        output_offsets=[0, 1],
    )
    before_input = task.get_batch_input_indices()
    before_output = task.get_batch_output_indices()
    before_o2i = task.output_to_input_positions()

    task.bind_data_frequency(datetime.timedelta(hours=6))

    assert task.get_batch_input_indices() == before_input
    assert task.get_batch_output_indices() == before_output
    assert task.output_to_input_positions() == before_o2i


def test_rebinding_same_frequency_is_noop() -> None:
    task = SpatialDownscaler(
        input_datasets=["input"],
        output_datasets=["output"],
        input_offsets=[0],
        output_offsets=[0],
    )
    task.bind_data_frequency(datetime.timedelta(hours=6))
    task.bind_data_frequency(datetime.timedelta(hours=6))  # no-op, must not raise
    assert task.get_offsets() == [datetime.timedelta(hours=0)]


def test_rebinding_different_frequency_raises() -> None:
    task = SpatialDownscaler(
        input_datasets=["input"],
        output_datasets=["output"],
        input_offsets=[0],
        output_offsets=[0],
    )
    task.bind_data_frequency(datetime.timedelta(hours=6))
    with pytest.raises(ValueError, match="already bound"):
        task.bind_data_frequency(datetime.timedelta(hours=1))


def test_bind_data_frequency_rejects_non_positive_timedelta() -> None:
    task = SpatialDownscaler(
        input_datasets=["input"],
        output_datasets=["output"],
        input_offsets=[0],
        output_offsets=[0],
    )
    with pytest.raises(ValueError, match="positive"):
        task.bind_data_frequency(datetime.timedelta(0))


def test_get_timestep_for_metadata_bound_and_unbound() -> None:
    task = SpatialDownscaler(
        input_datasets=["input"],
        output_datasets=["output"],
        input_offsets=[0],
        output_offsets=[0],
    )
    with pytest.raises(RuntimeError, match="bind_data_frequency"):
        task._get_timestep_for_metadata()

    task.bind_data_frequency(datetime.timedelta(hours=6))
    assert task._get_timestep_for_metadata() == "6h"


# ── Seam integration: bind_task_frequency + compute_relative_date_indices ────


def test_bind_task_frequency_binds_and_relative_indices_equal_integer_offsets() -> None:
    task = SpatialDownscaler(
        input_datasets=["input"],
        output_datasets=["output"],
        input_offsets=[-1, 0, 1, 2],
        output_offsets=[0, 1],
    )
    readers = {
        "input": SimpleNamespace(frequency=datetime.timedelta(hours=6)),
        "output": SimpleNamespace(frequency=datetime.timedelta(hours=6)),
    }

    bind_task_frequency(task, readers)
    relative_date_indices = compute_relative_date_indices(task, readers)

    expected = sorted(set(task._integer_input_offsets) | set(task._integer_output_offsets))
    assert relative_date_indices == {"input": expected, "output": expected}


def test_bind_task_frequency_differing_frequencies_raises() -> None:
    task = SpatialDownscaler(
        input_datasets=["input"],
        output_datasets=["output"],
        input_offsets=[0],
        output_offsets=[0],
    )
    readers = {
        "input": SimpleNamespace(frequency=datetime.timedelta(hours=6)),
        "output": SimpleNamespace(frequency=datetime.timedelta(hours=1)),
    }
    with pytest.raises(ValueError, match="one frequency"):
        bind_task_frequency(task, readers)


def test_bind_task_frequency_noop_for_task_without_bind_data_frequency() -> None:
    task = MagicMock(spec=[])  # no bind_data_frequency attribute
    readers = {
        "input": SimpleNamespace(frequency=datetime.timedelta(hours=6)),
        "output": SimpleNamespace(frequency=datetime.timedelta(hours=1)),
    }
    bind_task_frequency(task, readers)  # must not raise


# ── fill_metadata ────────────────────────────────────────────────────────────


def test_fill_metadata_records_integer_offsets_and_roles() -> None:
    task = SpatialDownscaler(
        input_datasets=["input", "output"],
        output_datasets=["output"],
        input_offsets=[-1, 0, 1, 2],
        output_offsets=[0, 1],
    )
    task.bind_data_frequency(datetime.timedelta(hours=6))

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
