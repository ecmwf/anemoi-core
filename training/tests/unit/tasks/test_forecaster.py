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
import torch
from omegaconf import DictConfig

from anemoi.models.data_indices.collection import IndexCollection
from anemoi.training.tasks import FlexibleForecaster
from anemoi.training.tasks import Forecaster
from anemoi.training.utils.masks import Boolean1DMask
from anemoi.training.utils.masks import NoOutputMask


def _make_minimal_index_collection(
    name_to_index: dict[str, int],
    *,
    forcing: list[str] | None = None,
    diagnostic: list[str] | None = None,
    target: list[str] | None = None,
) -> IndexCollection:
    cfg = DictConfig(
        {
            "forcing": forcing or [],
            "diagnostic": diagnostic or [],
            "target": target or [],
        },
    )
    return IndexCollection(cfg, name_to_index)


_NAME_TO_INDEX: dict[str, int] = {"A": 0, "B": 1}


def _data_indices_single() -> dict[str, IndexCollection]:
    """Minimal data_indices for a single dataset named 'data'."""
    return {"data": _make_minimal_index_collection(_NAME_TO_INDEX)}


# ── Forecaster: offsets and steps ─────────────────────────────────────────────


def test_forecaster_single_input_offset() -> None:
    """multistep_input=1 produces a single input offset at t=0."""
    task = Forecaster(multistep_input=1, multistep_output=1, timestep="6h")
    assert task.offsets.input == [datetime.timedelta(0)]


def test_forecaster_multi_input_offsets_are_sorted() -> None:
    """multistep_input=2 produces sorted offsets [-6h, 0h]."""
    task = Forecaster(multistep_input=2, multistep_output=1, timestep="6h")
    assert task.offsets.input == [datetime.timedelta(hours=-6), datetime.timedelta(0)]


def test_forecaster_single_output_offset() -> None:
    """multistep_output=1 produces one output offset at +1 timestep."""
    task = Forecaster(multistep_input=1, multistep_output=1, timestep="6h")
    assert task.offsets.output == [datetime.timedelta(hours=6)]


def test_forecaster_multi_output_offsets() -> None:
    """multistep_output=2 produces offsets [+6h, +12h]."""
    task = Forecaster(multistep_input=1, multistep_output=2, timestep="6h")
    assert task.offsets.output == [datetime.timedelta(hours=6), datetime.timedelta(hours=12)]


def test_forecaster_steps_is_single_element() -> None:
    """Default rollout start=1 produces steps=({"rollout_step": 0},)."""
    task = Forecaster(multistep_input=1, multistep_output=1, timestep="6h", rollout={"start": 1})
    assert list(task.steps("training")) == [{"rollout_step": 0}]
    assert list(task.steps("validation")) == [{"rollout_step": 0}]
    assert list(task.steps("testing")) == [{"rollout_step": 0}]


def test_forecaster_steps_reflect_rollout_start() -> None:
    """Rollout start=2 produces two steps at construction time."""
    task = Forecaster(multistep_input=1, multistep_output=1, timestep="6h", rollout={"start": 2})
    assert list(task.steps("training")) == [{"rollout_step": 0}, {"rollout_step": 1}]
    assert list(task.steps("validation")) == [{"rollout_step": 0}]
    assert list(task.steps("testing")) == [{"rollout_step": 0}, {"rollout_step": 1}]


def test_forecaster_steps_reflect_validation_rollout() -> None:
    """Rollout with validation_rollout=3 produces three steps for validation only."""
    task = Forecaster(multistep_input=1, multistep_output=1, timestep="6h", validation_rollout=3)
    assert list(task.steps("training")) == [{"rollout_step": 0}]
    assert list(task.steps("validation")) == [{"rollout_step": 0}, {"rollout_step": 1}, {"rollout_step": 2}]
    assert list(task.steps("testing")) == [{"rollout_step": 0}]


def test_forecaster_metric_name_encodes_rollout_step() -> None:
    """get_metric_name returns a string containing the rollout step index."""
    task = Forecaster(multistep_input=1, multistep_output=1, timestep="6h")
    assert task.get_metric_name(rollout_step=0) == "_rstep0"
    assert task.get_metric_name(rollout_step=3) == "_rstep3"


# ── Forecaster: rollout curriculum ────────────────────────────────────────────


def test_forecaster_rollout_increases_on_epoch_end() -> None:
    """on_train_epoch_end increments rollout.step up to maximum."""
    task = Forecaster(
        multistep_input=1,
        multistep_output=1,
        timestep="6h",
        rollout={"start": 1, "epoch_increment": 1, "maximum": 3},
    )
    assert task.rollout.step == 1
    task.on_train_epoch_end(0)
    assert task.rollout.step == 2
    task.on_train_epoch_end(1)
    assert task.rollout.step == 3


def test_forecaster_rollout_does_not_exceed_maximum() -> None:
    """rollout.step is capped at maximum even when on_train_epoch_end is called repeatedly."""
    task = Forecaster(
        multistep_input=1,
        multistep_output=1,
        timestep="6h",
        rollout={"start": 1, "epoch_increment": 1, "maximum": 2},
    )
    for epoch in range(10):
        task.on_train_epoch_end(epoch)
    assert task.rollout.step == 2


def test_forecaster_rollout_no_increment_when_zero() -> None:
    """epoch_increment=0 means rollout.step stays at start permanently."""
    task = Forecaster(
        multistep_input=1,
        multistep_output=1,
        timestep="6h",
        rollout={"start": 1, "epoch_increment": 0, "maximum": 5},
    )
    for epoch in range(10):
        task.on_train_epoch_end(epoch)
    assert task.rollout.step == 1


# ── Forecaster: batch slicing ─────────────────────────────────────────────────


def test_forecaster_get_inputs_returns_correct_number_of_time_steps() -> None:
    """get_inputs extracts multistep_input time steps from the batch."""
    task = Forecaster(multistep_input=2, multistep_output=1, timestep="6h")
    data_indices = _data_indices_single()
    b, e, g, v = 2, 1, 4, len(_NAME_TO_INDEX)
    # offsets = [-6h, 0h, +6h] → 3 time steps in batch
    batch = {"data": torch.randn(b, 3, e, g, v)}
    x = task.get_inputs(batch, data_indices)
    assert x["data"].shape[1] == 2  # multistep_input=2


def test_forecaster_get_targets_returns_correct_number_of_time_steps() -> None:
    """get_targets extracts multistep_output time steps from the batch."""
    task = Forecaster(multistep_input=2, multistep_output=1, timestep="6h")
    b, e, g, v = 2, 1, 4, len(_NAME_TO_INDEX)
    batch = {"data": torch.randn(b, 3, e, g, v)}
    y = task.get_targets(batch)
    assert y["data"].shape[1] == 1  # multistep_output=1


def test_forecaster_get_inputs_and_targets_are_disjoint_in_time() -> None:
    """Input and target time indices do not overlap for a single-step forecaster."""
    task = Forecaster(multistep_input=1, multistep_output=1, timestep="6h")
    input_indices = task.get_batch_input_indices()
    output_indices = task.get_batch_output_indices(rollout_step=0)
    assert set(input_indices).isdisjoint(set(output_indices))


# ── Forecaster: _advance_dataset_input ────────────────────────────────────────


@pytest.mark.parametrize(
    ("n_step_input", "n_step_output", "expected"),
    [
        (2, 3, [4.0, 5.0]),
        (2, 2, [3.0, 4.0]),
        (3, 2, [3.0, 4.0, 5.0]),
    ],
)
def test_rollout_advance_input_keeps_latest_steps(
    n_step_input: int,
    n_step_output: int,
    expected: list[float],
) -> None:
    """_advance_dataset_input slides the window and fills with model predictions."""
    data_indices = _make_minimal_index_collection(_NAME_TO_INDEX)
    task = Forecaster(multistep_input=n_step_input, multistep_output=n_step_output, timestep="6h")

    b, e, g, v = 1, 1, 2, len(_NAME_TO_INDEX)
    x = torch.zeros((b, n_step_input, e, g, v), dtype=torch.float32)
    for step in range(n_step_input):
        x[:, step] = float(step + 1)

    y_pred = torch.stack(
        [
            torch.full((b, e, g, v), float(n_step_input + step), dtype=torch.float32)
            for step in range(1, n_step_output + 1)
        ],
        dim=1,
    )
    batch = torch.zeros((b, n_step_input + n_step_output, e, g, v), dtype=torch.float32)

    updated = task._advance_dataset_input(
        x,
        y_pred,
        batch,
        rollout_step=0,
        output_mask=NoOutputMask(),
        data_indices=data_indices,
    )
    kept_steps = updated[0, :, 0, 0, 0].tolist()
    assert kept_steps == expected, (
        f"Next input steps (n_step_input={n_step_input}, n_step_output={n_step_output}) "
        f"should be {expected}, got {kept_steps}."
    )
    for idx, value in enumerate(expected):
        assert torch.all(updated[:, idx] == value)


def test_rollout_advance_input_reapplies_boundary_truth_and_refreshes_forcing() -> None:
    """Boundary-masked prognostics are reset from truth before the next rollout step."""
    name_to_index = {"prog": 0, "force": 1}
    data_indices = _make_minimal_index_collection(name_to_index, forcing=["force"])
    output_mask = Boolean1DMask({"cutout_mask": torch.tensor([True, False])}, "cutout_mask")
    task = Forecaster(multistep_input=2, multistep_output=1, timestep="6h")

    # tensor dims: (batch, time, ens, grid, variable)
    x = torch.zeros((1, 2, 1, 2, 2), dtype=torch.float32)
    y_pred = torch.tensor([[[[[10.0], [20.0]]]]], dtype=torch.float32)
    batch = torch.zeros((1, 3, 1, 2, 2), dtype=torch.float32)
    batch[:, 2, 0, :, 0] = torch.tensor([100.0, 200.0])
    batch[:, 2, 0, :, 1] = torch.tensor([1000.0, 2000.0])

    updated = task._advance_dataset_input(
        x,
        y_pred,
        batch,
        rollout_step=0,
        data_indices=data_indices,
        output_mask=output_mask,
        grid_shard_slice=slice(None),
    )

    # prognostic variable, 1st grid point (cutout_mask=True) should be from y_pred,
    # 2nd grid point (cutout_mask=False) should be from batch
    torch.testing.assert_close(updated[0, -1, 0, :, 0], torch.tensor([10.0, 200.0]))
    # forcing variable should be refreshed from batch for both grid points
    torch.testing.assert_close(updated[0, -1, 0, :, 1], torch.tensor([1000.0, 2000.0]))


# ── BaseForecaster: get_offsets / get_output_offsets ─────────────────────────


def test_get_offsets_training_uses_rollout_maximum() -> None:
    """get_offsets in training mode covers offsets up to rollout.maximum steps."""
    task = Forecaster(
        multistep_input=1,
        multistep_output=1,
        timestep="6h",
        rollout={"start": 1, "maximum": 2},
    )
    # rollout.maximum=2 ; input [0h], output step 0 [6h], output step 1 [12h]
    offsets = task.get_offsets("training")
    assert datetime.timedelta(0) in offsets
    assert datetime.timedelta(hours=6) in offsets
    assert datetime.timedelta(hours=12) in offsets


def test_get_offsets_validation_uses_validation_rollout() -> None:
    """get_offsets in validation mode uses validation_rollout, not training rollout."""
    task = Forecaster(
        multistep_input=1,
        multistep_output=1,
        timestep="6h",
        rollout={"start": 1, "maximum": 3},
        validation_rollout=2,
    )
    training_offsets = task.get_offsets("training")
    validation_offsets = task.get_offsets("validation")
    assert len(validation_offsets) < len(training_offsets)


def test_get_output_offsets_no_shift_at_step_zero() -> None:
    """get_output_offsets at rollout_step=0 returns the base output offsets unchanged."""
    task = Forecaster(multistep_input=1, multistep_output=1, timestep="6h")
    assert task.get_output_offsets(rollout_step=0) == [datetime.timedelta(hours=6)]


def test_get_output_offsets_shifts_by_step_shift() -> None:
    """get_output_offsets at rollout_step=N shifts each offset by N * step_shift."""
    task = Forecaster(multistep_input=1, multistep_output=1, timestep="6h")
    # step_shift=6h, rollout_step=2 ; shift=12h ; [6h + 12h] = [18h]
    assert task.get_output_offsets(rollout_step=2) == [datetime.timedelta(hours=18)]


def test_get_output_offsets_multi_output_with_rollout_shift() -> None:
    """Multi-output offsets are all shifted uniformly across rollout steps."""
    task = Forecaster(multistep_input=1, multistep_output=2, timestep="6h")
    # base output offsets: [6h, 12h], step_shift=12h, rollout_step=1
    shifted = task.get_output_offsets(rollout_step=1)
    assert shifted == [datetime.timedelta(hours=18), datetime.timedelta(hours=24)]


# ── FlexibleForecaster ────────────────────────────────────────────────────────


def test_flexible_forecaster_input_offsets_from_strings() -> None:
    """Input offsets are correctly parsed from duration strings."""
    # [-6H, 0H] + 12H = [6H, 12H] ; contained in output_offsets  so valid shift exists
    task = FlexibleForecaster(input_offsets=["-6H", "0H"], output_offsets=["6H", "12H"])
    assert task.offsets.input == [datetime.timedelta(hours=-6), datetime.timedelta(0)]


def test_flexible_forecaster_output_offsets_from_strings() -> None:
    """Output offsets are correctly parsed from duration strings."""
    task = FlexibleForecaster(input_offsets=["0H"], output_offsets=["6H", "12H"])
    assert task.offsets.output == [datetime.timedelta(hours=6), datetime.timedelta(hours=12)]


def test_flexible_forecaster_step_shift_defaults_to_max_valid() -> None:
    """Without an explicit step_shift, the largest valid shift is inferred automatically."""
    # valid shifts for input=[0H], output=[6H,12H]; S=6H and S=12H are valid so max=12H
    task = FlexibleForecaster(input_offsets=["0H"], output_offsets=["6H", "12H"])
    assert task.offsets.step_shift == datetime.timedelta(hours=12)


def test_flexible_forecaster_step_shift_explicit() -> None:
    """An explicit step_shift is accepted when it satisfies shifted input offsets in union of input and output."""
    # [-6H, 0H] + 6H = [0H, 6H]: 0H in input (preserved), 6H in output (predicted)  so  valid
    task = FlexibleForecaster(input_offsets=["-6H", "0H"], output_offsets=["6H"], step_shift="6H")
    assert task.offsets.step_shift == datetime.timedelta(hours=6)


def test_flexible_forecaster_non_uniform_output_offsets() -> None:
    """Non-uniform output offsets are stored correctly in sorted order."""
    task = FlexibleForecaster(input_offsets=["0H"], output_offsets=["18H", "6H", "12H"])
    assert task.offsets.output == [
        datetime.timedelta(hours=6),
        datetime.timedelta(hours=12),
        datetime.timedelta(hours=18),
    ]


def test_flexible_forecaster_timestep_for_metadata_is_none() -> None:
    """_get_timestep_for_metadata returns None (no single regular timestep)."""
    task = FlexibleForecaster(input_offsets=["0H"], output_offsets=["6H"])
    assert task._get_timestep_for_metadata() is None


def test_flexible_forecaster_rollout_curriculum_is_inherited() -> None:
    """FlexibleForecaster inherits rollout curriculum from BaseForecaster."""
    task = FlexibleForecaster(
        input_offsets=["0H"],
        output_offsets=["6H"],
        rollout={"start": 1, "epoch_increment": 1, "maximum": 3},
    )
    assert task.rollout.step == 1
    task.on_train_epoch_end(0)
    assert task.rollout.step == 2
    task.on_train_epoch_end(1)
    assert task.rollout.step == 3


def test_flexible_forecaster_rollout_does_not_exceed_maximum() -> None:
    """rollout.step is capped at maximum even when called repeatedly."""
    task = FlexibleForecaster(
        input_offsets=["0H"],
        output_offsets=["6H"],
        rollout={"start": 1, "epoch_increment": 1, "maximum": 2},
    )
    for epoch in range(10):
        task.on_train_epoch_end(epoch)
    assert task.rollout.step == 2


def test_flexible_forecaster_steps_match_rollout_start() -> None:
    """steps() returns the correct number of rollout steps for each mode."""
    task = FlexibleForecaster(
        input_offsets=["0H"],
        output_offsets=["6H"],
        rollout={"start": 2},
    )
    assert list(task.steps("training")) == [{"rollout_step": 0}, {"rollout_step": 1}]
    assert list(task.steps("validation")) == [{"rollout_step": 0}]


def test_flexible_forecaster_get_offsets_uses_step_shift() -> None:
    """get_offsets correctly incorporates the inferred step_shift over multiple rollout steps."""
    task = FlexibleForecaster(
        input_offsets=["0H"],
        output_offsets=["6H"],
        rollout={"start": 1, "maximum": 2},
    )
    # step_shift=6h; rollout.maximum=2 → offsets: [0h, 6h, 12h]
    offsets = task.get_offsets("training")
    assert datetime.timedelta(0) in offsets
    assert datetime.timedelta(hours=6) in offsets
    assert datetime.timedelta(hours=12) in offsets


def test_flexible_forecaster_multiple_valid_shifts_picks_max() -> None:
    """When multiple valid shifts exist, the largest one is chosen as default."""
    # I=[-6H, 0H], O=[6H, 12H], I union O={-6H,0H,6H,12H}:
    #   S=6H:  I_shifted = [0H, 6H], valid
    #   S=12H: I_shifted = [6H, 12H], valid
    #   S=18H: I_shifted = [12H, 18H], invalid
    # max valid = 12H
    task = FlexibleForecaster(input_offsets=["-6H", "0H"], output_offsets=["6H", "12H"])
    assert task.offsets.step_shift == datetime.timedelta(hours=12)


def test_flexible_forecaster_invalid_step_shift_raises() -> None:
    """An explicit step_shift that does not satisfy shifted input in union of input and output raises ValueError."""
    with pytest.raises(ValueError, match="not a valid autoregressive shift"):
        FlexibleForecaster(input_offsets=["0H"], output_offsets=["6H"], step_shift="12H")


def test_flexible_forecaster_step_shift_raises_when_gap_not_covered() -> None:
    """step_shift raises when a shifted input falls outside union input and output entirely."""
    # I=[0H, 3H], O=[6H], S=6H: I_shifted = [6H, 9H], invalid
    with pytest.raises(ValueError, match="not a valid autoregressive shift"):
        FlexibleForecaster(input_offsets=["0H", "3H"], output_offsets=["6H"], step_shift="6H")


def test_flexible_forecaster_no_valid_shift_raises() -> None:
    """When no valid shift exists, construction raises ValueError."""
    # I=[0H, 6H], O=[4H]: only candidate S=4H; I_shifted = [4H, 10H], invalid , so no valid shift
    with pytest.raises(ValueError, match="No valid autoregressive step_shift"):
        FlexibleForecaster(input_offsets=["0H", "6H"], output_offsets=["4H"])


# ── FlexibleForecaster: preserve-slot advance_input ───────────────────────────


def test_flexible_forecaster_advance_input_preserves_and_predicts() -> None:
    """_advance_dataset_input copies preserved slots and fills predicted slots.

    Config: I=[-6H, 0H], O=[6H], S=6H (inferred).
    After one advance:
      new slot 0 (offset 0H relative to T') ; old slot 1 (offset 0H relative to T)  [preserved]
      new slot 1 (offset 6H relative to T') ; y_pred slot 0 (offset 6H)             [predicted]
    """
    data_indices = _make_minimal_index_collection(_NAME_TO_INDEX)
    task = FlexibleForecaster(input_offsets=["-6H", "0H"], output_offsets=["6H"])
    assert task.offsets.step_shift == datetime.timedelta(hours=6)

    b, e, g, v = 1, 1, 2, len(_NAME_TO_INDEX)
    # x[:, 0] = 1.0  (-6h slot), x[:, 1] = 2.0  (0h slot)
    x = torch.zeros((b, 2, e, g, v), dtype=torch.float32)
    x[:, 0] = 1.0
    x[:, 1] = 2.0

    y_pred = torch.full((b, 1, e, g, v), 3.0, dtype=torch.float32)

    # batch covers all offsets [-6h, 0h, 6h] ; 3 time steps
    batch = torch.zeros((b, 3, e, g, v), dtype=torch.float32)

    updated = task._advance_dataset_input(
        x,
        y_pred,
        batch,
        rollout_step=0,
        output_mask=NoOutputMask(),
        data_indices=data_indices,
    )

    # Slot 0: 0H offset ; preserved from old slot 1 (value 2.0)
    assert torch.all(updated[:, 0] == 2.0), f"preserved slot should be 2.0, got {updated[:, 0]}"
    # Slot 1: 6H offset ; predicted (value 3.0)
    assert torch.all(updated[:, 1] == 3.0), f"predicted slot should be 3.0, got {updated[:, 1]}"
