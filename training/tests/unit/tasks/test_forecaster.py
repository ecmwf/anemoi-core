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
from anemoi.training.tasks import Forecaster
from anemoi.training.tasks import OffsetForecaster
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
    assert task._input_offsets == [datetime.timedelta(0)]


def test_forecaster_multi_input_offsets_are_sorted() -> None:
    """multistep_input=2 produces sorted offsets [-6h, 0h]."""
    task = Forecaster(multistep_input=2, multistep_output=1, timestep="6h")
    assert task._input_offsets == [datetime.timedelta(hours=-6), datetime.timedelta(0)]


def test_forecaster_single_output_offset() -> None:
    """multistep_output=1 produces one output offset at +1 timestep."""
    task = Forecaster(multistep_input=1, multistep_output=1, timestep="6h")
    assert task._output_offsets == [datetime.timedelta(hours=6)]


def test_forecaster_multi_output_offsets() -> None:
    """multistep_output=2 produces offsets [+6h, +12h]."""
    task = Forecaster(multistep_input=1, multistep_output=2, timestep="6h")
    assert task._output_offsets == [datetime.timedelta(hours=6), datetime.timedelta(hours=12)]


def test_forecaster_steps_is_single_element() -> None:
    """Default rollout start=1 produces steps=({"rollout_step": 0},)."""
    task = Forecaster(multistep_input=1, multistep_output=1, timestep="6h", rollout={"start": 1})
    assert list(task.steps("training")) == [{"rollout_step": 0}]
    assert list(task.steps("validation")) == [{"rollout_step": 0}]
    assert list(task.steps("testing")) == [{"rollout_step": 0}]


def test_forecaster_steps_reflect_rollout_start() -> None:
    """Rollout start=2 produces two steps at construction time."""
    task = Forecaster(multistep_input=1, multistep_output=1, timestep="6h", rollout={"start": 2, "maximum": 2})
    assert list(task.steps("training")) == [{"rollout_step": 0}, {"rollout_step": 1}]
    assert list(task.steps("validation")) == [{"rollout_step": 0}, {"rollout_step": 1}]
    assert list(task.steps("testing")) == [{"rollout_step": 0}, {"rollout_step": 1}]


def test_forecaster_validation_rollout_none_follows_training_rollout() -> None:
    """Unset validation_rollout follows the current training rollout."""
    task = Forecaster(
        multistep_input=1,
        multistep_output=1,
        timestep="6h",
        rollout={"start": 1, "epoch_increment": 1, "maximum": 3},
    )

    assert list(task.steps("validation")) == [{"rollout_step": 0}]
    assert task.get_offsets(mode="validation") == [datetime.timedelta(0), datetime.timedelta(hours=6)]

    task.on_train_epoch_end(0)

    assert list(task.steps("validation")) == [{"rollout_step": 0}, {"rollout_step": 1}]
    assert task.get_offsets(mode="validation") == [
        datetime.timedelta(0),
        datetime.timedelta(hours=6),
        datetime.timedelta(hours=12),
    ]


def test_forecaster_steps_reflect_validation_rollout() -> None:
    """Rollout with validation_rollout=3 produces three steps for validation only."""
    task = Forecaster(multistep_input=1, multistep_output=1, timestep="6h", validation_rollout=3)
    assert list(task.steps("training")) == [{"rollout_step": 0}]
    assert list(task.steps("validation")) == [{"rollout_step": 0}, {"rollout_step": 1}, {"rollout_step": 2}]
    assert list(task.steps("testing")) == [{"rollout_step": 0}]


def test_forecaster_training_offsets_reflect_current_rollout() -> None:
    """Training offsets grow with the current rollout instead of always using the configured maximum."""
    task = Forecaster(
        multistep_input=1,
        multistep_output=1,
        timestep="6h",
        rollout={"start": 1, "epoch_increment": 1, "maximum": 3},
    )

    assert task.get_offsets(mode="training") == [datetime.timedelta(0), datetime.timedelta(hours=6)]
    task.on_train_epoch_end(0)
    assert task.get_offsets(mode="training") == [
        datetime.timedelta(0),
        datetime.timedelta(hours=6),
        datetime.timedelta(hours=12),
    ]


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
        data_frequency="6h",
        rollout={"start": 1, "epoch_increment": 1, "maximum": 3},
    )
    assert task.rollout.step == 1
    task.on_train_epoch_end(0)
    assert task.rollout.step == 2
    task.on_train_epoch_end(1)
    assert task.rollout.step == 3


def test_forecaster_rollout_increases_after_configured_number_of_epochs() -> None:
    """epoch_increment counts completed epochs before increasing the rollout."""
    task = Forecaster(
        multistep_input=1,
        multistep_output=1,
        timestep="6h",
        rollout={"start": 1, "epoch_increment": 2, "maximum": 3},
    )

    task.on_train_epoch_end(0)
    assert task.rollout.step == 1
    task.on_train_epoch_end(1)
    assert task.rollout.step == 2
    task.on_train_epoch_end(2)
    assert task.rollout.step == 2
    task.on_train_epoch_end(3)
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


# ── RolloutConfig: state_dict / load_state_dict ───────────────────────────────


def test_rollout_config_state_dict_captures_current_step() -> None:
    """state_dict returns the live step and last_increased_epoch, not the initial start value."""
    task = Forecaster(
        multistep_input=1,
        multistep_output=1,
        timestep="6h",
        rollout={"start": 1, "epoch_increment": 1, "maximum": 5},
    )
    task.on_train_epoch_end(0)
    task.on_train_epoch_end(1)
    assert task.rollout.state_dict() == {"step": 3, "last_increased_epoch": 1}


def test_rollout_config_load_state_dict_restores_step() -> None:
    """load_state_dict overwrites step and last_increased_epoch regardless of current value."""
    from anemoi.training.tasks.forecaster import RolloutConfig

    cfg = RolloutConfig(start=1, epoch_increment=1, maximum=10)
    cfg.load_state_dict({"step": 7, "last_increased_epoch": 5})
    assert cfg.step == 7
    assert cfg._last_increased_epoch == 5


def test_rollout_config_increase_is_idempotent_per_epoch() -> None:
    """on_train_epoch_end called twice with the same epoch does not double-increment."""
    task = Forecaster(
        multistep_input=1,
        multistep_output=1,
        timestep="6h",
        rollout={"start": 1, "epoch_increment": 1, "maximum": 5},
    )
    task.on_train_epoch_end(0)
    task.on_train_epoch_end(0)  # second call with same epoch — must be a no-op
    assert task.rollout.step == 2


# ── Forecaster: training_runtime_state_dict / load_training_runtime_state_dict ─────────────────────


def test_forecaster_training_runtime_state_dict_round_trip() -> None:
    """Saving and loading extra state restores rollout.step exactly."""
    task = Forecaster(
        multistep_input=1,
        multistep_output=1,
        timestep="6h",
        rollout={"start": 1, "epoch_increment": 1, "maximum": 10},
    )
    task.on_train_epoch_end(0)
    task.on_train_epoch_end(1)
    assert task.rollout.step == 3

    saved = task.training_runtime_state_dict()

    fresh = Forecaster(
        multistep_input=1,
        multistep_output=1,
        timestep="6h",
        rollout={"start": 1, "epoch_increment": 1, "maximum": 10},
    )
    assert fresh.rollout.step == 1
    fresh.load_training_runtime_state_dict(saved)
    assert fresh.rollout.step == 3


def test_forecaster_load_training_runtime_state_dict_missing_key_is_noop() -> None:
    """load_training_runtime_state_dict with an empty dict leaves rollout.step unchanged."""
    task = Forecaster(multistep_input=1, multistep_output=1, timestep="6h", rollout={"start": 2})
    task.load_training_runtime_state_dict({})
    assert task.rollout.step == 2


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


def test_forecaster_get_targets_raises_when_batch_is_short_of_time_steps() -> None:
    """A batch sized for an earlier rollout fails before producing an empty slice."""
    task = Forecaster(
        multistep_input=2,
        multistep_output=1,
        timestep="6h",
        rollout={"start": 1, "epoch_increment": 1, "maximum": 2},
    )
    batch = {"data": torch.randn(2, 3, 1, 4, len(_NAME_TO_INDEX))}
    assert task.get_targets(batch, rollout_step=0)["data"].shape[1] == 1

    task.rollout.increase(current_epoch=0)

    with pytest.raises(ValueError, match="requires index 3") as exc_info:
        task.get_targets(batch, rollout_step=1)

    assert str(exc_info.value) == (
        "Batch for dataset 'data' contains 3 time steps, but requires index 3 (indices [3]). "
        "The dataloader's time window does not match the task rollout."
    )


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


# ── OffsetForecaster: equivalence with Forecaster on a regular grid ────────────

_TIMESTEP_HOURS = 6


def _offset_equivalent(
    multistep_input: int,
    multistep_output: int,
    timestep_hours: int = _TIMESTEP_HOURS,
) -> OffsetForecaster:
    """Build the ``OffsetForecaster`` equivalent to ``Forecaster(N, M, timestep)``.

    A regular forecaster reading ``N`` steps and predicting ``M`` steps on a grid of
    spacing ``timestep`` maps onto input offsets ``[-(N-1)T, ..., 0]``, output offsets
    ``[T, ..., MT]`` and a rollout shift of ``MT``.
    """
    input_offsets = [f"{-i * timestep_hours}h" for i in range(multistep_input)]
    output_offsets = [f"{(i + 1) * timestep_hours}h" for i in range(multistep_output)]
    rollout_shift = f"{multistep_output * timestep_hours}h"
    return OffsetForecaster(
        input_offsets=input_offsets,
        output_offsets=output_offsets,
        rollout_shift=rollout_shift,
    )


@pytest.mark.parametrize(
    ("n_step_input", "n_step_output", "expected"),
    [
        (1, 1, [2.0]),
        (2, 2, [3.0, 4.0]),
        (2, 3, [4.0, 5.0]),
        (3, 2, [3.0, 4.0, 5.0]),
        (3, 1, [2.0, 3.0, 4.0]),
        (1, 2, [3.0]),
    ],
)
def test_offset_forecaster_advance_matches_forecaster(
    n_step_input: int,
    n_step_output: int,
    expected: list[float],
) -> None:
    """OffsetForecaster._advance_dataset_input matches the legacy Forecaster on a regular grid."""
    data_indices = _make_minimal_index_collection(_NAME_TO_INDEX)
    legacy = Forecaster(multistep_input=n_step_input, multistep_output=n_step_output, timestep="6h")
    offset = _offset_equivalent(n_step_input, n_step_output)

    b, e, g, v = 1, 1, 2, len(_NAME_TO_INDEX)
    x = torch.zeros((b, n_step_input, e, g, v), dtype=torch.float32)
    for step in range(n_step_input):
        x[:, step] = float(step + 1)

    y_pred = torch.stack(
        [
            torch.full((b, e, g, v), float(n_step_input + step + 1), dtype=torch.float32)
            for step in range(n_step_output)
        ],
        dim=1,
    )
    batch = torch.zeros((b, n_step_input + n_step_output, e, g, v), dtype=torch.float32)

    out_legacy = legacy._advance_dataset_input(
        x.clone(),
        y_pred,
        batch,
        rollout_step=0,
        output_mask=NoOutputMask(),
        data_indices=data_indices,
    )
    out_offset = offset._advance_dataset_input(
        x.clone(),
        y_pred,
        batch,
        rollout_step=0,
        output_mask=NoOutputMask(),
        data_indices=data_indices,
    )

    torch.testing.assert_close(out_offset, out_legacy)
    # Anchor against the known-correct legacy behaviour so a shared bug cannot hide.
    assert out_offset[0, :, 0, 0, 0].tolist() == expected


def test_offset_forecaster_advance_matches_forecaster_with_boundary_and_forcing() -> None:
    """Equivalence also holds on the boundary-mask and forcing-refresh code paths."""
    name_to_index = {"prog": 0, "force": 1}
    data_indices = _make_minimal_index_collection(name_to_index, forcing=["force"])
    legacy = Forecaster(multistep_input=2, multistep_output=1, timestep="6h")
    offset = _offset_equivalent(2, 1)

    def _make_inputs() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # tensor dims: (batch, time, ens, grid, variable)
        x = torch.zeros((1, 2, 1, 2, 2), dtype=torch.float32)
        y_pred = torch.tensor([[[[[10.0], [20.0]]]]], dtype=torch.float32)
        batch = torch.zeros((1, 3, 1, 2, 2), dtype=torch.float32)
        batch[:, 2, 0, :, 0] = torch.tensor([100.0, 200.0])
        batch[:, 2, 0, :, 1] = torch.tensor([1000.0, 2000.0])
        return x, y_pred, batch

    x, y_pred, batch = _make_inputs()
    out_legacy = legacy._advance_dataset_input(
        x,
        y_pred,
        batch,
        rollout_step=0,
        data_indices=data_indices,
        output_mask=Boolean1DMask({"cutout_mask": torch.tensor([True, False])}, "cutout_mask"),
        grid_shard_slice=slice(None),
    )

    x, y_pred, batch = _make_inputs()
    out_offset = offset._advance_dataset_input(
        x,
        y_pred,
        batch,
        rollout_step=0,
        data_indices=data_indices,
        output_mask=Boolean1DMask({"cutout_mask": torch.tensor([True, False])}, "cutout_mask"),
        grid_shard_slice=slice(None),
    )

    torch.testing.assert_close(out_offset, out_legacy)


# ── OffsetForecaster: advance on irregular grids (no Forecaster equivalent) ────


@pytest.mark.parametrize(
    ("input_offsets", "output_offsets", "expected"),
    [
        # Mixed advance: input slot 0 is reused from the input window (inin),
        # slot 1 is filled from the first prediction (outin). Shift inferred as 6h.
        (["-6h", "0h"], ["6h", "9h"], [2.0, 10.0]),
        # Two reused input slots plus one prediction. Shift inferred as 6h.
        (["-12h", "-6h", "0h"], ["6h", "9h"], [2.0, 3.0, 10.0]),
        # Both slots refreshed from non-adjacent predictions. Shift inferred as 10h.
        (["-6h", "0h"], ["4h", "6h", "10h"], [10.0, 30.0]),
    ],
)
def test_offset_forecaster_advance_irregular_offsets(
    input_offsets: list[str],
    output_offsets: list[str],
    expected: list[float],
) -> None:
    """_advance_dataset_input handles irregular grids that no legacy Forecaster can represent."""
    data_indices = _make_minimal_index_collection(_NAME_TO_INDEX)
    task = OffsetForecaster(input_offsets=input_offsets, output_offsets=output_offsets)

    n_input = len(input_offsets)
    n_output = len(output_offsets)
    b, e, g, v = 1, 1, 2, len(_NAME_TO_INDEX)
    x = torch.zeros((b, n_input, e, g, v), dtype=torch.float32)
    for step in range(n_input):
        x[:, step] = float(step + 1)

    y_pred = torch.stack(
        [torch.full((b, e, g, v), float(10 * (step + 1)), dtype=torch.float32) for step in range(n_output)],
        dim=1,
    )
    batch = torch.zeros((b, n_input + n_output, e, g, v), dtype=torch.float32)

    updated = task._advance_dataset_input(
        x,
        y_pred,
        batch,
        rollout_step=0,
        output_mask=NoOutputMask(),
        data_indices=data_indices,
    )
    assert updated[0, :, 0, 0, 0].tolist() == expected


# ── OffsetForecaster: _convert_and_validate ───────────────────────────────────


@pytest.mark.parametrize(
    ("input_offsets", "output_offsets", "rollout_shift", "match"),
    [
        # duplicate offsets are not well-formed
        (["0h", "0h"], ["6h"], "default", "input_offsets contains duplicate"),
        (["0h"], ["6h", "6h"], "default", "output_offsets contains duplicate"),
        # an output must come strictly after every input for a forecasting task
        (["0h", "6h"], ["6h", "12h"], "default", "strictly greater"),
        (["-6h", "0h"], ["-3h", "3h"], "default", "strictly greater"),
        # no valid shift exists
        (["-6h", "0h"], ["7h", "10h"], "default", "No valid autoregressive rollout shift"),
        # explicit shift that repeats an output across rollout steps is rejected
        (["-6h", "0h"], ["6h", "12h"], "6h", "is not a valid autoregressive"),
    ],
)
def test_offset_convert_and_validate_rejects_invalid(
    input_offsets: list[str],
    output_offsets: list[str],
    rollout_shift: str,
    match: str,
) -> None:
    """_convert_and_validate raises ValueError for ill-formed or inconsistent offsets."""
    with pytest.raises(ValueError, match=match):
        OffsetForecaster._convert_and_validate(input_offsets, output_offsets, rollout_shift)


@pytest.mark.parametrize(
    ("input_offsets", "output_offsets", "rollout_shift", "expected_hours"),
    [
        # single step: only valid shift equals the output horizon
        (["0h"], ["6h"], "default", 6),
        # regular grid: default infers the output horizon M*T
        (["-6h", "0h"], ["6h", "12h"], "default", 12),
        # same, but supplied explicitly
        (["-6h", "0h"], ["6h", "12h"], "12h", 12),
        # irregular grid with several valid shifts: default picks the largest
        (["0h"], ["6h", "10h"], "default", 10),
        # ...and a smaller valid shift is accepted when requested
        (["0h"], ["6h", "10h"], "6h", 6),
    ],
)
def test_offset_convert_and_validate_returns_expected_rollout_shift(
    input_offsets: list[str],
    output_offsets: list[str],
    rollout_shift: str,
    expected_hours: int,
) -> None:
    """_convert_and_validate returns the expected rollout shift for valid offsets."""
    _, _, shift = OffsetForecaster._convert_and_validate(input_offsets, output_offsets, rollout_shift)
    assert shift == datetime.timedelta(hours=expected_hours)


def _hours(*values: float) -> list[datetime.timedelta]:
    return [datetime.timedelta(hours=v) for v in values]


@pytest.mark.parametrize(
    ("input_offsets", "output_offsets", "expected_inputs", "expected_outputs"),
    [
        # strings are parsed and sorted ascending
        (["0h", "-6h"], ["12h", "6h"], _hours(-6, 0), _hours(6, 12)),
        # single offsets
        (["0h"], ["6h"], _hours(0), _hours(6)),
        # mixed units are normalised to timedeltas
        (["-360m", "0h"], ["720m", "6h"], _hours(-6, 0), _hours(6, 12)),
        # fractional-hour (sub-grid) offsets
        (["0h"], ["45m", "90m"], _hours(0), _hours(0.75, 1.5)),
    ],
)
def test_offset_convert_and_validate_returns_sorted_timedeltas(
    input_offsets: list[str],
    output_offsets: list[str],
    expected_inputs: list[datetime.timedelta],
    expected_outputs: list[datetime.timedelta],
) -> None:
    """_convert_and_validate parses offset strings into sorted timedeltas."""
    converted_inputs, converted_outputs, _ = OffsetForecaster._convert_and_validate(
        input_offsets,
        output_offsets,
        "default",
    )
    assert converted_inputs == expected_inputs
    assert converted_outputs == expected_outputs
    assert all(isinstance(offset, datetime.timedelta) for offset in converted_inputs + converted_outputs)
