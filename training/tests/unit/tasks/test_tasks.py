# (C) Copyright 2026 Anemoi contributors.
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
from anemoi.training.tasks import Autoencoder
from anemoi.training.tasks import Forecaster
from anemoi.training.tasks import TemporalDownscaler


def _make_minimal_index_collection(name_to_index: dict[str, int]) -> IndexCollection:
    cfg = DictConfig({"forcing": [], "diagnostic": [], "target": []})
    return IndexCollection(cfg, name_to_index)


_NAME_TO_INDEX: dict[str, int] = {"A": 0, "B": 1}


def _data_indices_single() -> dict[str, IndexCollection]:
    """Minimal data_indices for a single dataset named 'data'."""
    return {"data": _make_minimal_index_collection(_NAME_TO_INDEX)}


# ── Forecaster: offsets and steps ─────────────────────────────────────────────


def test_forecaster_single_input_offset() -> None:
    """multistep_input=1 produces a single input offset at t=0."""
    task = Forecaster(multistep_input=1, multistep_output=1, timestep="6h")
    assert task.input_offsets == [datetime.timedelta(0)]


def test_forecaster_multi_input_offsets_are_sorted() -> None:
    """multistep_input=2 produces sorted offsets [-6h, 0h]."""
    task = Forecaster(multistep_input=2, multistep_output=1, timestep="6h")
    assert task.input_offsets == [datetime.timedelta(hours=-6), datetime.timedelta(0)]


def test_forecaster_single_output_offset() -> None:
    """multistep_output=1 produces one output offset at +1 timestep."""
    task = Forecaster(multistep_input=1, multistep_output=1, timestep="6h")
    assert task.output_offsets == [datetime.timedelta(hours=6)]


def test_forecaster_multi_output_offsets() -> None:
    """multistep_output=2 produces offsets [+6h, +12h]."""
    task = Forecaster(multistep_input=1, multistep_output=2, timestep="6h")
    assert task.output_offsets == [datetime.timedelta(hours=6), datetime.timedelta(hours=12)]


def test_forecaster_steps_on_init_is_single_element() -> None:
    """Default rollout start=1 produces steps=({"rollout_step": 0},)."""
    task = Forecaster(multistep_input=1, multistep_output=1, timestep="6h", rollout={"start": 1})
    assert list(task.steps) == [{"rollout_step": 0}]


def test_forecaster_steps_on_init_reflect_rollout_start() -> None:
    """Rollout start=2 produces two steps at construction time."""
    task = Forecaster(multistep_input=1, multistep_output=1, timestep="6h", rollout={"start": 2})
    assert list(task.steps) == [{"rollout_step": 0}, {"rollout_step": 1}]
    assert task.num_steps == 2


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
    data_indices = _data_indices_single()
    b, e, g, v = 2, 1, 4, len(_NAME_TO_INDEX)
    batch = {"data": torch.randn(b, 3, e, g, v)}
    y = task.get_targets(batch, data_indices)
    assert y["data"].shape[1] == 1  # multistep_output=1


def test_forecaster_get_inputs_and_targets_are_disjoint_in_time() -> None:
    """Input and target time indices do not overlap for a single-step forecaster."""
    task = Forecaster(multistep_input=1, multistep_output=1, timestep="6h")
    input_indices = task.get_batch_input_indices()
    output_indices = task.get_batch_output_indices(rollout_step=0)
    assert set(input_indices).isdisjoint(set(output_indices))


# ── Forecaster: plot adapter ───────────────────────────────────────────────────


def test_forecaster_plot_adapter_output_times_equals_num_output_timesteps() -> None:
    """ForecasterPlotAdapter.output_times reflects num_output_timesteps."""
    task = Forecaster(multistep_input=1, multistep_output=1, timestep="6h")
    assert task._plot_adapter.output_times == task.num_output_timesteps == 1


def test_forecaster_plot_adapter_output_times_for_multi_output() -> None:
    """output_times is 2 when multistep_output=2."""
    task = Forecaster(multistep_input=1, multistep_output=2, timestep="6h")
    assert task._plot_adapter.output_times == 2


def test_forecaster_plot_adapter_get_init_step_returns_minus_one() -> None:
    """ForecasterPlotAdapter.get_init_step() returns -1 (last input time step)."""
    task = Forecaster(multistep_input=1, multistep_output=1, timestep="6h")
    assert task._plot_adapter.get_init_step() == -1


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
        data_indices=data_indices,
    )
    kept_steps = updated[0, :, 0, 0, 0].tolist()
    assert kept_steps == expected, (
        f"Next input steps (n_step_input={n_step_input}, n_step_output={n_step_output}) "
        f"should be {expected}, got {kept_steps}."
    )
    for idx, value in enumerate(expected):
        assert torch.all(updated[:, idx] == value)


# ── Autoencoder ────────────────────────────────────────────────────────────────


def test_autoencoder_input_and_output_offsets_are_both_zero() -> None:
    """Autoencoder operates on a single snapshot at t=0."""
    task = Autoencoder()
    assert task.input_offsets == [datetime.timedelta(0)]
    assert task.output_offsets == [datetime.timedelta(0)]
    assert task.offsets == [datetime.timedelta(0)]


def test_autoencoder_has_exactly_one_step_with_no_kwargs() -> None:
    """Autoencoder runs exactly one step and passes no step-specific kwargs."""
    task = Autoencoder()
    assert list(task.steps) == [{}]
    assert task.num_steps == 1


def test_autoencoder_advance_input_returns_input_unchanged() -> None:
    """advance_input for a single-step task is a no-op (returns first positional arg)."""
    task = Autoencoder()
    x = {"data": torch.randn(2, 1, 1, 4, 2)}
    result = task.advance_input(x, {}, {})
    assert result is x


def test_autoencoder_plot_adapter_output_times_is_one() -> None:
    """AutoencoderPlotAdapter reports output_times=1."""
    task = Autoencoder()
    assert task._plot_adapter.output_times == 1


# ── TemporalDownscaler ────────────────────────────────────────────────


def test_temporal_downscaler_interior_offsets_only() -> None:
    """No boundaries: only interior interpolation steps are produced."""
    task = TemporalDownscaler(
        input_timestep="6h",
        output_timestep="2h",
        output_left_boundary=False,
        output_right_boundary=False,
    )
    expected = [datetime.timedelta(hours=2), datetime.timedelta(hours=4)]
    assert task.output_offsets == expected


def test_temporal_downscaler_left_boundary_included() -> None:
    """output_left_boundary=True adds t=0 to the output offsets."""
    task = TemporalDownscaler(
        input_timestep="6h",
        output_timestep="2h",
        output_left_boundary=True,
        output_right_boundary=False,
    )
    expected = [datetime.timedelta(hours=0), datetime.timedelta(hours=2), datetime.timedelta(hours=4)]
    assert task.output_offsets == expected


def test_temporal_downscaler_right_boundary_included() -> None:
    """output_right_boundary=True adds t=input_timestep to the output offsets."""
    task = TemporalDownscaler(
        input_timestep="6h",
        output_timestep="2h",
        output_left_boundary=False,
        output_right_boundary=True,
    )
    expected = [datetime.timedelta(hours=2), datetime.timedelta(hours=4), datetime.timedelta(hours=6)]
    assert task.output_offsets == expected


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
    assert task.output_offsets == expected


def test_temporal_downscaler_num_output_timesteps_matches_offsets() -> None:
    """num_output_timesteps equals the length of output_offsets."""
    task = TemporalDownscaler(
        input_timestep="6h",
        output_timestep="2h",
        output_left_boundary=True,
        output_right_boundary=True,
    )
    assert task.num_output_timesteps == len(task.output_offsets) == 4


def test_temporal_downscaler_input_offsets_are_boundary_pair() -> None:
    """Input offsets are always [0h, input_timestep] regardless of output settings."""
    task = TemporalDownscaler(input_timestep="6h", output_timestep="2h")
    assert task.input_offsets == [datetime.timedelta(0), datetime.timedelta(hours=6)]


def test_temporal_downscaler_plot_adapter_output_times() -> None:
    """TemporalDownscalerPlotAdapter.output_times reflects the number of output offsets."""
    task = TemporalDownscaler(
        input_timestep="6h",
        output_timestep="2h",
        output_left_boundary=True,
        output_right_boundary=True,
    )
    assert task._plot_adapter.output_times == 4


def test_temporal_downscaler_plot_adapter_get_init_step_is_zero() -> None:
    """TemporalDownscalerPlotAdapter.get_init_step() returns 0 (first input step)."""
    task = TemporalDownscaler(
        input_timestep="6h",
        output_timestep="2h",
        output_left_boundary=True,
        output_right_boundary=True,
    )
    assert task._plot_adapter.get_init_step() == 0
