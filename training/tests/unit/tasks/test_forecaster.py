# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import datetime
import itertools

import pytest
import torch
from omegaconf import DictConfig

from anemoi.models.data import Batch
from anemoi.models.data import TabularSource
from anemoi.models.data import TensorLayout
from anemoi.models.data_indices.collection import IndexCollection
from anemoi.training.tasks import Forecaster
from anemoi.training.tasks import OffsetForecaster
from anemoi.training.utils.masks import Boolean1DMask
from anemoi.training.utils.masks import NoOutputMask
from tests.batch_builders import build_batch


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


def test_forecaster_validation_unrolls_at_least_training_rollout() -> None:
    """validation_rollout below the training rollout still unrolls all training steps."""
    task = Forecaster(
        multistep_input=1,
        multistep_output=1,
        timestep="6h",
        rollout={"start": 3, "maximum": 3},
        validation_rollout=2,
    )

    assert list(task.steps("validation")) == [{"rollout_step": 0}, {"rollout_step": 1}, {"rollout_step": 2}]
    assert task.get_offsets(mode="validation") == task.get_offsets(mode="training")


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
    layout = TensorLayout(batch=0, time=1, ensemble=2, grid=3, variables=4)
    batch = build_batch(
        data={"data": torch.randn(b, 3, e, g, v)},
        coordinates={"data": torch.rand(g, 2)},
        layouts={"data": layout},
        variables={"data": list(_NAME_TO_INDEX)},
    )
    x = task.get_inputs(batch, data_indices)
    assert x["data"].data.shape[1] == 2  # multistep_input=2


def test_forecaster_get_targets_returns_correct_number_of_time_steps() -> None:
    """get_targets extracts multistep_output time steps from the batch."""
    task = Forecaster(multistep_input=2, multistep_output=1, timestep="6h")
    data_indices = _data_indices_single()
    b, e, g, v = 2, 1, 4, len(_NAME_TO_INDEX)
    layout = TensorLayout(batch=0, time=1, ensemble=2, grid=3, variables=4)
    batch = build_batch(
        data={"data": torch.randn(b, 3, e, g, v)},
        coordinates={"data": torch.rand(g, 2)},
        layouts={"data": layout},
        variables={"data": list(_NAME_TO_INDEX)},
    )
    y, _template, _forcing = task.get_targets(batch, data_indices)
    assert y["data"].data.shape[1] == 1  # multistep_output=1


def test_forecaster_get_targets_template_describes_model_outputs() -> None:
    """The target template holds the model's output variables, in model order, at the target nodes."""
    task = Forecaster(multistep_input=2, multistep_output=1, timestep="6h")
    forcing = [next(iter(_NAME_TO_INDEX))]
    data_indices = {"data": _make_minimal_index_collection(_NAME_TO_INDEX, forcing=forcing)}
    b, e, g, v = 2, 1, 4, len(_NAME_TO_INDEX)
    batch = build_batch(
        data={"data": torch.randn(b, 3, e, g, v)},
        coordinates={"data": torch.rand(g, 2)},
        layouts={"data": TensorLayout(batch=0, time=1, ensemble=2, grid=3, variables=4)},
        variables={"data": list(_NAME_TO_INDEX)},
    )
    y, template, forcings = task.get_targets(batch, data_indices)

    template = template["data"]
    assert not hasattr(template, "data")
    assert template.variables == data_indices["data"].model.output.ordered_names
    assert forcing[0] not in template.variables
    assert forcings["data"].variables == forcing
    assert template.time_size == y["data"].time_size == 1
    assert torch.equal(template.coordinates, y["data"].coordinates)


def test_forecaster_get_targets_raises_when_batch_is_short_of_time_steps() -> None:
    """A batch sized for an earlier rollout fails before producing an empty slice."""
    task = Forecaster(
        multistep_input=2,
        multistep_output=1,
        timestep="6h",
        rollout={"start": 1, "epoch_increment": 1, "maximum": 2},
    )
    data_indices = _data_indices_single()
    batch = build_batch(
        data={"data": torch.randn(2, 3, 1, 4, len(_NAME_TO_INDEX))},
        coordinates={"data": torch.rand(4, 2)},
        layouts={"data": TensorLayout(batch=0, time=1, ensemble=2, grid=3, variables=4)},
        variables={"data": list(_NAME_TO_INDEX)},
    )
    targets, _, _ = task.get_targets(batch, data_indices, rollout_step=0)
    assert targets["data"].data.shape[1] == 1

    task.rollout.increase(current_epoch=0)

    with pytest.raises(ValueError, match="requires index 3") as exc_info:
        task.get_targets(batch, data_indices, rollout_step=1)

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


# ── BaseForecaster: _advance_gridded_input ────────────────────────────────────


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
    """_advance_gridded_input slides the window and fills with model predictions."""
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
    output_values = torch.zeros((b, n_step_output, e, g, v), dtype=torch.float32)

    updated = task._advance_gridded_input(
        x,
        y_pred,
        output_values,
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
    output_values = torch.zeros((1, 1, 1, 2, 2), dtype=torch.float32)
    output_values[:, 0, 0, :, 0] = torch.tensor([100.0, 200.0])
    output_values[:, 0, 0, :, 1] = torch.tensor([1000.0, 2000.0])

    updated = task._advance_gridded_input(
        x,
        y_pred,
        output_values,
        data_indices=data_indices,
        output_mask=output_mask,
    )

    # prognostic variable, 1st grid point (cutout_mask=True) should be from y_pred,
    # 2nd grid point (cutout_mask=False) should be from the output-time truth
    torch.testing.assert_close(updated[0, -1, 0, :, 0], torch.tensor([10.0, 200.0]))
    # forcing variable should be refreshed from the output-time values for both grid points
    torch.testing.assert_close(updated[0, -1, 0, :, 1], torch.tensor([1000.0, 2000.0]))


@pytest.mark.parametrize(
    ("input_values", "expected_values"),
    [([10.0], [1000.0]), ([10.0, 20.0], [20.0, 1000.0]), ([10.0, 20.0, 30.0], [20.0, 30.0, 1000.0])],
)
def test_rollout_shifts_true_state_into_input_only_grid(
    input_values: list[float],
    expected_values: list[float],
) -> None:
    """A dataset without a prediction shifts its inputs and takes its true state at the new input time."""
    n_input = len(input_values)
    task = Forecaster(multistep_input=n_input, multistep_output=1, timestep="6h")
    layout = TensorLayout(batch=0, time=1, ensemble=2, grid=3, variables=4)
    forecast = torch.arange(1.0, n_input + 1).reshape(1, n_input, 1, 1, 1)
    conditioning = torch.tensor(input_values).reshape(1, n_input, 1, 1, 1).requires_grad_()
    coordinates = {"forecast": torch.zeros(1, 2), "conditioning": torch.ones(1, 2)}
    batch = build_batch(
        data={"forecast": forecast, "conditioning": conditioning},
        coordinates=coordinates,
        layouts=dict.fromkeys(coordinates, layout),
        variables={name: ["A"] for name in coordinates},
        statistics={name: {} for name in coordinates},
    )
    prediction = torch.tensor([float(n_input + 1)]).reshape(1, 1, 1, 1, 1).requires_grad_()
    predicted = batch.with_data({"forecast": prediction})
    # The input-only source has no prediction, so its newest input comes from the output-time values.
    output_values = batch.with_data(
        {"forecast": torch.zeros_like(prediction), "conditioning": torch.full_like(prediction, 1000.0)},
    )
    advanced = task.advance_input(
        batch,
        predicted,
        output_values,
        data_indices={name: _make_minimal_index_collection({"A": 0}) for name in coordinates},
        output_mask={name: NoOutputMask() for name in coordinates},
    )

    torch.testing.assert_close(advanced["forecast"].data.flatten(), torch.arange(2.0, n_input + 2))
    torch.testing.assert_close(advanced["conditioning"].data.flatten(), torch.tensor(expected_values))
    assert advanced["conditioning"].coordinates is coordinates["conditioning"]
    assert advanced["conditioning"].variables == ["A"]
    torch.testing.assert_close(batch["forecast"].data.flatten(), torch.arange(1.0, n_input + 1))
    torch.testing.assert_close(batch["conditioning"].data.flatten(), torch.tensor(input_values))
    (advanced["forecast"].data.sum() + advanced["conditioning"].data.sum()).backward()
    torch.testing.assert_close(prediction.grad, torch.ones_like(prediction))
    # The oldest conditioning input is shifted out; the others stay inputs.
    expected_grad = torch.ones_like(conditioning)
    expected_grad[:, 0] = 0.0
    torch.testing.assert_close(conditioning.grad, expected_grad)


# ── Forecaster: rollout of tabular (observation) datasets ─────────────────────

_OBS_LAYOUT = TensorLayout(ensemble=0, grid=1, variables=2)
SIX_HOURS = 6 * 3600.0


def _obs_data_indices() -> dict[str, IndexCollection]:
    """Data indices of an 'obs' dataset with a prognostic 'p' and a forcing 'f'."""
    return {"obs": _make_minimal_index_collection({"p": 0, "f": 1}, forcing=["f"])}


def _obs_window(nodes: int, p: list[float], f: float | None = None) -> torch.Tensor:
    """An ``(ensemble, nodes, variables)`` window: member ``m`` holds ``p[m]``; ``f``, if given, is a second channel."""
    members = torch.tensor(p).view(-1, 1, 1).expand(-1, nodes, 1)
    if f is None:
        return members
    return torch.cat([members, torch.full_like(members, f)], dim=-1)


def _obs_batch(windows: list[tuple[torch.Tensor, float]], variables: list[str] | None = None) -> Batch:
    """A one-sample 'obs' batch with one time window per ``(data, tag)``.

    The nodes of a window have their coordinates and timedeltas set to its ``tag``, so a test can
    tell which window ended up where.
    """
    data = torch.cat([window for window, _ in windows], dim=1)
    coordinates = torch.cat([torch.full((window.shape[1], 2), tag) for window, tag in windows])
    timedeltas = torch.cat([torch.full((window.shape[1],), tag) for window, tag in windows])
    stops = list(itertools.accumulate(window.shape[1] for window, _ in windows))
    boundaries = tuple(slice(start, stop) for start, stop in zip([0, *stops], stops, strict=False))
    return build_batch(
        data={"obs": [data]},
        coordinates={"obs": [coordinates]},
        timedeltas={"obs": [timedeltas]},
        boundaries={"obs": [boundaries]},
        layouts={"obs": _OBS_LAYOUT},
        variables={"obs": variables or ["p", "f"]},
    )


@pytest.mark.parametrize(("decoded", "expected_p"), [(True, 1.0), (False, 7.0)])
def test_advance_input_feeds_output_obs_window_back_as_input(decoded: bool, expected_p: float) -> None:
    """The next input window is the output window, whatever its size, with the observed forcings.

    A decoded dataset takes its predicted prognostics there; one without a decoder takes the observations.
    """
    task = Forecaster(multistep_input=1, multistep_output=1, timestep="6h")
    x = _obs_batch([(_obs_window(3, p=[0.0], f=0.0), 0.0)])
    truth = _obs_batch([(_obs_window(5, p=[7.0], f=8.0), 6.0)])
    y_pred = _obs_batch([(_obs_window(5, p=[1.0]), 6.0)], variables=["p"]) if decoded else Batch({})

    advanced = task.advance_input(x, y_pred, truth, data_indices=_obs_data_indices())["obs"]

    assert isinstance(advanced, TabularSource)
    assert advanced.boundaries == [(slice(0, 5),)]
    torch.testing.assert_close(advanced.data[0][..., 0], torch.full((1, 5), expected_p))
    torch.testing.assert_close(advanced.data[0][..., 1], torch.full((1, 5), 8.0))
    torch.testing.assert_close(advanced.coordinates[0], torch.full((5, 2), 6.0))
    # The forecast time moves on by one rollout shift.
    torch.testing.assert_close(advanced.timedeltas[0], torch.full((5,), 6.0 - SIX_HOURS))


def test_advance_input_cycles_each_ensemble_member_obs_window() -> None:
    """Each member takes its own predicted obs; the observed forcings are shared by all members."""
    task = Forecaster(multistep_input=1, multistep_output=1, timestep="6h")
    data_indices = _obs_data_indices()
    x = _obs_batch([(_obs_window(3, p=[0.0, 0.0], f=0.0), 0.0)])

    for nodes, member_p, f in [(5, [1.0, 2.0], 8.0), (4, [10.0, 20.0], 9.0)]:
        truth = _obs_batch([(_obs_window(nodes, p=[7.0], f=f), 6.0)])
        y_pred = _obs_batch([(_obs_window(nodes, p=member_p), 6.0)], variables=["p"])

        x = task.advance_input(x, y_pred, truth, data_indices=data_indices)

        torch.testing.assert_close(x["obs"].data[0], _obs_window(nodes, p=member_p, f=f))


def test_advance_input_decoded_obs_window_becomes_most_recent_input() -> None:
    """With two input windows, the latest input moves back one slot and the decoded window follows it."""
    task = Forecaster(multistep_input=2, multistep_output=1, timestep="6h")
    x = _obs_batch([(_obs_window(3, p=[-6.0], f=0.0), -6.0), (_obs_window(4, p=[0.0], f=0.0), 0.0)])
    truth = _obs_batch([(_obs_window(5, p=[7.0], f=8.0), 6.0)])
    y_pred = _obs_batch([(_obs_window(5, p=[1.0]), 6.0)], variables=["p"])

    advanced = task.advance_input(x, y_pred, truth, data_indices=_obs_data_indices())["obs"]

    assert advanced.boundaries == [(slice(0, 4), slice(4, 9))]
    expected = torch.cat([_obs_window(4, p=[0.0], f=0.0), _obs_window(5, p=[1.0], f=8.0)], dim=1)
    torch.testing.assert_close(advanced.data[0], expected)
    torch.testing.assert_close(advanced.timedeltas[0], torch.tensor([0.0] * 4 + [6.0] * 5) - SIX_HOURS)


def test_advance_input_measures_obs_times_from_the_new_forecast_time() -> None:
    """After advancing, the kept history and the new input window are measured from the next forecast time."""
    task = Forecaster(multistep_input=2, multistep_output=1, timestep="6h")
    # Times in seconds from the current forecast time: history at -6h and 0h, the output window at +6h.
    x = _obs_batch([(_obs_window(3, p=[0.0], f=0.0), -SIX_HOURS), (_obs_window(4, p=[0.0], f=0.0), 0.0)])
    truth = _obs_batch([(_obs_window(5, p=[7.0], f=8.0), SIX_HOURS)])

    advanced = task.advance_input(x, Batch({}), truth, data_indices=_obs_data_indices())["obs"]

    torch.testing.assert_close(advanced.timedeltas[0], torch.tensor([-SIX_HOURS] * 4 + [0.0] * 5))


def test_get_targets_measures_rollout_targets_from_the_step_forecast_time() -> None:
    """Targets of a later rollout step are measured from that step's forecast time."""
    task = Forecaster(multistep_input=1, multistep_output=1, timestep="6h", rollout={"start": 2, "maximum": 2})
    # One window per offset of the sample: 0h, +6h, +12h, measured from the sample's forecast time.
    batch = _obs_batch(
        [
            (_obs_window(2, p=[0.0], f=0.0), 0.0),
            (_obs_window(3, p=[1.0], f=0.0), SIX_HOURS),
            (_obs_window(4, p=[2.0], f=0.0), 2 * SIX_HOURS),
        ],
    )

    first, _, first_forcing = task.get_targets(batch, data_indices=_obs_data_indices(), rollout_step=0)
    second, second_template, second_forcing = task.get_targets(batch, data_indices=_obs_data_indices(), rollout_step=1)

    torch.testing.assert_close(first["obs"].timedeltas[0], torch.full((3,), SIX_HOURS))
    torch.testing.assert_close(first_forcing["obs"].timedeltas[0], torch.full((3,), SIX_HOURS))
    torch.testing.assert_close(second["obs"].timedeltas[0], torch.full((4,), SIX_HOURS))
    torch.testing.assert_close(second_forcing["obs"].timedeltas[0], torch.full((4,), SIX_HOURS))
    torch.testing.assert_close(second_template["obs"].timedeltas[0], torch.full((4,), SIX_HOURS))


def test_advance_input_advances_gridded_and_tabular_datasets_together() -> None:
    """One advance_input call advances each dataset of a mixed batch with its own kind of rollout."""
    task = Forecaster(multistep_input=1, multistep_output=1, timestep="6h")
    grid_coordinates = torch.rand(2, 2)

    def mixed_batch(grid_value: float, obs: TabularSource) -> Batch:
        return build_batch(
            data={"grid": torch.full((1, 1, 1, 2, 1), grid_value), "obs": obs.data},
            coordinates={"grid": grid_coordinates, "obs": obs.coordinates},
            timedeltas={"obs": obs.timedeltas},
            boundaries={"obs": obs.boundaries},
            layouts={"grid": TensorLayout(batch=0, time=1, ensemble=2, grid=3, variables=4), "obs": _OBS_LAYOUT},
            variables={"grid": ["A"], "obs": obs.variables},
        )

    x = mixed_batch(1.0, _obs_batch([(_obs_window(3, p=[0.0], f=0.0), 0.0)])["obs"])
    y_pred = mixed_batch(2.0, _obs_batch([(_obs_window(5, p=[1.0]), 6.0)], variables=["p"])["obs"])
    truth = mixed_batch(0.0, _obs_batch([(_obs_window(5, p=[7.0], f=8.0), 6.0)])["obs"])

    advanced = task.advance_input(
        x,
        y_pred,
        truth,
        data_indices={"grid": _make_minimal_index_collection({"A": 0}), **_obs_data_indices()},
        output_mask={"grid": NoOutputMask(), "obs": NoOutputMask()},
    )

    assert advanced.dataset_names == ("grid", "obs")
    torch.testing.assert_close(advanced["grid"].data, torch.full((1, 1, 1, 2, 1), 2.0))
    assert advanced["grid"].coordinates is grid_coordinates
    assert advanced["obs"].boundaries == [(slice(0, 5),)]
    torch.testing.assert_close(advanced["obs"].data[0], _obs_window(5, p=[1.0], f=8.0))


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
    """An OffsetForecaster with Forecaster's offsets has the same advance map, and advances as expected."""
    data_indices = _make_minimal_index_collection(_NAME_TO_INDEX)
    forecaster = Forecaster(multistep_input=n_step_input, multistep_output=n_step_output, timestep="6h")
    offset = _offset_equivalent(n_step_input, n_step_output)

    assert offset._advance_map == forecaster._advance_map

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
    batch = torch.zeros((b, n_step_output, e, g, v), dtype=torch.float32)

    updated = offset._advance_gridded_input(
        x,
        y_pred,
        batch,
        output_mask=NoOutputMask(),
        data_indices=data_indices,
    )
    assert updated[0, :, 0, 0, 0].tolist() == expected


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
    """_advance_gridded_input handles irregular grids that no Forecaster can represent."""
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
    batch = torch.zeros((b, n_output, e, g, v), dtype=torch.float32)

    updated = task._advance_gridded_input(
        x,
        y_pred,
        batch,
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
        # the latest input defines the forecast initialisation time
        (["-12h", "-6h"], ["6h"], "default", "latest input offset must be 0h"),
        (["6h"], ["12h"], "default", "latest input offset must be 0h"),
        # an output must come strictly after every input for a forecasting task
        (["-6h", "0h"], ["0h", "12h"], "default", "strictly greater"),
        (["-6h", "0h"], ["-3h", "3h"], "default", "strictly greater"),
        # no valid shift exists
        (["-6h", "0h"], ["7h", "10h"], "default", "No valid autoregressive rollout shift"),
        # outputs from consecutive rollout steps must not overlap or interleave
        (["-6h", "0h"], ["6h", "12h"], "6h", "is not a valid autoregressive"),
        (["0h"], ["2h", "5h"], "2h", "is not a valid autoregressive"),
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
