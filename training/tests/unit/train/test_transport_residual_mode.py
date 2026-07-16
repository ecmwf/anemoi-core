# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""ResidualPredictionMode wiring and the loud rejection of residual models in other modes."""

from types import SimpleNamespace

import pytest
import torch
from omegaconf import DictConfig

from anemoi.training.train.methods.transport import ResidualPredictionMode
from anemoi.training.train.methods.transport import StatePredictionMode
from anemoi.training.train.methods.transport import TransportTraining


class _RecordingModelModel:
    def __init__(self, pairs: dict) -> None:
        self._residual_pairs = pairs
        self.set_positions: list[int] | None = None

    def set_output_reference_positions(self, positions) -> None:
        self.set_positions = list(positions)


class _Task:
    def __init__(self, inputs: tuple[str, ...], outputs: tuple[str, ...], positions: list[int]) -> None:
        self.input_datasets = inputs
        self.output_datasets = outputs
        self._positions = positions

    def output_to_input_positions(self) -> list[int]:
        return self._positions


def _make_module(
    pairs: dict,
    *,
    inputs: tuple[str, ...] = ("input",),
    outputs: tuple[str, ...] = ("output",),
    positions: list[int] | None = None,
    residual_processors: bool = True,
) -> SimpleNamespace:
    procs = {"output": object()} if residual_processors else None
    return SimpleNamespace(
        model=SimpleNamespace(
            model=_RecordingModelModel(pairs),
            pre_processors_residuals=procs,
            post_processors_residuals=procs,
        ),
        task=_Task(inputs, outputs, positions if positions is not None else [0]),
    )


def test_residual_mode_sets_reference_positions_from_task() -> None:
    module = _make_module({"output": "input"}, positions=[1])
    mode = ResidualPredictionMode(module)
    assert mode._reference_positions == [1]
    assert module.model.model.set_positions == [1]


def test_residual_mode_rejects_model_without_pairs() -> None:
    module = _make_module({})
    with pytest.raises(ValueError, match="residual_prediction"):
        ResidualPredictionMode(module)


def test_residual_mode_rejects_missing_residual_processors() -> None:
    module = _make_module({"output": "input"}, residual_processors=False)
    with pytest.raises(ValueError, match="residual processors"):
        ResidualPredictionMode(module)


def test_residual_mode_rejects_task_missing_source_dataset() -> None:
    module = _make_module({"output": "input"}, inputs=("other",))
    with pytest.raises(ValueError, match="missing_sources"):
        ResidualPredictionMode(module)


def _make_base_training(pairs: dict, prediction_mode_obj, mode_name: str) -> TransportTraining:
    # TransportTraining is the concrete subclass; it inherits the rejection helper unchanged.
    obj = TransportTraining.__new__(TransportTraining)
    obj.model = SimpleNamespace(model=SimpleNamespace(_residual_pairs=pairs))
    obj._prediction_mode = prediction_mode_obj
    obj.config = DictConfig({"training": {"transport": {"prediction_mode": mode_name}}})
    return obj


def test_base_training_rejects_residual_model_in_state_mode() -> None:
    state_mode = StatePredictionMode.__new__(StatePredictionMode)
    obj = _make_base_training({"output": "input"}, state_mode, "state")
    with pytest.raises(ValueError, match="prediction_mode='residual'"):
        obj._reject_residual_model_in_non_residual_mode()


def test_base_training_allows_residual_model_in_residual_mode() -> None:
    residual_mode = ResidualPredictionMode.__new__(ResidualPredictionMode)
    obj = _make_base_training({"output": "input"}, residual_mode, "residual")
    obj._reject_residual_model_in_non_residual_mode()  # must not raise


def test_base_training_allows_non_residual_model_in_state_mode() -> None:
    state_mode = StatePredictionMode.__new__(StatePredictionMode)
    obj = _make_base_training({}, state_mode, "state")
    obj._reject_residual_model_in_non_residual_mode()  # must not raise


# ── raw-batch normalization override ──────────────────────────────────────────
def _base_training_with_mode(prediction_mode_obj) -> TransportTraining:
    obj = TransportTraining.__new__(TransportTraining)
    obj._prediction_mode = prediction_mode_obj
    return obj


def test_normalize_batch_residual_mode_leaves_batch_raw() -> None:
    """Residual mode keeps the batch RAW: _normalize_batch must not touch the tensors."""
    residual_mode = ResidualPredictionMode.__new__(ResidualPredictionMode)
    obj = _base_training_with_mode(residual_mode)

    original = {"input": torch.tensor([1.0, 2.0]), "output": torch.tensor([3.0, 4.0])}
    out = obj._normalize_batch({k: v.clone() for k, v in original.items()})

    assert set(out) == set(original)
    for key in original:
        assert torch.equal(out[key], original[key])


def test_normalize_batch_state_mode_normalizes_in_place() -> None:
    """State mode still normalizes the whole batch via the model pre-processors."""

    class _Doubler:
        def __call__(self, x: torch.Tensor) -> torch.Tensor:
            return x * 2

    state_mode = StatePredictionMode.__new__(StatePredictionMode)
    obj = _base_training_with_mode(state_mode)
    obj.model = SimpleNamespace(pre_processors={"input": _Doubler()})

    out = obj._normalize_batch({"input": torch.tensor([1.0, 2.0])})
    assert torch.equal(out["input"], torch.tensor([2.0, 4.0]))


def test_residual_mode_model_input_normalizes_each_dataset() -> None:
    """ResidualPredictionMode.model_input normalizes raw conditioning inputs once, per dataset."""

    class _Scaler:
        def __init__(self, factor: float) -> None:
            self.factor = factor
            self.seen_in_place: list[bool] = []

        def __call__(self, x: torch.Tensor, in_place: bool = True) -> torch.Tensor:
            self.seen_in_place.append(in_place)
            return x * self.factor

    mode = ResidualPredictionMode.__new__(ResidualPredictionMode)
    pre = {"input": _Scaler(10.0), "forcings": _Scaler(100.0)}
    mode.module = SimpleNamespace(model=SimpleNamespace(pre_processors=pre))

    x = {"input": torch.tensor([1.0, 2.0]), "forcings": torch.tensor([3.0])}
    out = mode.model_input(x)

    assert torch.equal(out["input"], torch.tensor([10.0, 20.0]))
    assert torch.equal(out["forcings"], torch.tensor([300.0]))
    # Never normalize the caller's raw tensors in place (raw x is still needed for the reference).
    assert pre["input"].seen_in_place == [False]
    assert pre["forcings"].seen_in_place == [False]
