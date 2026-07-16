# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""ResidualPredictionMode wiring, the residual-target math (absorbed from the model), and the loud
rejection of residual models in other modes.

The residual-target construction (formerly ``AnemoiTransportResidualModelEncProcDec.compute_residual``)
now lives in ``ResidualPredictionMode._build_residual_target``. Its correctness tests — hand-computed
values, a PERMUTED model/data ordering fixture, and the round-trip residual -> reconstruction — live
here. The reconstruction half (``add_interp_to_state``) is still a model method, so the round trip
exercises the mode and the model together.
"""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from omegaconf import DictConfig

from anemoi.models.data_indices.collection import IndexCollection
from anemoi.models.models.transport_encoder_processor_decoder import AnemoiTransportResidualModelEncProcDec
from anemoi.models.preprocessing import Processors
from anemoi.models.preprocessing.normalizer import InputNormalizer
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


# ── residual-target math (absorbed from the model's former compute_residual) ────
# Permuted ordering fixture: the target carries a "target"-role variable (tgt) so its MODEL-output
# order differs from its DATA-output order. A bug that only shows up when the two disagree is caught.
# target "output": global order u(0) tgt(1) v(2) dp(3) diag(4) force(5)
#   forcing=[force], diagnostic=[diag], target=[tgt] -> prognostic=[u, v, dp]
#   residual channels = prognostic minus direct(dp) = [u, v]; model.output=[u, v, dp, diag]
# source "input": global order u(0) v(1) sforce(2), forcing=[sforce], prognostic=[u, v]
_TARGET_NAME_TO_INDEX = {"u": 0, "tgt": 1, "v": 2, "dp": 3, "diag": 4, "force": 5}
_SOURCE_NAME_TO_INDEX = {"u": 0, "v": 1, "sforce": 2}

# per-channel stdevs (mean-std normalization with zero mean => norm = x / stdev)
_TARGET_STATE_STDEV = [2.0, 1.0, 4.0, 5.0, 8.0, 1.0]  # u tgt v dp diag force
_TARGET_RESIDUAL_STDEV = [10.0, 1.0, 20.0, 1.0, 1.0, 1.0]  # only u, v matter
_SOURCE_STATE_STDEV = [3.0, 6.0, 1.0]  # u v sforce

# physical target state in DATA_OUTPUT order [u, tgt, v, dp, diag] (RAW, as kept by the raw batch)
_Y_PHYS = [6.0, 99.0, 8.0, 15.0, 16.0]
# physical interpolated source for residual channels [u, v] (RAW reference)
_INTERP_PHYS = [3.0, 12.0]


def _target_indices() -> IndexCollection:
    cfg = DictConfig({"forcing": ["force"], "diagnostic": ["diag"], "target": ["tgt"]})
    return IndexCollection(cfg, dict(_TARGET_NAME_TO_INDEX))


def _source_indices() -> IndexCollection:
    cfg = DictConfig({"forcing": ["sforce"], "diagnostic": [], "target": []})
    return IndexCollection(cfg, dict(_SOURCE_NAME_TO_INDEX))


def _norm(stdev: list[float]) -> dict:
    n = len(stdev)
    return {
        "mean": np.zeros(n, dtype=np.float32),
        "stdev": np.asarray(stdev, dtype=np.float32),
        "minimum": np.zeros(n, dtype=np.float32),
        "maximum": np.ones(n, dtype=np.float32),
    }


def _processor_pair(indices: IndexCollection, stats: dict) -> tuple[Processors, Processors]:
    cfg = DictConfig({"default": "mean-std"})
    pre = Processors([["normalizer", InputNormalizer(cfg, indices, {k: v.copy() for k, v in stats.items()})]])
    post = Processors(
        [["normalizer", InputNormalizer(cfg, indices, {k: v.copy() for k, v in stats.items()})]], inverse=True
    )
    return pre, post


def _bare_model() -> AnemoiTransportResidualModelEncProcDec:
    """Bare ``__new__`` model carrying only what the index/reconstruction helpers consult."""
    model = AnemoiTransportResidualModelEncProcDec.__new__(AnemoiTransportResidualModelEncProcDec)
    model.data_indices = {"input": _source_indices(), "output": _target_indices()}
    model._residual_pairs = {"output": "input"}
    model._direct_prediction = {"output": ["dp"]}
    model.n_step_input = 2
    model.n_step_output = 1
    return model


def _residual_math_mode() -> tuple[ResidualPredictionMode, AnemoiTransportResidualModelEncProcDec, dict]:
    """Build a mode over minimal stubs plus the processors the residual-target math needs."""
    model = _bare_model()
    pre_t, post_t = _processor_pair(_target_indices(), _norm(_TARGET_STATE_STDEV))
    pre_s, post_s = _processor_pair(_source_indices(), _norm(_SOURCE_STATE_STDEV))
    pre_res, post_res = _processor_pair(_target_indices(), _norm(_TARGET_RESIDUAL_STDEV))
    procs = {
        "pre_state": {"output": pre_t, "input": pre_s},
        "post_state": {"output": post_t, "input": post_s},
        "pre_res": {"output": pre_res},
        "post_res": {"output": post_res},
    }
    mode = ResidualPredictionMode.__new__(ResidualPredictionMode)
    mode._residual_pairs = model._residual_pairs
    mode.module = SimpleNamespace(
        model=SimpleNamespace(
            model=model,
            pre_processors=procs["pre_state"],
            pre_processors_residuals=procs["pre_res"],
        )
    )
    return mode, model, procs


def _data_output_tensor(values: list[float]) -> torch.Tensor:
    # shape (batch=1, time=1, ensemble=1, grid=1, vars)
    return torch.tensor(values, dtype=torch.float32).reshape(1, 1, 1, 1, -1)


def _reduce_to_model_output(model: AnemoiTransportResidualModelEncProcDec, data_output: torch.Tensor) -> torch.Tensor:
    positions = model.data_indices["output"].model_output_positions_in_data_output
    idx = torch.as_tensor(positions, dtype=torch.long)
    return data_output.index_select(-1, idx)


def test_build_residual_target_index_spaces_and_values() -> None:
    mode, _model, _procs = _residual_math_mode()
    y = _data_output_tensor(_Y_PHYS)  # RAW physical
    x_interp = torch.tensor(_INTERP_PHYS, dtype=torch.float32).reshape(1, 1, 1, 1, 2)  # RAW physical

    out = mode._build_residual_target(y, x_interp, target_dataset="output", skip_imputation=True)

    # DATA_OUTPUT order [u, tgt, v, dp, diag]:
    #   u = residual_norm(6-3)/10 = 0.3 ; v = residual_norm(8-12)/20 = -0.2
    #   dp (direct) state-normalized = 15/5 = 3.0 ; diag state-normalized = 16/8 = 2.0
    #   tgt (target-role, untouched) keeps its raw input value 99.0
    expected = _data_output_tensor([0.3, 99.0, -0.2, 3.0, 2.0])
    assert out.shape == y.shape
    assert torch.allclose(out, expected, atol=1e-6)


def test_build_residual_target_direct_and_diagnostic_stay_state_normalized() -> None:
    mode, model, _procs = _residual_math_mode()
    tgt = model.data_indices["output"]

    y = _data_output_tensor(_Y_PHYS)
    x_interp = torch.tensor(_INTERP_PHYS, dtype=torch.float32).reshape(1, 1, 1, 1, 2)
    out = mode._build_residual_target(y, x_interp, target_dataset="output", skip_imputation=True)

    dp_pos = tgt.data.output.positions_for_names(["dp"])[0]
    diag_pos = tgt.data.output.positions_for_names(["diag"])[0]
    # direct-prediction and diagnostic channels equal the state-normalized raw y.
    assert torch.allclose(out[..., dp_pos], _data_output_tensor([15.0 / 5.0])[..., 0], atol=1e-6)
    assert torch.allclose(out[..., diag_pos], _data_output_tensor([16.0 / 8.0])[..., 0], atol=1e-6)


def test_build_residual_target_rejects_mismatched_interp_channels() -> None:
    mode, _model, _procs = _residual_math_mode()
    y = _data_output_tensor(_Y_PHYS)
    # 1-channel interp instead of the 2 residual channels: torch would broadcast silently.
    x_interp = torch.tensor([3.0], dtype=torch.float32).reshape(1, 1, 1, 1, 1)
    with pytest.raises(ValueError, match="expected 2"):
        mode._build_residual_target(y, x_interp, target_dataset="output", skip_imputation=True)


def test_roundtrip_residual_target_then_add_interp_recovers_physical_state() -> None:
    mode, model, procs = _residual_math_mode()

    y = _data_output_tensor(_Y_PHYS)  # RAW physical
    x_interp = torch.tensor(_INTERP_PHYS, dtype=torch.float32).reshape(1, 1, 1, 1, 2)  # RAW physical
    residual_target = mode._build_residual_target(y, x_interp, target_dataset="output", skip_imputation=True)

    # Network consumes/emits MODEL_OUTPUT layout: reduce the data-output residual target.
    model_target = _reduce_to_model_output(model, residual_target)

    # Reference: RAW physical full source interp (u, v, sforce). sforce is irrelevant.
    state_inp = torch.tensor([_INTERP_PHYS[0], _INTERP_PHYS[1], 0.0], dtype=torch.float32).reshape(1, 1, 1, 1, 3)

    reconstructed = model.add_interp_to_state(
        state_inp,
        model_target,
        post_processors_state=procs["post_state"],
        post_processors_residuals=procs["post_res"],
        target_dataset="output",
        source_dataset="input",
        skip_imputation=True,
    )

    # physical y in MODEL_OUTPUT order [u, v, dp, diag]
    expected = torch.tensor([6.0, 8.0, 15.0, 16.0], dtype=torch.float32).reshape(1, 1, 1, 1, 4)
    assert reconstructed.shape == expected.shape
    assert torch.allclose(reconstructed, expected, atol=1e-5)
