# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Model-side correctness tests for the residual downscaling inference remnant.

The residual target math now lives in ``ResidualPredictionMode`` (see
``training/tests/unit/train/test_transport_residual_mode.py``); this file covers only what stays
model-side: the reference-alignment buffer, the channel-index helpers, and the physical
reconstruction (:meth:`add_interp_to_state`).

The target dataset deliberately carries a ``target``-role variable so that its MODEL-output
ordering differs from its DATA-output ordering. This exercises the data-output/model-output
index-space handling in ``add_interp_to_state``: a bug that only shows up when the two orderings
disagree would be caught here. ``add_interp_to_state`` adds a RAW physical ``interp(source)``
reference back, so the reconstruction is exact without any reliance on interpolation/normalization
commuting.
"""

import numpy as np
import torch
from omegaconf import DictConfig

from anemoi.models.data_indices.collection import IndexCollection
from anemoi.models.models.transport_encoder_processor_decoder import AnemoiTransportResidualModelEncProcDec
from anemoi.models.preprocessing import Processors
from anemoi.models.preprocessing.normalizer import InputNormalizer

# ── index collections ─────────────────────────────────────────────────────────
# target "output": global order u(0) tgt(1) v(2) dp(3) diag(4) force(5)
#   forcing=[force], diagnostic=[diag], target=[tgt] -> prognostic=[u, v, dp]
#   residual channels = prognostic minus direct(dp) minus diagnostics = [u, v]
# source "input": global order u(0) v(1) sforce(2), forcing=[sforce], prognostic=[u, v]
_TARGET_NAME_TO_INDEX = {"u": 0, "tgt": 1, "v": 2, "dp": 3, "diag": 4, "force": 5}
_SOURCE_NAME_TO_INDEX = {"u": 0, "v": 1, "sforce": 2}


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


# per-channel stdevs (mean-std normalization with zero mean => norm = x / stdev)
_TARGET_STATE_STDEV = [2.0, 1.0, 4.0, 5.0, 8.0, 1.0]  # u tgt v dp diag force
_TARGET_RESIDUAL_STDEV = [10.0, 1.0, 20.0, 1.0, 1.0, 1.0]  # only u, v matter
_SOURCE_STATE_STDEV = [3.0, 6.0, 1.0]  # u v sforce


def _make_model(n_step_input: int = 2, n_step_output: int = 1) -> AnemoiTransportResidualModelEncProcDec:
    """Bare ``__new__`` instance carrying only what the kept model-side methods consult."""
    model = AnemoiTransportResidualModelEncProcDec.__new__(AnemoiTransportResidualModelEncProcDec)
    model.data_indices = {"input": _source_indices(), "output": _target_indices()}
    model._residual_pairs = {"output": "input"}
    model._direct_prediction = {"output": ["dp"]}
    model.n_step_input = n_step_input
    model.n_step_output = n_step_output
    return model


def _processors() -> dict[str, dict[str, Processors]]:
    tgt = _target_indices()
    src = _source_indices()
    pre_t, post_t = _processor_pair(tgt, _norm(_TARGET_STATE_STDEV))
    pre_s, post_s = _processor_pair(src, _norm(_SOURCE_STATE_STDEV))
    pre_res, post_res = _processor_pair(tgt, _norm(_TARGET_RESIDUAL_STDEV))
    return {
        "pre_state": {"output": pre_t, "input": pre_s},
        "post_state": {"output": post_t, "input": post_s},
        "pre_res": {"output": pre_res},
        "post_res": {"output": post_res},
    }


# ── channel-index helpers ──────────────────────────────────────────────────────
def test_channel_name_resolution() -> None:
    model = _make_model()
    # u, v are prognostic and not direct -> residual; dp is direct; diag is diagnostic.
    assert model._residual_names("output") == ["u", "v"]
    assert model._direct_names("output") == ["dp"]
    assert model._diagnostic_names("output") == ["diag"]


def test_get_matching_channel_indices() -> None:
    model = _make_model()
    matching = model.get_matching_channel_indices("output")
    # source data.input.full order is [u, v, sforce]; residual channels [u, v] -> positions [0, 1]
    assert matching.tolist() == [0, 1]


# ── add_interp_to_state reconstruction (physical, MODEL_OUTPUT layout) ──────────
def test_add_interp_to_state_reconstructs_physical_state_by_hand() -> None:
    model = _make_model()
    procs = _processors()
    tgt = model.data_indices["output"]
    n_model_out = len(tgt.model.output.full)

    # Network output in MODEL_OUTPUT layout, placed by name (residual-normalized on u, v;
    # state-normalized on dp, diag).
    model_output = torch.zeros(1, 1, 1, 1, n_model_out, dtype=torch.float32)
    for name, val in {"u": 0.3, "v": -0.2, "dp": 3.0, "diag": 2.0}.items():
        model_output[..., int(tgt.model.output.name_to_index[name])] = val

    # RAW interpolated source, FULL source input channels [u, v, sforce]; sforce is irrelevant.
    state_inp = torch.tensor([3.0, 12.0, 0.0], dtype=torch.float32).reshape(1, 1, 1, 1, 3)

    out = model.add_interp_to_state(
        state_inp,
        model_output,
        post_processors_state=procs["post_state"],
        post_processors_residuals=procs["post_res"],
        target_dataset="output",
        source_dataset="input",
        skip_imputation=True,
    )

    # Hand-computed physical state:
    #   u = 0.3 * residual_stdev[u]=10 + interp_u=3  = 6
    #   v = -0.2 * residual_stdev[v]=20 + interp_v=12 = 8
    #   dp = 3.0 * state_stdev[dp]=5   = 15   (direct prediction, state-normalized)
    #   diag = 2.0 * state_stdev[diag]=8 = 16 (diagnostic, state-normalized)
    expected = {"u": 6.0, "v": 8.0, "dp": 15.0, "diag": 16.0}
    for name, val in expected.items():
        pos = int(tgt.model.output.name_to_index[name])
        assert torch.allclose(out[..., pos], torch.tensor(val), atol=1e-5), name


# ── reference-alignment buffer ──────────────────────────────────────────────────
def test_set_output_reference_positions_validation() -> None:
    import pytest

    model = _make_model(n_step_input=2, n_step_output=2)
    torch.nn.Module.__init__(model)
    model.register_buffer("_output_reference_positions", torch.full((2,), -1, dtype=torch.long))

    with pytest.raises(ValueError, match="length n_step_output"):
        model.set_output_reference_positions([0])
    with pytest.raises(ValueError, match=r"\[0, n_step_input"):
        model.set_output_reference_positions([0, 2])  # 2 == n_step_input, out of range
    with pytest.raises(ValueError, match=r"\[0, n_step_input"):
        model.set_output_reference_positions([-1, 0])

    model.set_output_reference_positions([1, 0])
    assert model._output_reference_positions.tolist() == [1, 0]


def test_select_reference_source_steps_rejects_unset_sentinel() -> None:
    import pytest

    model = _make_model(n_step_input=2, n_step_output=2)
    torch.nn.Module.__init__(model)
    model.register_buffer("_output_reference_positions", torch.full((2,), -1, dtype=torch.long))

    source_history = torch.randn(1, 2, 1, 3, 3)
    with pytest.raises(RuntimeError, match="unset"):
        model._select_reference_source_steps(source_history)

    model.set_output_reference_positions([1, 0])
    selected = model._select_reference_source_steps(source_history)
    assert selected.shape == (1, 2, 1, 3, 3)
    assert torch.allclose(selected[:, 0], source_history[:, 1])
    assert torch.allclose(selected[:, 1], source_history[:, 0])
