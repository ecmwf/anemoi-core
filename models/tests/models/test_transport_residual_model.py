# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Correctness tests for the residual downscaling transport model (raw-batch, three-role).

The target dataset deliberately carries a ``target``-role variable so that its MODEL-output
ordering differs from its DATA-output ordering. This exercises the data-output/model-output
index-space handling in ``compute_residual`` / ``add_interp_to_state``: a bug that only shows up
when the two orderings disagree would be caught here.

Raw-batch contract under test: ``compute_residual`` consumes RAW physical ``y`` and a RAW physical
``interp(source)`` reference, and ``add_interp_to_state`` adds a RAW physical reference back, so the
round trip is exact without any reliance on interpolation/normalization commuting.
"""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from omegaconf import DictConfig
from torch_geometric.data import HeteroData

from anemoi.models.data_indices.collection import IndexCollection
from anemoi.models.models.transport_encoder_processor_decoder import AnemoiTransportResidualModelEncProcDec
from anemoi.models.preprocessing import Processors
from anemoi.models.preprocessing.normalizer import InputNormalizer

# ── index collections ─────────────────────────────────────────────────────────
# target "output": global order u(0) tgt(1) v(2) dp(3) diag(4) force(5)
#   forcing=[force], diagnostic=[diag], target=[tgt] -> prognostic=[u, v, dp]
#   residual channels = prognostic minus direct(dp) minus diagnostics = [u, v]
# source "input": global order u(0) v(1) sforce(2), forcing=[sforce], prognostic=[u, v]
# passthrough "forcings": pf1(0) pf2(1), all forcing (no prognostic, no output)
_TARGET_NAME_TO_INDEX = {"u": 0, "tgt": 1, "v": 2, "dp": 3, "diag": 4, "force": 5}
_SOURCE_NAME_TO_INDEX = {"u": 0, "v": 1, "sforce": 2}
_PASSTHROUGH_NAME_TO_INDEX = {"pf1": 0, "pf2": 1}


def _target_indices() -> IndexCollection:
    cfg = DictConfig({"forcing": ["force"], "diagnostic": ["diag"], "target": ["tgt"]})
    return IndexCollection(cfg, dict(_TARGET_NAME_TO_INDEX))


def _source_indices() -> IndexCollection:
    cfg = DictConfig({"forcing": ["sforce"], "diagnostic": [], "target": []})
    return IndexCollection(cfg, dict(_SOURCE_NAME_TO_INDEX))


def _passthrough_indices() -> IndexCollection:
    cfg = DictConfig({"forcing": ["pf1", "pf2"], "diagnostic": [], "target": []})
    return IndexCollection(cfg, dict(_PASSTHROUGH_NAME_TO_INDEX))


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


def _make_model(
    n_step_input: int = 2,
    n_step_output: int = 1,
    with_passthrough: bool = False,
) -> AnemoiTransportResidualModelEncProcDec:
    model = AnemoiTransportResidualModelEncProcDec.__new__(AnemoiTransportResidualModelEncProcDec)
    if with_passthrough:
        data_indices = {"input": _source_indices(), "forcings": _passthrough_indices(), "output": _target_indices()}
    else:
        data_indices = {"input": _source_indices(), "output": _target_indices()}
    model.data_indices = data_indices
    model._residual_pairs = {"output": "input"}
    model._direct_prediction = {"output": ["dp"]}
    # These are derived by __init__ from data_indices; set them for the bare __new__ instance.
    model._residual_sources = {"input"}
    model._residual_targets = {"output"}
    model._passthrough_datasets = [
        name for name in data_indices if name not in model._residual_sources and name not in model._residual_targets
    ]
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


def _data_output_tensor(values: list[float]) -> torch.Tensor:
    # shape (batch=1, time=1, ensemble=1, grid=1, vars)
    return torch.tensor(values, dtype=torch.float32).reshape(1, 1, 1, 1, -1)


# physical target state in DATA_OUTPUT order [u, tgt, v, dp, diag] (RAW, as kept by the raw batch)
_Y_PHYS = [6.0, 99.0, 8.0, 15.0, 16.0]
# physical interpolated source for residual channels [u, v] (RAW reference)
_INTERP_PHYS = [3.0, 12.0]


def test_compute_residual_index_spaces_and_values() -> None:
    model = _make_model()
    procs = _processors()

    y = _data_output_tensor(_Y_PHYS)  # RAW physical
    x_interp = torch.tensor(_INTERP_PHYS, dtype=torch.float32).reshape(1, 1, 1, 1, 2)  # RAW physical

    out = model.compute_residual(
        y,
        x_interp,
        pre_processors_state=procs["pre_state"],
        pre_processors_residuals=procs["pre_res"],
        target_dataset="output",
        source_dataset="input",
        skip_imputation=True,
    )

    # DATA_OUTPUT order [u, tgt, v, dp, diag]:
    #   u = residual_norm(6-3)/10 = 0.3 ; v = residual_norm(8-12)/20 = -0.2
    #   dp (direct) state-normalized = 15/5 = 3.0 ; diag state-normalized = 16/8 = 2.0
    #   tgt (target-role, untouched) keeps its raw input value 99.0
    expected = _data_output_tensor([0.3, 99.0, -0.2, 3.0, 2.0])
    assert out.shape == y.shape
    assert torch.allclose(out, expected, atol=1e-6)


def test_direct_and_diagnostic_channels_stay_state_normalized() -> None:
    model = _make_model()
    procs = _processors()
    tgt = model.data_indices["output"]

    y = _data_output_tensor(_Y_PHYS)
    x_interp = torch.tensor(_INTERP_PHYS, dtype=torch.float32).reshape(1, 1, 1, 1, 2)
    out = model.compute_residual(
        y,
        x_interp,
        pre_processors_state=procs["pre_state"],
        pre_processors_residuals=procs["pre_res"],
        target_dataset="output",
        source_dataset="input",
        skip_imputation=True,
    )

    dp_pos = tgt.data.output.positions_for_names(["dp"])[0]
    diag_pos = tgt.data.output.positions_for_names(["diag"])[0]
    # direct-prediction and diagnostic channels equal the state-normalized raw y.
    assert torch.allclose(out[..., dp_pos], _data_output_tensor([15.0 / 5.0])[..., 0], atol=1e-6)
    assert torch.allclose(out[..., diag_pos], _data_output_tensor([16.0 / 8.0])[..., 0], atol=1e-6)


def test_compute_residual_rejects_mismatched_interp_channels() -> None:
    model = _make_model()
    procs = _processors()
    y = _data_output_tensor(_Y_PHYS)
    # 1-channel interp instead of the 2 residual channels: torch would broadcast silently.
    x_interp = torch.tensor([3.0], dtype=torch.float32).reshape(1, 1, 1, 1, 1)
    with pytest.raises(ValueError, match="expected 2"):
        model.compute_residual(
            y,
            x_interp,
            pre_processors_state=procs["pre_state"],
            pre_processors_residuals=procs["pre_res"],
            target_dataset="output",
            source_dataset="input",
            skip_imputation=True,
        )


def _reduce_to_model_output(model: AnemoiTransportResidualModelEncProcDec, data_output: torch.Tensor) -> torch.Tensor:
    positions = model.data_indices["output"].model_output_positions_in_data_output
    idx = torch.as_tensor(positions, dtype=torch.long)
    return data_output.index_select(-1, idx)


def test_roundtrip_add_interp_to_state_recovers_physical_state() -> None:
    model = _make_model()
    procs = _processors()

    y = _data_output_tensor(_Y_PHYS)  # RAW physical
    x_interp = torch.tensor(_INTERP_PHYS, dtype=torch.float32).reshape(1, 1, 1, 1, 2)  # RAW physical
    residual_target = model.compute_residual(
        y,
        x_interp,
        pre_processors_state=procs["pre_state"],
        pre_processors_residuals=procs["pre_res"],
        target_dataset="output",
        source_dataset="input",
        skip_imputation=True,
    )

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


def test_calculate_input_dim_uses_source_full_plus_passthrough_full() -> None:
    model = _make_model(n_step_input=2, n_step_output=3, with_passthrough=True)
    model.node_attributes = SimpleNamespace(attr_ndims={"output": 7})
    model.num_output_channels = {"output": len(model.data_indices["output"].model.output.full)}

    # per_step = source input full (u,v,sforce=3) + passthrough "forcings" full (pf1,pf2=2) = 5
    # input_history = 2 * 5 + 7 = 17 ; corrupted_target = 3 * 4 = 12 -> 29
    assert model._passthrough_datasets == ["forcings"]
    assert model._calculate_input_dim("output") == 2 * 5 + 7 + 3 * 4


def test_calculate_input_dim_zero_passthrough_is_source_only() -> None:
    model = _make_model(n_step_input=2, n_step_output=1, with_passthrough=False)
    model.node_attributes = SimpleNamespace(attr_ndims={"output": 4})
    model.num_output_channels = {"output": len(model.data_indices["output"].model.output.full)}

    assert model._passthrough_datasets == []
    # per_step = source input full (3); no target forcing, no pass-through.
    assert model._calculate_input_dim("output") == 2 * 3 + 4 + 1 * 4


def test_build_residual_conditioning_concats_interp_then_passthrough_in_x_order() -> None:
    model = _make_model(n_step_input=2, n_step_output=1, with_passthrough=True)
    grid = 5
    # Stub the interpolation layer: "project" the source onto the (same-size) target grid as identity.
    model.residual = {"output": lambda src, *args, **kwargs: src}

    source = torch.arange(1 * 2 * 1 * grid * 3, dtype=torch.float32).reshape(1, 2, 1, grid, 3)
    passthrough = torch.arange(1 * 2 * 1 * grid * 2, dtype=torch.float32).reshape(1, 2, 1, grid, 2) + 100.0
    x = {"input": source, "forcings": passthrough}

    conditioning = model._build_residual_conditioning(x)["output"]

    # conditioning = interp(source) (3 ch) ++ passthrough (2 ch) in x iteration order.
    assert conditioning.shape == (1, 2, 1, grid, 5)
    assert torch.equal(conditioning[..., :3], source)
    assert torch.equal(conditioning[..., 3:], passthrough)


def test_build_residual_conditioning_requires_source_in_inputs() -> None:
    model = _make_model(with_passthrough=True)
    model.residual = {"output": lambda src, *args, **kwargs: src}
    with pytest.raises(KeyError, match="requires source dataset"):
        model._build_residual_conditioning({"forcings": torch.zeros(1, 2, 1, 4, 2)})


def test_multiple_targets_with_passthrough_raises_ambiguity() -> None:
    # Two residual targets plus a pass-through dataset cannot be disambiguated. This raises at
    # construction (before the graph is even touched), so a minimal config is enough.
    data_indices = {
        "src1": _source_indices(),
        "src2": _source_indices(),
        "pf": _passthrough_indices(),
        "out1": _target_indices(),
        "out2": _target_indices(),
    }
    cfg = DictConfig({"model": {"model": {"residual_prediction": {"out1": "src1", "out2": "src2"}}}})
    with pytest.raises(ValueError, match="ambiguous"):
        AnemoiTransportResidualModelEncProcDec(
            model_config=cfg,
            data_indices=data_indices,
            statistics={},
            n_step_input=1,
            n_step_output=1,
            graph_data=HeteroData(),
        )


def test_set_output_reference_positions_validation() -> None:
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
