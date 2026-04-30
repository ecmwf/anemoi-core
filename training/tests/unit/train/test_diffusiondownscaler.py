# (C) Copyright 2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Tests for GraphDiffusionDownscaler and diffusion downscaler model."""

from __future__ import annotations

import torch
from omegaconf import DictConfig

from anemoi.models.data_indices.collection import IndexCollection
from anemoi.training.train.tasks.diffusiondownscaler import GraphDiffusionDownscaler

# ============================================================
# Helpers
# ============================================================


def _make_index_collection(
    name_to_index: dict[str, int],
    forcing: list[str] | None = None,
    diagnostic: list[str] | None = None,
) -> IndexCollection:
    cfg = DictConfig({"forcing": forcing or [], "diagnostic": diagnostic or [], "target": []})
    return IndexCollection(cfg, name_to_index)


class _IdentityProcessor(torch.nn.Module):
    def forward(self, x, in_place=False, **_kwargs):
        return x if in_place else x.clone()


class _ScaleBy2Processor(torch.nn.Module):
    def forward(self, x, in_place=False, **_kwargs):
        return x * 2.0


def _make_mixed_model():
    """Create a minimal AnemoiD2ModelEncProcDec mock with mixed prognostic/diagnostic."""
    from anemoi.models.models.diffusiondownscaler_encoder_processor_decoder import AnemoiD2ModelEncProcDec

    model = object.__new__(AnemoiD2ModelEncProcDec)
    model.data_indices = {
        "in_lres": _make_index_collection({"10u": 0, "10v": 1, "2t": 2}),
        "out_hres": _make_index_collection({"10u": 0, "tp": 1}, diagnostic=["tp"]),
    }
    model._residual_pairs = {"out_hres": "in_lres"}
    model._matching_channel_indices_out_hres = torch.tensor([0])
    model._matching_indices_keys = [("out_hres", "in_lres", "_matching_channel_indices_out_hres")]
    return model


# ============================================================
# Tests
# ============================================================


def test_get_noise_level_dict():
    """_get_noise_level returns dict[str, Tensor] for sigma and weight."""
    ds = GraphDiffusionDownscaler.__new__(GraphDiffusionDownscaler)
    ds.rho = 7.0
    ds.lognormal_mean = -1.2
    ds.lognormal_std = 1.2
    ds.training_approach = "probabilistic_low_noise"
    ds._residual_pairs = {"out_hres": "in_lres"}

    shape = {"out_hres": (4, 1, 1, 1, 1)}
    sigma, weight = ds._get_noise_level(
        shape=shape,
        sigma_max=100000.0,
        sigma_min=0.02,
        sigma_data=1.0,
        rho=7.0,
        device=torch.device("cpu"),
    )
    assert isinstance(sigma, dict) and "out_hres" in sigma
    assert sigma["out_hres"].shape == (4, 1, 1, 1, 1)
    assert (sigma["out_hres"] > 0).all()
    assert (weight["out_hres"] > 0).all()


def test_noise_scales_with_sigma():
    """_noise_target produces larger noise for larger sigma (dict interface)."""
    ds = GraphDiffusionDownscaler.__new__(GraphDiffusionDownscaler)
    x = {"out_hres": torch.zeros(2, 1, 1, 10, 3)}

    torch.manual_seed(42)
    result_small = ds._noise_target(x, {"out_hres": torch.full((2, 1, 1, 1, 1), 0.01)})
    torch.manual_seed(42)
    result_large = ds._noise_target(x, {"out_hres": torch.full((2, 1, 1, 1, 1), 100.0)})

    assert result_large["out_hres"].abs().mean() > result_small["out_hres"].abs().mean()


def test_channel_reindex_correctness():
    """_match_tensor_channels reindexes a tensor to match output channel order."""
    from anemoi.models.models.diffusiondownscaler_encoder_processor_decoder import AnemoiD2ModelEncProcDec

    input_idx = {"t2m": 0, "u10": 1, "v10": 2, "sp": 3}
    output_idx = {"sp": 0, "t2m": 1, "u10": 2}
    indices = AnemoiD2ModelEncProcDec._match_tensor_channels(None, input_idx, output_idx)

    x = torch.tensor([[10.0, 20.0, 30.0, 40.0]])  # t2m=10, u10=20, v10=30, sp=40
    assert torch.equal(x[..., indices], torch.tensor([[40.0, 10.0, 20.0]]))


def test_mixed_residual_target():
    """Prognostic channels get y-x_interp, diagnostic channels get y directly."""
    out_indices = _make_index_collection({"10u": 0, "tp": 1}, diagnostic=["tp"])
    prog_out = out_indices.model.output.prognostic
    diag_out = out_indices.model.output.diagnostic

    y = torch.tensor([[[[[150.0, 5.0]]]]])  # 10u=150, tp=5
    x_interp = torch.tensor([[[[[100.0]]]]])  # matched 10u

    target = y.clone()
    target[..., prog_out] = y[..., prog_out] - x_interp  # 150 - 100 = 50

    assert torch.allclose(target[..., prog_out], torch.tensor([[[[[50.0]]]]]))
    assert torch.allclose(target[..., diag_out], torch.tensor([[[[[5.0]]]]]))


def test_compute_residuals_with_tendency():
    """compute_residuals applies tendency processor to prognostic, state processor to diagnostic."""
    model = _make_mixed_model()

    target = model.compute_residuals(
        y=torch.tensor([[[[[150.0, 5.0]]]]]),
        x_interp=torch.tensor([[[[[100.0]]]]]),
        pre_processors_state=_IdentityProcessor(),
        pre_processors_tendencies=_ScaleBy2Processor(),
    )

    prog_out = model.data_indices["out_hres"].model.output.prognostic
    diag_out = model.data_indices["out_hres"].model.output.diagnostic

    # Prognostic: (150-100) * 2 = 100, Diagnostic: 5.0 (identity)
    assert torch.allclose(target[..., prog_out], torch.tensor([[[[[100.0]]]]]))
    assert torch.allclose(target[..., diag_out], torch.tensor([[[[[5.0]]]]]))


def test_add_interp_reconstruction():
    """add_interp_to_state adds x_interp to prognostic channels, leaves diagnostic alone."""
    model = _make_mixed_model()
    identity = _IdentityProcessor()

    result = model.add_interp_to_state(
        state_inp=torch.tensor([[[[[100.0, 200.0, 300.0]]]]]),  # in_lres: 10u, 10v, 2t
        model_output=torch.tensor([[[[[50.0, 5.0]]]]]),  # out_hres: 10u_residual, tp_direct
        post_processors_state={"in_lres": identity, "out_hres": identity},
        post_processors_tendencies=None,
    )

    prog_out = model.data_indices["out_hres"].model.output.prognostic
    diag_out = model.data_indices["out_hres"].model.output.diagnostic

    # 10u: 50 + 100 = 150, tp: 5 (unchanged)
    assert torch.allclose(result[..., prog_out], torch.tensor([[[[[150.0]]]]]))
    assert torch.allclose(result[..., diag_out], torch.tensor([[[[[5.0]]]]]))


def _make_dp_model():
    """Model where tp is prognostic but direct-predicted (not residual)."""
    from anemoi.models.models.diffusiondownscaler_encoder_processor_decoder import AnemoiD2ModelEncProcDec

    model = object.__new__(AnemoiD2ModelEncProcDec)
    model.data_indices = {
        "in_lres": _make_index_collection({"10u": 0, "10v": 1, "tp": 2}),
        "out_hres": _make_index_collection({"10u": 0, "10v": 1, "tp": 2}),
    }
    model._residual_pairs = {"out_hres": "in_lres"}
    model._matching_channel_indices_out_hres = torch.tensor([0, 1, 2])
    model._matching_indices_keys = [("out_hres", "in_lres", "_matching_channel_indices_out_hres")]
    # dp buffers: tp (model idx 2) is direct-predicted
    model._direct_prediction_indices_out_hres = torch.tensor([2], dtype=torch.long)
    model._direct_prediction_data_indices_out_hres = torch.tensor([2], dtype=torch.long)
    return model


def test_compute_residuals_with_dp():
    """dp vars get raw y (state-normalized), not residual (tendency-normalized)."""
    model = _make_dp_model()

    target = model.compute_residuals(
        y=torch.tensor([[[[[150.0, 200.0, 5.0]]]]]),
        x_interp=torch.tensor([[[[[100.0, 150.0, 3.0]]]]]),
        pre_processors_state=_IdentityProcessor(),
        pre_processors_tendencies=_ScaleBy2Processor(),
    )

    # 10u: (150-100)*2=100, 10v: (200-150)*2=100, tp: identity(5)=5 (raw, state-normalized)
    assert torch.allclose(target, torch.tensor([[[[[100.0, 100.0, 5.0]]]]]))


def test_add_interp_with_dp():
    """dp vars get state-denormalized raw prediction, no x_interp addition."""
    model = _make_dp_model()
    identity = _IdentityProcessor()

    result = model.add_interp_to_state(
        state_inp=torch.tensor([[[[[100.0, 150.0, 3.0]]]]]),
        model_output=torch.tensor([[[[[50.0, 50.0, 5.0]]]]]),
        post_processors_state={"in_lres": identity, "out_hres": identity},
        post_processors_tendencies=None,
    )

    # 10u: 50+100=150, 10v: 50+150=200, tp: identity(5)=5 (no x_interp)
    assert torch.allclose(result, torch.tensor([[[[[150.0, 200.0, 5.0]]]]]))


def test_add_interp_with_dp_and_tendencies():
    """dp overwrite is correct even when tendency processors are used for initial denorm."""
    model = _make_dp_model()
    identity = _IdentityProcessor()
    scale2 = _ScaleBy2Processor()

    result = model.add_interp_to_state(
        state_inp=torch.tensor([[[[[100.0, 150.0, 3.0]]]]]),
        model_output=torch.tensor([[[[[50.0, 50.0, 5.0]]]]]),
        post_processors_state={"in_lres": identity, "out_hres": identity},
        post_processors_tendencies={"out_hres": scale2},
    )

    # tendency denorm all: [100,100,10] → prognostic += state(inp): [200,250,13]
    # dp overwrite: state_denorm(model[...,2]) = identity(5) = 5
    assert torch.allclose(result, torch.tensor([[[[[200.0, 250.0, 5.0]]]]]))


def test_resolve_direct_prediction_indices():
    """Field names are correctly resolved to model-space and data-space indices."""
    from anemoi.training.train.tasks.diffusiondownscaler import _resolve_direct_prediction_indices

    data_indices = _make_index_collection({"10u": 0, "10v": 1, "tp": 2})
    dp_model_idx, dp_data_idx = _resolve_direct_prediction_indices(["tp"], data_indices)

    assert dp_model_idx is not None
    assert dp_model_idx.tolist() == [2]
    assert dp_data_idx.tolist() == [2]


def test_resolve_direct_prediction_empty():
    """Empty dp_fields returns (None, None)."""
    from anemoi.training.train.tasks.diffusiondownscaler import _resolve_direct_prediction_indices

    data_indices = _make_index_collection({"10u": 0, "10v": 1, "tp": 2})
    dp_model_idx, dp_data_idx = _resolve_direct_prediction_indices([], data_indices)

    assert dp_model_idx is None
    assert dp_data_idx is None


def test_resolve_direct_prediction_skips_diagnostic():
    """dp field that is diagnostic (not prognostic) is skipped with warning."""
    from anemoi.training.train.tasks.diffusiondownscaler import _resolve_direct_prediction_indices

    data_indices = _make_index_collection({"10u": 0, "tp": 1}, diagnostic=["tp"])
    dp_model_idx, dp_data_idx = _resolve_direct_prediction_indices(["tp"], data_indices)

    # tp is diagnostic, not prognostic — should be skipped
    assert dp_model_idx is None
    assert dp_data_idx is None


def test_apply_interpolate_to_high_res_output_shape():
    """apply_interpolate_to_high_res returns (batch, 1, 1, grid_hres, vars) given a 4-D input."""
    from anemoi.models.models.diffusiondownscaler_encoder_processor_decoder import AnemoiD2ModelEncProcDec

    batch, grid_lres, grid_hres, n_vars = 2, 16, 64, 3

    # Mock residual callable: takes (batch, time, ensemble, grid, vars) → same shape with grid_hres
    def _fake_residual(x_5d, grid_shard_shapes=None, model_comm_group=None):
        b, t, ens, _, v = x_5d.shape
        return torch.zeros(b, t, ens, grid_hres, v)

    model = object.__new__(AnemoiD2ModelEncProcDec)
    model.residual = {"in_lres": _fake_residual}

    x = torch.zeros(batch, 1, grid_lres, n_vars)  # (batch, ensemble=1, grid_lres, vars)
    out = model.apply_interpolate_to_high_res(x)

    assert out.shape == (batch, 1, 1, grid_hres, n_vars), (
        f"Expected (batch=2, 1, 1, grid_hres=64, vars=3), got {tuple(out.shape)}"
    )


def test_dp_buffer_round_trip():
    """Buffer registered by task init is readable by _get_direct_prediction_indices."""
    import torch.nn as nn

    from anemoi.models.models.diffusiondownscaler_encoder_processor_decoder import AnemoiD2ModelEncProcDec
    from anemoi.training.train.tasks.diffusiondownscaler import _resolve_direct_prediction_indices

    # Simulate what task.__init__ does: resolve → register
    data_indices = _make_index_collection({"10u": 0, "10v": 1, "tp": 2})
    dp_model_idx, dp_data_idx = _resolve_direct_prediction_indices(["tp"], data_indices)

    model = object.__new__(AnemoiD2ModelEncProcDec)
    # AnemoiD2ModelEncProcDec inherits register_buffer from nn.Module; init nn.Module state manually
    nn.Module.__init__(model)

    model.register_buffer("_direct_prediction_indices_out_hres", dp_model_idx, persistent=True)
    model.register_buffer("_direct_prediction_data_indices_out_hres", dp_data_idx, persistent=True)

    # Verify _get_direct_prediction_indices reads them back correctly
    result_model, result_data = model._get_direct_prediction_indices("out_hres")
    assert result_model is not None
    assert result_model.tolist() == [2]
    assert result_data is not None
    assert result_data.tolist() == [2]
