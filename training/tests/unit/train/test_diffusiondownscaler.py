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
