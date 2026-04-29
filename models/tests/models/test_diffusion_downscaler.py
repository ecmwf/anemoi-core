# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Tests for AnemoiD2ModelEncProcDec: compute_residuals, add_interp_to_state, and round-trips."""

from __future__ import annotations

import torch
from omegaconf import DictConfig

from anemoi.models.data_indices.collection import IndexCollection
from anemoi.models.models.diffusiondownscaler_encoder_processor_decoder import AnemoiD2ModelEncProcDec

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


class _ScaleBy(torch.nn.Module):
    def __init__(self, factor: float) -> None:
        super().__init__()
        self.factor = factor

    def forward(self, x, in_place=False, **_kwargs):
        return x * self.factor


def _make_mixed_model() -> AnemoiD2ModelEncProcDec:
    """Minimal AnemoiD2ModelEncProcDec mock: 1 prognostic + 1 diagnostic in out_hres."""
    model = object.__new__(AnemoiD2ModelEncProcDec)
    model.data_indices = {
        "in_lres": _make_index_collection({"10u": 0, "10v": 1, "2t": 2}),
        "out_hres": _make_index_collection({"10u": 0, "tp": 1}, diagnostic=["tp"]),
    }
    model._residual_pairs = {"out_hres": "in_lres"}
    model._matching_channel_indices_out_hres = torch.tensor([0])
    model._matching_indices_keys = [("out_hres", "in_lres", "_matching_channel_indices_out_hres")]
    return model


def _make_all_prognostic_model() -> AnemoiD2ModelEncProcDec:
    """All out_hres channels are prognostic (present in both source and target)."""
    model = object.__new__(AnemoiD2ModelEncProcDec)
    model.data_indices = {
        "in_lres": _make_index_collection({"10u": 0, "10v": 1, "2t": 2}),
        "out_hres": _make_index_collection({"10u": 0, "10v": 1}),
    }
    model._residual_pairs = {"out_hres": "in_lres"}
    model._matching_channel_indices_out_hres = torch.tensor([0, 1])
    model._matching_indices_keys = [("out_hres", "in_lres", "_matching_channel_indices_out_hres")]
    return model


def _make_all_diagnostic_model() -> AnemoiD2ModelEncProcDec:
    """All out_hres channels are diagnostic (none present in source)."""
    model = object.__new__(AnemoiD2ModelEncProcDec)
    model.data_indices = {
        "in_lres": _make_index_collection({"10u": 0, "10v": 1}),
        "out_hres": _make_index_collection({"tp": 0, "cp": 1}, diagnostic=["tp", "cp"]),
    }
    model._residual_pairs = {"out_hres": "in_lres"}
    model._matching_channel_indices_out_hres = torch.tensor([], dtype=torch.long)
    model._matching_indices_keys = [("out_hres", "in_lres", "_matching_channel_indices_out_hres")]
    return model


# ============================================================
# _match_tensor_channels
# ============================================================


def test_channel_reindex_correctness() -> None:
    """_match_tensor_channels reindexes a tensor to match output channel order."""
    input_idx = {"t2m": 0, "u10": 1, "v10": 2, "sp": 3}
    output_idx = {"sp": 0, "t2m": 1, "u10": 2}

    model = object.__new__(AnemoiD2ModelEncProcDec)
    indices = model._match_tensor_channels(input_idx, output_idx)

    x = torch.tensor([[10.0, 20.0, 30.0, 40.0]])  # t2m=10, u10=20, v10=30, sp=40
    assert torch.equal(x[..., indices], torch.tensor([[40.0, 10.0, 20.0]]))


# ============================================================
# compute_residuals
# ============================================================


def test_compute_residuals_prognostic_minus_interp() -> None:
    """Prognostic channels get y − x_interp, diagnostic channels get y directly."""
    model = _make_mixed_model()

    target = model.compute_residuals(
        y=torch.tensor([[[[[150.0, 5.0]]]]]),
        x_interp=torch.tensor([[[[[100.0]]]]]),
        pre_processors_state=_IdentityProcessor(),
        pre_processors_tendencies=_IdentityProcessor(),
        target_dataset="out_hres",
        skip_imputation=True,
    )

    prog_out = model.data_indices["out_hres"].model.output.prognostic
    diag_out = model.data_indices["out_hres"].model.output.diagnostic

    assert torch.allclose(target[..., prog_out], torch.tensor([[[[[50.0]]]]]))
    assert torch.allclose(target[..., diag_out], torch.tensor([[[[[5.0]]]]]))


def test_compute_residuals_tendency_processor_applied_to_prog() -> None:
    """Tendency processor is applied to prognostic residuals but NOT to diagnostic."""
    model = _make_mixed_model()

    target = model.compute_residuals(
        y=torch.tensor([[[[[150.0, 5.0]]]]]),
        x_interp=torch.tensor([[[[[100.0]]]]]),
        pre_processors_state=_IdentityProcessor(),
        pre_processors_tendencies=_ScaleBy2Processor(),
        target_dataset="out_hres",
        skip_imputation=True,
    )

    prog_out = model.data_indices["out_hres"].model.output.prognostic
    diag_out = model.data_indices["out_hres"].model.output.diagnostic

    # Prognostic: (150-100) * 2 = 100, Diagnostic: 5.0 (state identity processor)
    assert torch.allclose(target[..., prog_out], torch.tensor([[[[[100.0]]]]]))
    assert torch.allclose(target[..., diag_out], torch.tensor([[[[[5.0]]]]]))


# ============================================================
# add_interp_to_state
# ============================================================


def test_add_interp_adds_to_prognostic_leaves_diagnostic() -> None:
    """add_interp_to_state adds x_interp to prognostic, leaves diagnostic unchanged."""
    model = _make_mixed_model()
    identity = _IdentityProcessor()

    result = model.add_interp_to_state(
        state_inp=torch.tensor([[[[[100.0, 200.0, 300.0]]]]]),  # in_lres: 10u, 10v, 2t
        model_output=torch.tensor([[[[[50.0, 5.0]]]]]),  # out_hres: 10u_residual, tp_direct
        post_processors_state={"in_lres": identity, "out_hres": identity},
        post_processors_tendencies=None,
        target_dataset="out_hres",
        source_dataset="in_lres",
        skip_imputation=True,
    )

    prog_out = model.data_indices["out_hres"].model.output.prognostic
    diag_out = model.data_indices["out_hres"].model.output.diagnostic

    # 10u: 50 + 100 = 150, tp: 5 (unchanged)
    assert torch.allclose(result[..., prog_out], torch.tensor([[[[[150.0]]]]]))
    assert torch.allclose(result[..., diag_out], torch.tensor([[[[[5.0]]]]]))


# ============================================================
# Round-trips
# ============================================================


def test_round_trip_residual_identity_processors() -> None:
    """compute_residuals followed by add_interp_to_state with identity processors recovers y."""
    model = _make_mixed_model()
    identity = _IdentityProcessor()

    y = torch.tensor([[[[[150.0, 5.0]]]]])
    x_interp = torch.tensor([[[[[100.0]]]]])

    residual_target = model.compute_residuals(
        y=y,
        x_interp=x_interp,
        pre_processors_state=identity,
        pre_processors_tendencies=identity,
        target_dataset="out_hres",
        skip_imputation=True,
    )

    y_reconstructed = model.add_interp_to_state(
        state_inp=torch.tensor([[[[[100.0, 200.0, 300.0]]]]]),
        model_output=residual_target,
        post_processors_state={"in_lres": identity, "out_hres": identity},
        post_processors_tendencies={"out_hres": identity},
        target_dataset="out_hres",
        source_dataset="in_lres",
        skip_imputation=True,
    )

    assert torch.allclose(y_reconstructed, y, atol=1e-5), f"Round-trip failed: expected {y}, got {y_reconstructed}"


def test_round_trip_residual_scale_processors() -> None:
    """Round-trip holds when pre/post processors are exact inverses (×2 / ×0.5)."""
    model = _make_mixed_model()
    identity = _IdentityProcessor()

    y = torch.tensor([[[[[150.0, 5.0]]]]])
    x_interp = torch.tensor([[[[[100.0]]]]])

    residual_target = model.compute_residuals(
        y=y,
        x_interp=x_interp,
        pre_processors_state=identity,
        pre_processors_tendencies=_ScaleBy(2.0),
        target_dataset="out_hres",
        skip_imputation=True,
    )

    y_reconstructed = model.add_interp_to_state(
        state_inp=torch.tensor([[[[[100.0, 200.0, 300.0]]]]]),
        model_output=residual_target,
        post_processors_state={"in_lres": identity, "out_hres": identity},
        post_processors_tendencies={"out_hres": _ScaleBy(0.5)},
        target_dataset="out_hres",
        source_dataset="in_lres",
        skip_imputation=True,
    )

    assert torch.allclose(
        y_reconstructed, y, atol=1e-5
    ), f"Round-trip with scaled processors failed: expected {y}, got {y_reconstructed}"


# ============================================================
# _build_matching_channel_indices / get_matching_channel_indices
# ============================================================


def test_build_matching_channel_indices_common_channels() -> None:
    """_build_matching_channel_indices returns source indices in target variable order."""
    model = object.__new__(AnemoiD2ModelEncProcDec)
    model.data_indices = {
        "in_lres": _make_index_collection({"10u": 0, "10v": 1, "2t": 2}),
        # out_hres has same 10u, 10v — but in reversed order
        "out_hres": _make_index_collection({"10v": 0, "10u": 1}),
    }
    model._residual_pairs = {"out_hres": "in_lres"}
    model._matching_channel_indices_out_hres = torch.empty(0)
    model._matching_indices_keys = []

    indices = model._build_matching_channel_indices("out_hres", "in_lres")

    # Output order: 10v first, 10u second → source indices [1, 0]
    assert indices.tolist() == [1, 0]


def test_get_matching_channel_indices_retrieves_buffer() -> None:
    """get_matching_channel_indices returns the pre-built buffer by target dataset name."""
    model = _make_mixed_model()
    result = model.get_matching_channel_indices("out_hres")
    assert torch.equal(result, torch.tensor([0]))


# ============================================================
# compute_residuals — edge cases
# ============================================================


def test_compute_residuals_no_tendency_processor_uses_state() -> None:
    """When pre_processors_tendencies=None, state processor is used for prognostic residuals."""
    model = _make_mixed_model()

    target = model.compute_residuals(
        y=torch.tensor([[[[[150.0, 5.0]]]]]),
        x_interp=torch.tensor([[[[[100.0]]]]]),
        pre_processors_state=_ScaleBy(3.0),  # will be used for prognostic fallback
        pre_processors_tendencies=None,
        target_dataset="out_hres",
        skip_imputation=True,
    )

    prog_out = model.data_indices["out_hres"].model.output.prognostic
    diag_out = model.data_indices["out_hres"].model.output.diagnostic

    # Prognostic fallback: (150 - 100) * 3 = 150
    assert torch.allclose(target[..., prog_out], torch.tensor([[[[[150.0]]]]]))
    # Diagnostic: 5.0 * 3 = 15.0
    assert torch.allclose(target[..., diag_out], torch.tensor([[[[[15.0]]]]]))


def test_compute_residuals_all_prognostic() -> None:
    """All-prognostic model: entire output is residual, no diagnostic branch."""
    model = _make_all_prognostic_model()
    identity = _IdentityProcessor()

    y = torch.tensor([[[[[10.0, 20.0]]]]])
    x_interp = torch.tensor([[[[[1.0, 2.0]]]]])

    target = model.compute_residuals(
        y=y,
        x_interp=x_interp,
        pre_processors_state=identity,
        pre_processors_tendencies=identity,
        target_dataset="out_hres",
        skip_imputation=True,
    )

    assert torch.allclose(target, y - x_interp, atol=1e-6)


def test_compute_residuals_all_diagnostic() -> None:
    """All-diagnostic model: entire output is direct prediction, x_interp is unused."""
    model = _make_all_diagnostic_model()
    identity = _IdentityProcessor()

    y = torch.tensor([[[[[7.0, 3.0]]]]])
    x_interp = torch.zeros(1, 1, 1, 1, 0)  # empty — no matching channels

    target = model.compute_residuals(
        y=y,
        x_interp=x_interp,
        pre_processors_state=identity,
        pre_processors_tendencies=identity,
        target_dataset="out_hres",
        skip_imputation=True,
    )

    # No residual: output equals y unchanged
    assert torch.allclose(target, y, atol=1e-6)


# ============================================================
# add_interp_to_state — edge cases
# ============================================================


def test_add_interp_all_prognostic() -> None:
    """All-prognostic: add x_interp to all channels via get_matching_channel_indices."""
    model = _make_all_prognostic_model()
    identity = _IdentityProcessor()

    state_inp = torch.tensor([[[[[10.0, 20.0, 30.0]]]]])  # in_lres (3 vars)
    model_output = torch.tensor([[[[[5.0, 8.0]]]]])  # residuals for 10u, 10v

    result = model.add_interp_to_state(
        state_inp=state_inp,
        model_output=model_output,
        post_processors_state={"in_lres": identity, "out_hres": identity},
        post_processors_tendencies={"out_hres": identity},
        target_dataset="out_hres",
        source_dataset="in_lres",
        skip_imputation=True,
    )

    # 10u: 5 + 10 = 15, 10v: 8 + 20 = 28
    assert torch.allclose(result, torch.tensor([[[[[15.0, 28.0]]]]]), atol=1e-6)


def test_add_interp_all_diagnostic() -> None:
    """All-diagnostic: no x_interp added, output is direct denorm only."""
    model = _make_all_diagnostic_model()
    identity = _IdentityProcessor()

    state_inp = torch.tensor([[[[[10.0, 20.0]]]]])  # in_lres (irrelevant, no prog channels)
    model_output = torch.tensor([[[[[7.0, 3.0]]]]])

    result = model.add_interp_to_state(
        state_inp=state_inp,
        model_output=model_output,
        post_processors_state={"in_lres": identity, "out_hres": identity},
        post_processors_tendencies={"out_hres": identity},
        target_dataset="out_hres",
        source_dataset="in_lres",
        skip_imputation=True,
    )

    assert torch.allclose(result, model_output, atol=1e-6)


def test_add_interp_no_tendency_processor() -> None:
    """When post_processors_tendencies=None, state processor is used for denorm."""
    model = _make_mixed_model()
    identity = _IdentityProcessor()
    scale2 = _ScaleBy(2.0)

    # With tendency=None, state post-processor (scale2) is used to denorm model_output
    result = model.add_interp_to_state(
        state_inp=torch.tensor([[[[[100.0, 200.0, 300.0]]]]]),
        model_output=torch.tensor([[[[[50.0, 5.0]]]]]),
        post_processors_state={"in_lres": identity, "out_hres": scale2},
        post_processors_tendencies=None,
        target_dataset="out_hres",
        source_dataset="in_lres",
        skip_imputation=True,
    )

    prog_out = model.data_indices["out_hres"].model.output.prognostic
    diag_out = model.data_indices["out_hres"].model.output.diagnostic

    # state post (scale2) applied to all: [50*2, 5*2] = [100, 10]
    # diagnostic overwritten by state post on diag slice: 5*2 = 10
    # prognostic: state_denorm[prog] + x_source_denorm[matching] = 100 + identity(100) = 200
    assert torch.allclose(result[..., prog_out], torch.tensor([[[[[200.0]]]]]), atol=1e-6)
    assert torch.allclose(result[..., diag_out], torch.tensor([[[[[10.0]]]]]), atol=1e-6)
