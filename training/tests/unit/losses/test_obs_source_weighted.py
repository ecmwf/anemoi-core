# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from types import SimpleNamespace

import pytest
import torch
from omegaconf import DictConfig

from anemoi.models.data_indices.collection import IndexCollection
from anemoi.training.losses import MSELoss
from anemoi.training.losses import ObsSourceWeightedMSELoss
from anemoi.training.losses import get_loss_function
from anemoi.training.utils.enums import TensorDim
from anemoi.training.utils.index_space import IndexSpace

NAMES = ["lsm", "z_500", "t_500", "u_500", "v_500", "z_850", "t_850", "u_850", "v_850", "10u"]
N2I = {n: i for i, n in enumerate(reversed(NAMES))}  # scrambled data order
NAN = float("nan")
GRID = 4


def _indices() -> IndexCollection:
    return IndexCollection(DictConfig({"forcing": ["lsm"], "diagnostic": []}), N2I)


def _loss(di: IndexCollection, **kwargs) -> ObsSourceWeightedMSELoss:
    loss = ObsSourceWeightedMSELoss(data_indices=di, **kwargs)
    loss.add_scaler(TensorDim.GRID, torch.full((GRID,), 1.0 / GRID), name="node_weights")
    return loss


def _target(di: IndexCollection, columns: dict[str, list[float]]) -> torch.Tensor:
    """DATA_FULL target (1, 1, 1, GRID, n); unspecified variables are NaN (lsm is 0)."""
    full = torch.full((1, 1, 1, GRID, len(N2I)), NAN)
    full[..., di.name_to_index["lsm"]] = 0.0
    for name, values in columns.items():
        full[0, 0, 0, :, di.name_to_index[name]] = torch.tensor(values)
    return full


# Cell 0: sonde (z, t, u, v); cell 1: aircraft (t, u, v); cells 2-3: AMV (u, v only). Level 850 only.
SOURCES_850 = {
    "z_850": [0.0, NAN, NAN, NAN],
    "t_850": [0.0, 0.0, NAN, NAN],
    "u_850": [0.0, 0.0, 0.0, 0.0],
    "v_850": [0.0, 0.0, 0.0, 0.0],
}


def _pred(di: IndexCollection, offset: float = 1.0) -> torch.Tensor:
    return torch.full((1, 1, 1, GRID, len(di.model.output.full)), offset)


def test_sources_classified_from_companion_presence() -> None:
    di = _indices()
    loss = _loss(di)
    masks = loss.source_masks(_target(di, SOURCES_850))
    col = loss.weighted_names.index("u_850")
    assert masks["sonde"][0, 0, 0, :, col].tolist() == [True, False, False, False]
    assert masks["aircraft"][0, 0, 0, :, col].tolist() == [False, True, False, False]
    assert masks["amv"][0, 0, 0, :, col].tolist() == [False, False, True, True]
    # No target at 500 hPa: nothing classified there.
    col500 = loss.weighted_names.index("u_500")
    assert not any(masks[s][..., col500].any() for s in masks)


def test_unit_weights_equal_plain_mse() -> None:
    di = _indices()
    target = _target(di, SOURCES_850)
    pred = _pred(di, 0.5)
    weighted = _loss(di, weights={"amv": 1.0, "aircraft": 1.0, "sonde": 1.0})
    plain = MSELoss(ignore_nans=True)
    plain.add_scaler(TensorDim.GRID, torch.full((GRID,), 1.0 / GRID), name="node_weights")
    expected = plain(pred, target[..., di.model_output_positions_in_data_full], squash=False)
    got = weighted(pred, target, squash=False, pred_layout=IndexSpace.MODEL_OUTPUT, target_layout=IndexSpace.DATA_FULL)
    torch.testing.assert_close(got, expected)


def test_source_weights_without_share_preservation() -> None:
    di = _indices()
    loss = _loss(di, weights={"amv": 0.25, "sonde": 3.0}, preserve_variable_share=False)
    per_var = loss(_pred(di), _target(di, SOURCES_850), squash=False)
    pos = di.model.output.name_to_position
    # Error 1 everywhere; node weight 1/4; weights sonde 3, aircraft 1, amv 0.25 x2.
    assert per_var[pos["u_850"]].item() == pytest.approx((3.0 + 1.0 + 0.25 + 0.25) / 4)
    assert per_var[pos["v_850"]].item() == pytest.approx((3.0 + 1.0 + 0.25 + 0.25) / 4)
    # t and z are not weighted by default; t_850 has two valid targets, z_850 one.
    assert per_var[pos["t_850"]].item() == pytest.approx(2.0 / 4)
    assert per_var[pos["z_850"]].item() == pytest.approx(1.0 / 4)
    assert loss.last_source_fractions == pytest.approx({"amv": 0.5, "aircraft": 0.25, "sonde": 0.25})


def test_share_preservation_keeps_variable_total_and_shifts_mix() -> None:
    di = _indices()
    target = _target(di, SOURCES_850)
    plain = _loss(di, weights={})(_pred(di), target, squash=False)
    loss = _loss(di, weights={"amv": 0.25, "sonde": 3.0}, preserve_variable_share=True)
    weighted = loss(_pred(di), target, squash=False)
    pos = di.model.output.name_to_position["u_850"]
    # Uniform error: the renormalised weights average to one, so the variable total is unchanged.
    assert weighted[pos].item() == pytest.approx(plain[pos].item())
    weights = loss.element_weights(target)[0, 0, 0, :, loss.weighted_names.index("u_850")]
    mean = (3.0 + 1.0 + 0.25 + 0.25) / 4
    torch.testing.assert_close(weights, torch.tensor([3.0, 1.0, 0.25, 0.25]) / mean)


def test_t_family_splits_sonde_and_aircraft() -> None:
    di = _indices()
    loss = _loss(di, variables=["t"], weights={"aircraft": 0.5, "sonde": 2.0}, preserve_variable_share=False)
    per_var = loss(_pred(di), _target(di, SOURCES_850), squash=False)
    pos = di.model.output.name_to_position
    assert per_var[pos["t_850"]].item() == pytest.approx((2.0 + 0.5) / 4)
    assert per_var[pos["u_850"]].item() == pytest.approx(1.0)  # winds unweighted when not listed


def test_built_by_factory_without_variable_mapper() -> None:
    di = _indices()
    config = DictConfig(
        {
            "_target_": "anemoi.training.losses.ObsSourceWeightedMSELoss",
            "scalers": ["node_weights"],
            "weights": {"amv": 0.3},
        },
    )
    scalers = {"node_weights": ((TensorDim.GRID.value,), torch.full((GRID,), 1.0 / GRID))}
    loss = get_loss_function(config, scalers=scalers, data_indices=di)
    assert isinstance(loss, ObsSourceWeightedMSELoss)
    value = loss(_pred(di), _target(di, SOURCES_850), pred_layout="model_output", target_layout="data_full")
    assert torch.isfinite(value)


def test_data_output_target_matches_data_full() -> None:
    """DASingleTraining slices targets to DATA_OUTPUT before the loss."""
    di = _indices()
    loss = _loss(di, weights={"amv": 0.25, "sonde": 3.0})
    full = _target(di, SOURCES_850)
    data_output = full[..., di.data.output.full]
    expected = loss(_pred(di), full, squash=False, target_layout="data_full")
    got = loss(_pred(di), data_output, squash=False, target_layout=IndexSpace.DATA_OUTPUT)
    torch.testing.assert_close(got, expected)
    torch.testing.assert_close(loss(_pred(di), data_output, squash=False), expected)  # layout inferred


def test_rejects_wrong_layout_and_bad_config() -> None:
    di = _indices()
    loss = _loss(di)
    with pytest.raises(ValueError, match="not supported"):
        loss(_pred(di), _target(di, SOURCES_850)[..., :5], target_layout="data_output")
    with pytest.raises(ValueError, match="unknown sources"):
        _loss(di, weights={"satob": 0.1})
    with pytest.raises(ValueError, match="no model output matches"):
        _loss(di, variables=["q"])
    no_z = IndexCollection(DictConfig({"forcing": [], "diagnostic": []}), {"t_850": 0, "u_850": 1, "v_850": 2})
    with pytest.raises(ValueError, match="z_850"):
        ObsSourceWeightedMSELoss(data_indices=no_z)


RPT_NAMES = [*NAMES, "sonde_reportype"]
RPT_N2I = {n: i for i, n in enumerate(reversed(RPT_NAMES))}


def _rpt_indices() -> IndexCollection:
    return IndexCollection(DictConfig({"forcing": ["lsm"], "diagnostic": [], "target": ["sonde_reportype"]}), RPT_N2I)


def _raw_normalizer(scale_rpt: float = 1.0) -> SimpleNamespace:
    mul = torch.ones(len(RPT_N2I))
    mul[RPT_N2I["sonde_reportype"]] = scale_rpt
    return SimpleNamespace(_norm_mul=mul, _norm_add=torch.zeros(len(RPT_N2I)))


def _rpt_target(di: IndexCollection, codes: list[float]) -> torch.Tensor:
    # Cell 0 sonde with z; cell 1 wind-only sonde (PILOT, code 3); cell 2 AMV; cell 3 no code, aircraft pattern.
    columns = {
        "z_850": [0.0, NAN, NAN, NAN],
        "t_850": [0.0, NAN, NAN, 0.0],
        "u_850": [0.0, 0.0, 0.0, 0.0],
        "v_850": [0.0, 0.0, 0.0, 0.0],
        "sonde_reportype": codes,
    }
    full = torch.full((1, 1, 1, GRID, len(RPT_N2I)), NAN)
    full[..., di.name_to_index["lsm"]] = 0.0
    for name, values in columns.items():
        full[0, 0, 0, :, di.name_to_index[name]] = torch.tensor(values)
    return full


def test_reportype_codes_override_pattern_and_fall_back() -> None:
    di = _rpt_indices()
    loss = ObsSourceWeightedMSELoss(data_indices=di, reportype_variable="sonde_reportype", normalizer=_raw_normalizer())
    masks = loss.source_masks(_rpt_target(di, [3.0, 3.0, 1.5, NAN]))
    col = loss.weighted_names.index("u_850")
    # The wind-only sonde (cell 1) stays a sonde; cell 3 has no code and falls back to the pattern.
    assert masks["sonde"][0, 0, 0, :, col].tolist() == [True, True, False, False]
    assert masks["amv"][0, 0, 0, :, col].tolist() == [False, False, True, False]
    assert masks["aircraft"][0, 0, 0, :, col].tolist() == [False, False, False, True]


def test_pattern_inference_uses_any_z_in_the_column() -> None:
    di = _indices()
    loss = _loss(di)
    target = _target(di, {"z_500": [0.0, NAN, NAN, NAN], "u_850": [0.0, 0.0, 0.0, 0.0]})
    masks = loss.source_masks(target)
    col = loss.weighted_names.index("u_850")
    # Cell 0 has z at 500 hPa only: still a sonde at 850 hPa.
    assert masks["sonde"][0, 0, 0, :, col].tolist() == [True, False, False, False]
    assert masks["amv"][0, 0, 0, :, col].tolist() == [False, True, True, True]


def test_data_output_scalers_restricted_to_model_outputs() -> None:
    di = _rpt_indices()
    assert len(di.data.output.full) == len(di.model.output.full) + 1  # the loss-only target variable
    loss = ObsSourceWeightedMSELoss(
        data_indices=di,
        reportype_variable="sonde_reportype",
        weights={"amv": 0.5},
        preserve_variable_share=False,
        normalizer=_raw_normalizer(),
    )
    loss.add_scaler(TensorDim.GRID, torch.full((GRID,), 1.0 / GRID), name="node_weights")
    out_pos = di.data.output.name_to_position
    general = torch.ones(len(di.data.output.full))
    general[out_pos["u_850"]] = 10.0
    loss.add_scaler(TensorDim.VARIABLE, general, name="general_variable")
    pred = torch.ones((1, 1, 1, GRID, len(di.model.output.full)))
    per_var = loss(pred, _rpt_target(di, [3.0, 3.0, 1.5, NAN]), squash=False)
    pos = di.model.output.name_to_position
    # u_850: weights sonde 1, sonde 1, amv 0.5, aircraft 1 -> 3.5 / 4, times the general weight 10.
    assert per_var[pos["u_850"]].item() == pytest.approx(10.0 * 3.5 / 4)
    loss.update_scaler("general_variable", 2.0 * general)
    per_var = loss(pred, _rpt_target(di, [3.0, 3.0, 1.5, NAN]), squash=False)
    assert per_var[pos["u_850"]].item() == pytest.approx(20.0 * 3.5 / 4)


def test_reportype_must_be_raw_and_present() -> None:
    di = _rpt_indices()
    with pytest.raises(ValueError, match="normaliser to 'none'"):
        ObsSourceWeightedMSELoss(data_indices=di, reportype_variable="sonde_reportype", normalizer=_raw_normalizer(0.5))
    with pytest.raises(ValueError, match="not in the data"):
        ObsSourceWeightedMSELoss(data_indices=_indices(), reportype_variable="sonde_reportype")


def test_reportype_target_in_data_output_layout() -> None:
    di = _rpt_indices()
    loss = ObsSourceWeightedMSELoss(
        data_indices=di,
        reportype_variable="sonde_reportype",
        weights={"amv": 0.5, "sonde": 2.0},
        normalizer=_raw_normalizer(),
    )
    loss.add_scaler(TensorDim.GRID, torch.full((GRID,), 1.0 / GRID), name="node_weights")
    full = _rpt_target(di, [3.0, 3.0, 1.5, NAN])
    data_output = full[..., di.data.output.full]
    pred = torch.ones((1, 1, 1, GRID, len(di.model.output.full)))
    expected = loss(pred, full, squash=False, target_layout="data_full")
    got = loss(pred, data_output, squash=False, target_layout="data_output")
    torch.testing.assert_close(got, expected)
    masks = loss.source_masks(data_output, "data_output")
    assert masks["sonde"][0, 0, 0, :, loss.weighted_names.index("u_850")].tolist() == [True, True, False, False]
