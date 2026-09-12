# (C) Copyright 2024- Anemoi contributors.
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
from anemoi.training.losses import CombinedLoss
from anemoi.training.losses import RefractivityOperatorLoss
from anemoi.training.losses import get_loss_function
from anemoi.training.losses.refractivity import G0
from anemoi.training.losses.refractivity import K1_DRY
from anemoi.training.losses.refractivity import refractivity_at_heights
from anemoi.training.losses.utils import check_loss_tree_variable_units
from anemoi.training.utils.index_space import IndexSpace

R_D = 287.06
LEVELS = [1000, 700, 300, 100]
# 5000 gpm sits in the 700-300 layer, 12000 gpm in the 300-100 layer, 30000 gpm above the ladder.
OBS = [("refrac_5000", 5000.0), ("refrac_12000", 12000.0), ("refrac_30000", 30000.0)]
SIGMA = 0.005


def _name_to_index(moist: bool = True) -> dict[str, int]:
    names = ["lsm"]
    for prefix in ("z", "t") + (("q",) if moist else ()):
        names += [f"{prefix}_{lvl}" for lvl in LEVELS]
    names += [name for name, _ in OBS]
    # Deliberately scramble the data order so position bookkeeping is exercised.
    order = names[::-1]
    return {name: i for i, name in enumerate(order)}


def _indices(moist: bool = True) -> IndexCollection:
    cfg = DictConfig({"forcing": ["lsm"], "diagnostic": [], "target": [name for name, _ in OBS]})
    return IndexCollection(cfg, _name_to_index(moist))


def _normalizer(data_indices: IndexCollection, seed: int = 0) -> SimpleNamespace:
    """Affine normaliser with identity on forcing/observation columns and random affine elsewhere."""
    g = torch.Generator().manual_seed(seed)
    n = len(data_indices.name_to_index)
    mul = torch.ones(n)
    add = torch.zeros(n)
    for name, idx in data_indices.name_to_index.items():
        # Mimic mean-std / min-max scaling: multipliers of order 1/std for each variable class.
        if name.startswith(("z_", "t_")):
            mul[idx] = torch.rand(1, generator=g) * 0.01 + 0.001
            add[idx] = torch.randn(1, generator=g)
        elif name.startswith("q_"):
            mul[idx] = torch.rand(1, generator=g) * 900.0 + 100.0
            add[idx] = torch.randn(1, generator=g)
    return SimpleNamespace(_norm_mul=mul, _norm_add=add)


def _physical_column(grid: int, moist: bool = True, seed: int = 1) -> dict[str, torch.Tensor]:
    """Hydrostatic columns with a lapse rate and exponentially decaying humidity."""
    g = torch.Generator().manual_seed(seed)
    p = torch.tensor(LEVELS, dtype=torch.float64)
    t_sfc = 280.0 + 10.0 * torch.randn(grid, 1, generator=g, dtype=torch.float64)
    t = t_sfc - 40.0 * torch.log(1000.0 / p)  # cools with height
    tv = t
    phi = torch.zeros_like(t)
    phi[:, 0] = 100.0 * torch.rand(grid, generator=g, dtype=torch.float64) * G0
    for k in range(1, len(LEVELS)):
        phi[:, k] = phi[:, k - 1] + R_D * 0.5 * (tv[:, k - 1] + tv[:, k]) * torch.log(p[k - 1] / p[k])
    cols = {"phi": phi, "t": t}
    if moist:
        cols["q"] = 8e-3 * torch.exp(-phi / (R_D * 250.0 * 1.5))
    return cols


def _make_pred_and_target(
    data_indices: IndexCollection,
    normalizer: SimpleNamespace,
    cols: dict[str, torch.Tensor],
    n_obs: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Normalised MODEL_OUTPUT prediction and DATA_FULL target (obs physical) for one (b,t,e) slice."""
    grid = n_obs.shape[0]
    full = torch.full((grid, len(data_indices.name_to_index)), float("nan"), dtype=torch.float32)
    for k, lvl in enumerate(LEVELS):
        for prefix, key in (("z", "phi"), ("t", "t"), ("q", "q")):
            if key not in cols:
                continue
            idx = data_indices.name_to_index[f"{prefix}_{lvl}"]
            full[:, idx] = (cols[key][:, k].float() * normalizer._norm_mul[idx] + normalizer._norm_add[idx]).float()
    for j, (name, _) in enumerate(OBS):
        if j < n_obs.shape[1]:
            full[:, data_indices.name_to_index[name]] = n_obs[:, j].float()
    full[:, data_indices.name_to_index["lsm"]] = 0.5
    pred = full[:, data_indices.model_output_positions_in_data_full].clone()
    return pred[None, None, None], full[None, None, None]


def _loss(data_indices: IndexCollection, normalizer: SimpleNamespace, **kwargs) -> RefractivityOperatorLoss:
    levels = [{"name": name, "height": h, "sigma": SIGMA} for name, h in OBS]
    defaults = {"levels": levels, "pressure_levels": LEVELS, "moist": True, "penalty_weight": 1.0}
    defaults.update(kwargs)
    return RefractivityOperatorLoss(data_indices=data_indices, normalizer=normalizer, **defaults)


def _true_refractivity(loss: RefractivityOperatorLoss, cols: dict[str, torch.Tensor]) -> torch.Tensor:
    n, _, _ = refractivity_at_heights(
        cols["phi"].float(),
        cols["t"].float(),
        cols.get("q", torch.zeros_like(cols["t"])).float() if loss.moist else None,
        loss.ln_p,
        loss.phi_target,
        moist=loss.moist,
        interp=loss.interp,
        q_interp=loss.q_interp,
    )
    return n


# --------------------------------------------------------------------------- operator


def test_operator_isothermal_dry_column_matches_closed_form() -> None:
    temp = 250.0
    p = torch.tensor([1000.0, 925.0, 850.0, 700.0, 500.0, 400.0, 300.0, 250.0, 200.0, 150.0, 100.0, 70.0, 50.0])
    phi = R_D * temp * torch.log(1000.0 / p)
    heights = torch.tensor([8000.0, 13000.0, 19500.0, 25000.0]) * G0
    for interp in ("linear_lnp", "hydrostatic_shape"):
        n, valid, _ = refractivity_at_heights(
            phi[None],
            torch.full_like(p, temp)[None],
            None,
            torch.log(p),
            heights,
            moist=False,
            interp=interp,
        )
        p_h = 1000.0 * torch.exp(-heights / (R_D * temp))
        expected = K1_DRY * p_h / temp
        assert valid.tolist() == [[True, True, True, False]]
        torch.testing.assert_close(n[0, :3], expected[:3], rtol=1e-5, atol=0)
        assert n[0, 3] == 1.0


def test_operator_moist_term_increases_refractivity() -> None:
    cols = _physical_column(grid=5)
    ln_p = torch.log(torch.tensor(LEVELS, dtype=torch.float64))
    heights = torch.tensor([5000.0 * G0], dtype=torch.float64)
    dry, _, _ = refractivity_at_heights(cols["phi"], cols["t"], None, ln_p, heights, moist=False)
    wet, _, _ = refractivity_at_heights(cols["phi"], cols["t"], cols["q"], ln_p, heights, moist=True)
    assert torch.all(wet > dry)


@pytest.mark.parametrize("interp", ["linear_lnp", "hydrostatic_shape"])
@pytest.mark.parametrize("q_interp", ["linear", "log"])
def test_operator_gradcheck(interp: str, q_interp: str) -> None:
    cols = _physical_column(grid=3)
    ln_p = torch.log(torch.tensor(LEVELS, dtype=torch.float64))
    heights = torch.tensor([5000.0 * G0, 12000.0 * G0], dtype=torch.float64)
    phi = cols["phi"].clone().requires_grad_(True)
    t = cols["t"].clone().requires_grad_(True)
    q = cols["q"].clone().requires_grad_(True)

    def fn(phi_: torch.Tensor, t_: torch.Tensor, q_: torch.Tensor) -> torch.Tensor:
        n, _, _ = refractivity_at_heights(phi_, t_, q_, ln_p, heights, moist=True, interp=interp, q_interp=q_interp)
        return torch.log(n)

    assert torch.autograd.gradcheck(fn, (phi, t, q), eps=1e-6, atol=1e-6, rtol=1e-4)


# --------------------------------------------------------------------------- loss


def test_loss_recovers_physical_operator_through_normalisation() -> None:
    """Observations offset by exactly one sigma give a level loss of one, whatever the normalisation."""
    di = _indices()
    norm = _normalizer(di)
    loss = _loss(di, norm)
    cols = _physical_column(grid=6)
    n_true = _true_refractivity(loss, cols)
    n_obs = n_true * torch.exp(torch.tensor(SIGMA))
    n_obs[:, 2] = float("nan")  # unbracketed level has no obs anyway
    pred, target = _make_pred_and_target(di, norm, cols, n_obs)

    value = loss(pred, target, pred_layout=IndexSpace.MODEL_OUTPUT, target_layout=IndexSpace.DATA_FULL)
    torch.testing.assert_close(value, torch.tensor(1.0), rtol=1e-3, atol=1e-4)
    torch.testing.assert_close(loss.last_level_losses[:2], torch.ones(2), rtol=1e-3, atol=1e-4)
    assert loss.last_level_losses[2] == 0.0
    assert loss.last_level_counts.tolist() == [6, 6, 0]


def test_target_layouts_are_equivalent() -> None:
    di = _indices()
    norm = _normalizer(di)
    loss = _loss(di, norm)
    cols = _physical_column(grid=4)
    n_obs = _true_refractivity(loss, cols) * 1.002
    pred, target_full = _make_pred_and_target(di, norm, cols, n_obs)
    target_out = target_full[..., di.data.output.full]

    v_full = loss(pred, target_full, pred_layout="model_output", target_layout="data_full")
    v_out = loss(pred, target_out, pred_layout="model_output", target_layout="data_output")
    v_inferred = loss(pred, target_out)
    torch.testing.assert_close(v_full, v_out)
    torch.testing.assert_close(v_full, v_inferred)
    with pytest.raises(ValueError, match="does not carry observation"):
        loss(pred, pred, target_layout="model_output")


def test_all_nan_observations_give_zero_loss_and_finite_gradients() -> None:
    di = _indices()
    norm = _normalizer(di)
    loss = _loss(di, norm)
    cols = _physical_column(grid=4)
    n_obs = torch.full((4, len(OBS)), float("nan"))
    pred, target = _make_pred_and_target(di, norm, cols, n_obs)
    pred.requires_grad_(True)
    value = loss(pred, target, target_layout="data_full")
    assert value.item() == 0.0
    value.backward()
    assert torch.isfinite(pred.grad).all()


def test_unbracketed_and_ambiguous_observations_are_masked_locally() -> None:
    di = _indices()
    norm = _normalizer(di)
    loss = _loss(di, norm)
    cols = _physical_column(grid=5)
    n_obs = _true_refractivity(loss, cols)
    n_obs[:, 2] = 10.0  # pretend an obs exists above the ladder
    # Disorder node 0 (swap z_700 and z_300): 5000 gpm is then bracketed by two layers
    # (ambiguous -> masked) while 12000 gpm still has exactly one bracketing layer (kept).
    cols["phi"][0, 1], cols["phi"][0, 2] = cols["phi"][0, 2].clone(), cols["phi"][0, 1].clone()
    pred, target = _make_pred_and_target(di, norm, cols, n_obs)
    value = loss(pred, target, target_layout="data_full")
    assert torch.isfinite(value)
    assert loss.last_level_counts.tolist() == [4, 5, 0]
    # 15 finite obs: 5 above the ladder (no bracket), 1 ambiguous at the disordered node.
    torch.testing.assert_close(loss.last_unbracketed_fraction, torch.tensor(5 / 15))
    torch.testing.assert_close(loss.last_ambiguous_fraction, torch.tensor(1 / 15))
    # 5 nodes x 3 layers; node 0 has two disordered layers (700->300 and 300->100 both inverted? no: only 700->300).
    assert 0.0 < loss.last_disordered_layer_fraction < 0.5


def test_minimum_thickness_hinge_is_zero_for_physical_columns_and_active_when_disordered() -> None:
    di = _indices()
    norm = _normalizer(di)
    loss = _loss(di, norm, monotonicity_penalty_weight=2.0)
    cols = _physical_column(grid=4)
    n_obs = torch.full((4, len(OBS)), float("nan"))  # no obs: only the hinge can contribute
    pred, target = _make_pred_and_target(di, norm, cols, n_obs)
    assert loss(pred, target, target_layout="data_full").item() == 0.0
    assert loss.last_monotonicity_penalty.item() == 0.0
    assert loss.last_disordered_layer_fraction.item() == 0.0

    # Collapse the 700->300 layer at node 1 to a quarter of its minimum thickness.
    dphi_min = loss.dphi_min[1].double()
    cols["phi"][1, 2] = cols["phi"][1, 1] + 0.25 * dphi_min
    cols["phi"][1, 3] = cols["phi"][1, 2] + 2.0 * loss.dphi_min[2].double()
    pred, target = _make_pred_and_target(di, norm, cols, n_obs)
    pred.requires_grad_(True)
    value = loss(pred, target, target_layout="data_full")
    # deficit = 0.75 at one of 4*3 (node, layer) pairs with uniform node weights -> 2 * 0.75^2 / 12
    torch.testing.assert_close(value, torch.tensor(2.0 * 0.75**2 / 12), rtol=1e-3, atol=1e-5)
    value.backward()
    pos = di.model.output.name_to_position
    g_lo = pred.grad[0, 0, 0, 1, pos["z_700"]]
    g_hi = pred.grad[0, 0, 0, 1, pos["z_300"]]
    # Denormalised gradient pushes the lower level down and the upper level up.
    assert g_lo * norm._norm_mul[di.name_to_index["z_700"]] > 0
    assert g_hi * norm._norm_mul[di.name_to_index["z_300"]] < 0
    # Not yet disordered (still increasing), so the disordered fraction stays zero while the hinge is active.
    assert loss.last_disordered_layer_fraction.item() == 0.0


def test_gradients_reach_bracketing_ladder_variables() -> None:
    di = _indices()
    norm = _normalizer(di)
    loss = _loss(di, norm)
    cols = _physical_column(grid=3)
    n_obs = _true_refractivity(loss, cols) * 1.01
    pred, target = _make_pred_and_target(di, norm, cols, n_obs)
    pred.requires_grad_(True)
    loss(pred, target, target_layout="data_full").backward()
    grad = pred.grad[0, 0, 0]
    pos = di.model.output.name_to_position
    # Both observed heights lie in layers touching 700, 300 and 100 hPa; 1000 hPa is never bracketing.
    for prefix in ("z", "t", "q"):
        assert grad[:, pos[f"{prefix}_1000"]].abs().sum() == 0
        for lvl in (700, 300, 100):
            assert grad[:, pos[f"{prefix}_{lvl}"]].abs().sum() > 0


def test_huber_limits_outliers() -> None:
    di = _indices()
    norm = _normalizer(di)
    k = 3.0
    loss = _loss(di, norm, huber_delta_sigmas=k)
    cols = _physical_column(grid=2)
    n_obs = _true_refractivity(loss, cols) * torch.exp(torch.tensor(10 * SIGMA))
    n_obs[:, 2] = float("nan")
    pred, target = _make_pred_and_target(di, norm, cols, n_obs)
    value = loss(pred, target, target_layout="data_full")
    torch.testing.assert_close(value, torch.tensor(2 * k * 10 - k * k), rtol=2e-3, atol=1e-3)


def test_squash_false_returns_model_output_width_vector() -> None:
    di = _indices()
    norm = _normalizer(di)
    loss = _loss(di, norm)
    cols = _physical_column(grid=3)
    n_obs = _true_refractivity(loss, cols) * 1.003
    pred, target = _make_pred_and_target(di, norm, cols, n_obs)
    vec = loss(pred, target, target_layout="data_full", squash=False)
    total = loss(pred, target, target_layout="data_full")
    assert vec.shape == (pred.shape[-1],)
    torch.testing.assert_close(vec.sum(), total, rtol=1e-5, atol=1e-6)


def test_dry_mode_without_humidity_in_model_output() -> None:
    di = _indices(moist=False)
    norm = _normalizer(di)
    levels = [{"name": "refrac_12000", "height": 12000.0, "sigma": SIGMA}]
    loss = _loss(di, norm, moist=False, levels=levels)
    cols = _physical_column(grid=3, moist=False)
    n_obs = torch.full((3, len(OBS)), float("nan"))
    n_obs[:, 1] = _true_refractivity(loss, cols)[:, 0] * torch.exp(torch.tensor(SIGMA))
    pred, target = _make_pred_and_target(di, norm, cols, n_obs)
    torch.testing.assert_close(loss(pred, target, target_layout="data_full"), torch.tensor(1.0), rtol=1e-3, atol=1e-4)


def test_construction_errors() -> None:
    di = _indices()
    norm = _normalizer(di)
    with pytest.raises(ValueError, match="dry_min_height"):
        _loss(di, norm, moist=False)
    with pytest.raises(ValueError, match="missing from the model output"):
        _loss(_indices(moist=False), _normalizer(_indices(moist=False)), moist=True)
    bad_norm = _normalizer(di)
    bad_norm._norm_mul[di.name_to_index["refrac_5000"]] = 0.1
    with pytest.raises(ValueError, match="physical units"):
        _loss(di, bad_norm)
    with pytest.raises(ValueError, match="normaliser"):
        _loss(di, None)
    with pytest.raises(ValueError, match=r"not in data\.output"):
        _loss(di, norm, levels=[{"name": "refrac_9999", "height": 9999.0, "sigma": SIGMA}])
    with pytest.raises(ValueError, match="strictly decreasing"):
        _loss(di, norm, pressure_levels=[1000, 1000, 300, 100])


def test_factory_builds_inside_combined_loss_and_unit_check_passes() -> None:
    di = _indices()
    norm = _normalizer(di)
    cfg = DictConfig(
        {
            "_target_": "anemoi.training.losses.CombinedLoss",
            "losses": [
                {"_target_": "anemoi.training.losses.MSELoss", "scalers": [], "ignore_nans": True},
                {
                    "_target_": "anemoi.training.losses.RefractivityOperatorLoss",
                    "scalers": [],
                    "pressure_levels": LEVELS,
                    "levels": [{"name": name, "height": h, "sigma": SIGMA} for name, h in OBS],
                },
            ],
            "loss_weights": [1.0, 0.5],
        },
    )
    loss = get_loss_function(cfg, scalers={}, data_indices=di, normalizer=norm)
    assert isinstance(loss, CombinedLoss)
    refrac = loss.losses[1]
    assert isinstance(refrac, RefractivityOperatorLoss)
    assert not hasattr(refrac, "predicted_variables")
    check_loss_tree_variable_units(loss, None)

    cols = _physical_column(grid=3)
    n_obs = _true_refractivity(refrac, cols) * 1.001
    pred, target = _make_pred_and_target(di, norm, cols, n_obs)
    target_out = target[..., di.data.output.full]
    value = loss(pred, target_out, pred_layout=IndexSpace.MODEL_OUTPUT, target_layout=IndexSpace.DATA_OUTPUT)
    assert torch.isfinite(value)


@pytest.mark.parametrize("corruption", ["negative_t", "zero_t", "nan_phi", "inf_t", "huge_q"])
def test_loss_and_gradients_stay_finite_on_unphysical_columns(corruption: str) -> None:
    di = _indices()
    norm = _normalizer(di)
    loss = _loss(di, norm, monotonicity_penalty_weight=1.0)
    cols = _physical_column(grid=4)
    n_obs = _true_refractivity(loss, cols) * 1.01
    if corruption == "negative_t":
        cols["t"][1] = -50.0
    elif corruption == "zero_t":
        cols["t"][1, 1:3] = 0.0
    elif corruption == "nan_phi":
        cols["phi"][2, 2] = float("nan")
    elif corruption == "inf_t":
        cols["t"][0, 2] = float("inf")
    elif corruption == "huge_q":
        cols["q"][3] = 5.0
    pred, target = _make_pred_and_target(di, norm, cols, n_obs)
    pred.requires_grad_(True)
    value = loss(pred, target, target_layout="data_full")
    assert torch.isfinite(value), corruption
    value.backward()
    assert torch.isfinite(pred.grad).all(), corruption
    # Healthy nodes still contribute.
    assert loss.last_level_counts[:2].min() >= 2


def test_non_finite_column_is_masked_not_bracketed() -> None:
    di = _indices()
    norm = _normalizer(di)
    loss = _loss(di, norm)
    cols = _physical_column(grid=3)
    n_obs = _true_refractivity(loss, cols)
    cols["phi"][0] = float("nan")
    pred, target = _make_pred_and_target(di, norm, cols, n_obs)
    loss(pred, target, target_layout="data_full")
    assert loss.last_level_counts.tolist() == [2, 2, 0]
    assert loss.last_disordered_layer_fraction.item() == 0.0
