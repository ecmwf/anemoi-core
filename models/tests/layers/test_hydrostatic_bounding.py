# (C) Copyright 2024- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import math

import numpy as np
import pytest
import torch
from hydra.utils import instantiate

from anemoi.models.layers.bounding import HydrostaticGeopotential
from anemoi.models.physics.constants import G0
from anemoi.models.physics.constants import R_D
from anemoi.models.physics.constants import VIRTUAL_TEMP_COEFF

LEVELS = [1000, 850, 500]
NAMES = [f"{p}_{lvl}" for p in ("z", "t", "q") for lvl in LEVELS] + ["other", "msl"]
# Model-output order deliberately differs from ladder order and from the statistics order.
NAME_TO_INDEX = {
    n: i
    for i, n in enumerate(
        ["other", "q_500", "z_500", "t_850", "z_1000", "msl", "q_1000", "t_500", "z_850", "t_1000", "q_850"]
    )
}
NAME_TO_INDEX_STATS = {n: i for i, n in enumerate(reversed(NAMES))}
NORMALIZER = {"z": "min-max", "t": "mean-std", "q": "mean-std"}
NORMALIZER_MSL = NORMALIZER | {"msl": "mean-std"}


@pytest.fixture
def statistics() -> dict:
    n = len(NAMES)
    rng = np.random.default_rng(0)
    mean = np.zeros(n)
    stdev = np.ones(n)
    minimum = np.zeros(n)
    maximum = np.ones(n)
    for name, i in NAME_TO_INDEX_STATS.items():
        if name.startswith("z_"):
            lvl = int(name.split("_")[1])
            centre = {1000: 1000.0, 850: 14000.0, 500: 55000.0}[lvl]
            minimum[i], maximum[i] = centre - 8000.0, centre + 8000.0
            mean[i], stdev[i] = centre, 2000.0
        elif name.startswith("t_"):
            mean[i], stdev[i] = 260.0, 10.0
        elif name.startswith("q_"):
            mean[i], stdev[i] = 3e-3, 2e-3
        elif name == "msl":
            mean[i], stdev[i] = 101300.0, 1000.0
        else:
            mean[i], stdev[i] = rng.normal(), 1.0
    return {"mean": mean, "stdev": stdev, "minimum": minimum, "maximum": maximum}


def _layer(statistics: dict, **kwargs) -> HydrostaticGeopotential:
    defaults = {
        "levels": LEVELS,
        "name_to_index": NAME_TO_INDEX,
        "statistics": statistics,
        "name_to_index_stats": NAME_TO_INDEX_STATS,
        "normalizer": NORMALIZER,
    }
    defaults.update(kwargs)
    return HydrostaticGeopotential(**defaults)


def _affine(statistics: dict, name: str) -> tuple[float, float]:
    i = NAME_TO_INDEX_STATS[name]
    if name.startswith("z_"):
        span = statistics["maximum"][i] - statistics["minimum"][i]
        return 1.0 / span, -statistics["minimum"][i] / span
    return 1.0 / statistics["stdev"][i], -statistics["mean"][i] / statistics["stdev"][i]


def _make_x(statistics: dict, phys: dict[str, torch.Tensor], shape: tuple[int, ...]) -> torch.Tensor:
    """Normalised tensor of the given leading shape with physical values per variable name."""
    x = torch.zeros(*shape, len(NAME_TO_INDEX))
    for name, idx in NAME_TO_INDEX.items():
        if name in phys:
            mul, add = _affine(statistics, name)
            x[..., idx] = phys[name] * mul + add
        else:
            x[..., idx] = 0.3
    return x


def _read(layer: HydrostaticGeopotential, x: torch.Tensor, name: str, statistics: dict) -> torch.Tensor:
    mul, add = _affine(statistics, name)
    return (x[..., NAME_TO_INDEX[name]] - add) / mul


def test_isothermal_dry_column_matches_closed_form(statistics: dict) -> None:
    layer = _layer(statistics)
    t = 250.0
    phys = {f"t_{lvl}": torch.tensor(t) for lvl in LEVELS} | {f"q_{lvl}": torch.tensor(0.0) for lvl in LEVELS}
    phys["z_1000"] = torch.tensor(500.0)
    phys["z_850"] = torch.tensor(-1.0)  # garbage in the derived heads must be overwritten
    phys["z_500"] = torch.tensor(-1.0)
    x = _make_x(statistics, phys, (2, 3))
    out = layer(x.clone())
    expected_850 = 500.0 + R_D * t * math.log(1000 / 850)
    expected_500 = expected_850 + R_D * t * math.log(850 / 500)
    torch.testing.assert_close(
        _read(layer, out, "z_850", statistics), torch.full((2, 3), expected_850), rtol=1e-5, atol=0.5
    )
    torch.testing.assert_close(
        _read(layer, out, "z_500", statistics), torch.full((2, 3), expected_500), rtol=1e-5, atol=0.5
    )
    # Anchor and unrelated channels untouched.
    torch.testing.assert_close(out[..., NAME_TO_INDEX["z_1000"]], x[..., NAME_TO_INDEX["z_1000"]])
    torch.testing.assert_close(out[..., NAME_TO_INDEX["other"]], x[..., NAME_TO_INDEX["other"]])


def test_virtual_temperature_thickens_the_layer(statistics: dict) -> None:
    layer = _layer(statistics)
    t = 280.0
    dry = (
        {f"t_{lvl}": torch.tensor(t) for lvl in LEVELS}
        | {f"q_{lvl}": torch.tensor(0.0) for lvl in LEVELS}
        | {"z_1000": torch.tensor(0.0)}
    )
    moist = dict(dry) | {"q_1000": torch.tensor(0.01), "q_850": torch.tensor(0.01)}
    z_dry = _read(layer, layer(_make_x(statistics, dry, (1,))), "z_850", statistics)
    z_moist = _read(layer, layer(_make_x(statistics, moist, (1,))), "z_850", statistics)
    torch.testing.assert_close(z_moist / z_dry, torch.tensor([1.0 + VIRTUAL_TEMP_COEFF * 0.01]), rtol=1e-5, atol=1e-6)


def test_consistent_column_is_a_fixed_point(statistics: dict) -> None:
    layer = _layer(statistics)
    t = {1000: 285.0, 850: 275.0, 500: 255.0}
    q = {1000: 8e-3, 850: 5e-3, 500: 1e-3}
    phys = {f"t_{lvl}": torch.tensor(v) for lvl, v in t.items()} | {f"q_{lvl}": torch.tensor(v) for lvl, v in q.items()}
    phi = 800.0
    phys["z_1000"] = torch.tensor(phi)
    for lo, hi in zip(LEVELS[:-1], LEVELS[1:]):
        tv = 0.5 * (t[lo] * (1 + VIRTUAL_TEMP_COEFF * q[lo]) + t[hi] * (1 + VIRTUAL_TEMP_COEFF * q[hi]))
        phi += R_D * tv * math.log(lo / hi)
        phys[f"z_{hi}"] = torch.tensor(phi)
    x = _make_x(statistics, phys, (4,))
    torch.testing.assert_close(layer(x.clone()), x, rtol=0, atol=1e-4)


def test_gradients_reach_t_q_and_anchor_but_not_derived_heads(statistics: dict) -> None:
    layer = _layer(statistics)
    leaf = torch.randn(1, 1, 1, 5, len(NAME_TO_INDEX), requires_grad=True)
    out = layer(leaf.clone())  # mimics the non-leaf clone in _assemble_output
    out[..., NAME_TO_INDEX["z_500"]].sum().backward()
    g = leaf.grad[0, 0, 0]
    for name in ("t_1000", "t_850", "t_500", "q_1000", "q_850", "q_500", "z_1000"):
        assert g[:, NAME_TO_INDEX[name]].abs().sum() > 0, name
    for name in ("z_850", "z_500", "other"):
        assert g[:, NAME_TO_INDEX[name]].abs().sum() == 0, name
    # Anchor leverage: d Phi_i / d Phi_0 = 1 for every derived level.
    mul_anchor, _ = _affine(statistics, "z_1000")
    mul_500, _ = _affine(statistics, "z_500")
    torch.testing.assert_close(
        g[:, NAME_TO_INDEX["z_1000"]], torch.full((5,), mul_500 / mul_anchor), rtol=1e-4, atol=1e-6
    )


def test_gradcheck_float64(statistics: dict) -> None:
    layer = _layer(statistics).double()
    x = torch.randn(3, len(NAME_TO_INDEX), dtype=torch.float64)
    for name in ("t_1000", "t_850", "t_500", "q_1000", "q_850", "q_500", "z_1000"):
        idx = NAME_TO_INDEX[name]

        def fn(v: torch.Tensor, idx: int = idx) -> torch.Tensor:
            y = x.clone()
            y[:, idx] = v
            return layer.integrate(y)[..., 1:]

        assert torch.autograd.gradcheck(
            fn, (x[:, idx].clone().requires_grad_(True),), eps=1e-6, atol=1e-6, rtol=1e-4
        ), name


def test_shape_and_dtype_preserved(statistics: dict) -> None:
    layer = _layer(statistics)
    for shape in ((2, 1, 3, 40), (7,)):
        x = torch.randn(*shape, len(NAME_TO_INDEX), dtype=torch.bfloat16)
        out = layer(x.clone())
        assert out.shape == x.shape and out.dtype == x.dtype
        assert torch.isfinite(out).all()


def test_geopotential_height_units(statistics: dict) -> None:
    layer = _layer(statistics, geopotential_units="m")
    t = 250.0
    phys = (
        {f"t_{lvl}": torch.tensor(t) for lvl in LEVELS}
        | {f"q_{lvl}": torch.tensor(0.0) for lvl in LEVELS}
        | {"z_1000": torch.tensor(100.0)}
    )
    out = layer(_make_x(statistics, phys, (1,)))
    expected = 100.0 + R_D * t * math.log(1000 / 850) / G0
    torch.testing.assert_close(_read(layer, out, "z_850", statistics), torch.tensor([expected]), rtol=1e-5, atol=0.05)


def test_strict_contract(statistics: dict) -> None:
    for missing in ("z_850", "t_850", "q_850"):
        n2i = {k: v for k, v in NAME_TO_INDEX.items() if k != missing}
        with pytest.raises(ValueError, match=missing):
            _layer(statistics, name_to_index=n2i)
    with pytest.raises(ValueError, match="strictly decreasing"):
        _layer(statistics, levels=[1000, 1000, 500])
    with pytest.raises(ValueError, match="normalizer"):
        _layer(statistics, normalizer={"z": "min-max", "t": "mean-std"})
    with pytest.raises(ValueError, match="geopotential_units"):
        _layer(statistics, geopotential_units="km")
    layer = _layer(statistics)
    assert layer.normalizer_methods["z_500"] == "min-max" and layer.normalizer_methods["q_850"] == "mean-std"
    assert layer.variables == ["z_850", "z_500"]


def test_hydra_instantiate(statistics: dict) -> None:
    cfg = {
        "_target_": "anemoi.models.layers.bounding.HydrostaticGeopotential",
        "levels": LEVELS,
        "normalizer": NORMALIZER,
        "check_finite": True,
    }
    layer = instantiate(
        cfg, name_to_index=NAME_TO_INDEX, statistics=statistics, name_to_index_stats=NAME_TO_INDEX_STATS
    )
    out = layer(torch.randn(2, len(NAME_TO_INDEX)))
    assert torch.isfinite(out).all()
    x = torch.randn(2, len(NAME_TO_INDEX))
    x[:, NAME_TO_INDEX["t_850"]] = float("nan")
    with pytest.raises(RuntimeError, match="non-finite"):
        layer(x)


def _isothermal_dry(t: float) -> dict[str, torch.Tensor]:
    return {f"t_{lvl}": torch.tensor(t) for lvl in LEVELS} | {f"q_{lvl}": torch.tensor(0.0) for lvl in LEVELS}


def test_msl_anchor_matches_closed_form(statistics: dict) -> None:
    layer = _layer(statistics, msl_anchor="msl", normalizer=NORMALIZER_MSL)
    t = 280.0
    phys = _isothermal_dry(t) | {"msl": torch.tensor([101300.0, 99000.0])}
    phys |= {f"z_{lvl}": torch.tensor(-1.0) for lvl in LEVELS}  # every z head is overwritten, anchor included
    x = _make_x(statistics, phys, (2,))
    out = layer(x.clone())
    expected_1000 = R_D * t * torch.log(phys["msl"].double() / 100000.0)
    expected_850 = expected_1000 + R_D * t * math.log(1000 / 850)
    torch.testing.assert_close(_read(layer, out, "z_1000", statistics).double(), expected_1000, rtol=1e-5, atol=0.5)
    torch.testing.assert_close(_read(layer, out, "z_850", statistics).double(), expected_850, rtol=1e-5, atol=0.5)
    # msl below 1000 hPa puts the 1000 hPa surface below sea level.
    assert _read(layer, out, "z_1000", statistics)[1] < 0
    torch.testing.assert_close(out[..., NAME_TO_INDEX["msl"]], x[..., NAME_TO_INDEX["msl"]])
    assert layer.variables == ["z_1000", "z_850", "z_500"]
    assert layer.normalizer_methods["msl"] == "mean-std"


def test_msl_anchor_hpa_units(statistics: dict) -> None:
    stats_hpa = {k: v.copy() for k, v in statistics.items()}
    i = NAME_TO_INDEX_STATS["msl"]
    stats_hpa["mean"][i], stats_hpa["stdev"][i] = 1013.0, 10.0
    layer_pa = _layer(statistics, msl_anchor="msl", normalizer=NORMALIZER_MSL)
    layer_hpa = _layer(stats_hpa, msl_anchor="msl", normalizer=NORMALIZER_MSL, msl_units="hPa")
    phys = _isothermal_dry(270.0)
    out_pa = layer_pa(_make_x(statistics, phys | {"msl": torch.tensor(102000.0)}, (1,)))
    out_hpa = layer_hpa(_make_x(stats_hpa, phys | {"msl": torch.tensor(1020.0)}, (1,)))
    torch.testing.assert_close(
        out_pa[..., layer_pa.data_index], out_hpa[..., layer_hpa.data_index], rtol=1e-5, atol=1e-5
    )


def test_msl_anchor_gradients_bypass_the_z_heads(statistics: dict) -> None:
    layer = _layer(statistics, msl_anchor="msl", normalizer=NORMALIZER_MSL)
    leaf = torch.randn(1, 5, len(NAME_TO_INDEX), requires_grad=True)
    layer(leaf.clone())[..., NAME_TO_INDEX["z_500"]].sum().backward()
    g = leaf.grad[0]
    for name in ("msl", "t_1000", "q_1000", "t_500"):
        assert g[:, NAME_TO_INDEX[name]].abs().sum() > 0, name
    for name in ("z_1000", "z_850", "z_500", "other"):
        assert g[:, NAME_TO_INDEX[name]].abs().sum() == 0, name


def test_msl_anchor_gradcheck_float64(statistics: dict) -> None:
    layer = _layer(statistics, msl_anchor="msl", normalizer=NORMALIZER_MSL).double()
    x = torch.randn(3, len(NAME_TO_INDEX), dtype=torch.float64)
    for name in ("msl", "t_1000", "q_1000", "t_850"):
        idx = NAME_TO_INDEX[name]

        def fn(v: torch.Tensor, idx: int = idx) -> torch.Tensor:
            y = x.clone()
            y[:, idx] = v
            return layer.integrate(y)

        assert torch.autograd.gradcheck(
            fn, (x[:, idx].clone().requires_grad_(True),), eps=1e-6, atol=1e-6, rtol=1e-4
        ), name


def test_msl_anchor_contract(statistics: dict) -> None:
    with pytest.raises(ValueError, match="msl"):
        _layer(statistics, msl_anchor="msl")  # normaliser method for msl missing
    with pytest.raises(ValueError, match="msl"):
        _layer(statistics, msl_anchor="msl", normalizer=NORMALIZER_MSL, msl_units="bar")
    n2i = {k: v for k, v in NAME_TO_INDEX.items() if k != "msl"}
    with pytest.raises(ValueError, match="msl"):
        _layer(statistics, msl_anchor="msl", normalizer=NORMALIZER_MSL, name_to_index=n2i)


# ----------------------------------------------------------------------------- sp anchor
# The orography forcing is deliberately named "z", like the geopotential prefix (the latw configs
# rename synop_z to z), so the separate orography_normalizer is exercised.
SP_NAME_TO_INDEX = NAME_TO_INDEX | {"sp": len(NAME_TO_INDEX)}
SP_STATS_NAMES = NAMES + ["sp", "z", "lsm"]
SP_NAME_TO_INDEX_STATS = {n: i for i, n in enumerate(reversed(SP_STATS_NAMES))}
DECODER_FORCING_NAMES = ["lsm", "z"]  # tensor order; orography is the second channel
NORMALIZER_SP = NORMALIZER | {"sp": "mean-std"}
OROG_MAX = 6000.0


@pytest.fixture
def sp_statistics(statistics: dict) -> dict:
    n = len(SP_STATS_NAMES)
    out = {k: np.zeros(n) for k in statistics}
    out["stdev"][:] = 1.0
    out["maximum"][:] = 1.0
    for name, i in SP_NAME_TO_INDEX_STATS.items():
        if name in NAME_TO_INDEX_STATS:
            for k in out:
                out[k][i] = statistics[k][NAME_TO_INDEX_STATS[name]]
        elif name == "sp":
            out["mean"][i], out["stdev"][i] = 96000.0, 9000.0
        elif name == "z":
            out["maximum"][i] = OROG_MAX
    return out


def _sp_layer(sp_statistics: dict, **kwargs) -> HydrostaticGeopotential:
    defaults = {
        "levels": LEVELS,
        "name_to_index": SP_NAME_TO_INDEX,
        "statistics": sp_statistics,
        "name_to_index_stats": SP_NAME_TO_INDEX_STATS,
        "normalizer": NORMALIZER_SP,
        "sp_anchor": "sp",
        "orography": "z",
        "orography_normalizer": "max",
        "decoder_forcing_names": DECODER_FORCING_NAMES,
    }
    defaults.update(kwargs)
    return HydrostaticGeopotential(**defaults)


def _sp_affine(stats: dict, name: str) -> tuple[float, float]:
    i = SP_NAME_TO_INDEX_STATS[name]
    if name.startswith("z_"):
        span = stats["maximum"][i] - stats["minimum"][i]
        return 1.0 / span, -stats["minimum"][i] / span
    return 1.0 / stats["stdev"][i], -stats["mean"][i] / stats["stdev"][i]


def _sp_inputs(stats: dict, phys: dict, orog_m: torch.Tensor, shape: tuple) -> tuple[torch.Tensor, torch.Tensor]:
    x = torch.full((*shape, len(SP_NAME_TO_INDEX)), 0.3, dtype=torch.float64)
    for name, idx in SP_NAME_TO_INDEX.items():
        if name in phys:
            mul, add = _sp_affine(stats, name)
            x[..., idx] = phys[name] * mul + add
    df = torch.full((*shape, len(DECODER_FORCING_NAMES)), 0.7, dtype=torch.float64)
    df[..., DECODER_FORCING_NAMES.index("z")] = orog_m / OROG_MAX
    return x.float(), df.float()


def _sp_read(stats: dict, x: torch.Tensor, name: str) -> torch.Tensor:
    mul, add = _sp_affine(stats, name)
    return (x[..., SP_NAME_TO_INDEX[name]].double() - add) / mul


def test_sp_anchor_isothermal_column_matches_closed_form(sp_statistics: dict) -> None:
    # Isothermal dry column: Phi(p) = Phi_s + R_d T ln(sp / p) at every level, above or below ground.
    layer = _sp_layer(sp_statistics)
    t = 250.0
    sp = torch.tensor([101000.0, 92000.0, 70000.0, 40000.0], dtype=torch.float64)  # k = 0, 1, 2, none
    orog = torch.tensor([0.0, 800.0, 3000.0, 7000.0], dtype=torch.float64)
    phys = _isothermal_dry(t) | {"sp": sp} | {f"z_{lvl}": torch.tensor(-1.0) for lvl in LEVELS}
    x, df = _sp_inputs(sp_statistics, phys, orog, (4,))
    out = layer(x.clone(), decoder_forcing=df)
    for lvl in LEVELS:
        expected = orog * G0 + R_D * t * torch.log(sp / (100.0 * lvl))
        torch.testing.assert_close(_sp_read(sp_statistics, out, f"z_{lvl}"), expected, rtol=1e-5, atol=1.0)
    assert layer.variables == ["z_1000", "z_850", "z_500"]
    torch.testing.assert_close(out[..., SP_NAME_TO_INDEX["sp"]], x[..., SP_NAME_TO_INDEX["sp"]])


def test_sp_anchor_exact_for_tv_linear_in_log_p(sp_statistics: dict) -> None:
    # Tv = a + b ln p is integrated exactly by the trapezoid and extrapolated exactly to sp.
    layer = _sp_layer(sp_statistics)
    a, b = -120.0, 34.0  # Tv(1000 hPa) ~ 271 K, Tv(500 hPa) ~ 248 K
    phys = {f"t_{lvl}": torch.tensor(a + b * math.log(100.0 * lvl)) for lvl in LEVELS}
    phys |= {f"q_{lvl}": torch.tensor(0.0) for lvl in LEVELS}
    # First above-ground level 1000 and 850 hPa; both have a level above to extrapolate from.
    sp = torch.tensor([99500.0, 90000.0], dtype=torch.float64)
    orog = torch.tensor([120.0, 1000.0], dtype=torch.float64)
    x, df = _sp_inputs(sp_statistics, phys | {"sp": sp}, orog, (2,))
    out = layer(x.clone(), decoder_forcing=df)
    for lvl in LEVELS:
        lp, ls = math.log(100.0 * lvl), torch.log(sp)
        expected = orog * G0 + R_D * (a * (ls - lp) + 0.5 * b * (ls**2 - lp**2))
        torch.testing.assert_close(_sp_read(sp_statistics, out, f"z_{lvl}"), expected, rtol=1e-5, atol=1.0)


def test_sp_anchor_gradients(sp_statistics: dict) -> None:
    layer = _sp_layer(sp_statistics)
    leaf = torch.randn(1, 6, len(SP_NAME_TO_INDEX), requires_grad=True)
    _, df = _sp_inputs(sp_statistics, {}, torch.full((1, 6), 500.0, dtype=torch.float64), (1, 6))
    layer(leaf.clone(), decoder_forcing=df)[..., SP_NAME_TO_INDEX["z_500"]].sum().backward()
    g = leaf.grad[0]
    for name in ("sp", "t_1000", "t_850", "t_500", "q_500"):
        assert g[:, SP_NAME_TO_INDEX[name]].abs().sum() > 0, name
    for name in ("z_1000", "z_850", "z_500", "msl", "other"):
        assert g[:, SP_NAME_TO_INDEX[name]].abs().sum() == 0, name


def test_sp_anchor_gradcheck_float64(sp_statistics: dict) -> None:
    layer = _sp_layer(sp_statistics).double()
    phys = {f"t_{lvl}": torch.tensor(270.0 - 0.02 * (1000 - lvl)) for lvl in LEVELS}
    phys |= {f"q_{lvl}": torch.tensor(2e-3) for lvl in LEVELS}
    sp = torch.tensor([99000.0, 80000.0, 60000.0], dtype=torch.float64)  # k = 0, 1, 2
    x, df = _sp_inputs(sp_statistics, phys | {"sp": sp}, torch.tensor([50.0, 1900.0, 4000.0]), (3,))
    x, df = x.double(), df.double()
    for name in ("sp", "t_1000", "t_850", "t_500", "q_850"):
        idx = SP_NAME_TO_INDEX[name]

        def fn(v: torch.Tensor, idx: int = idx) -> torch.Tensor:
            y = x.clone()
            y[:, idx] = v
            return layer.integrate(y, df)

        assert torch.autograd.gradcheck(
            fn, (x[:, idx].clone().requires_grad_(True),), eps=1e-6, atol=1e-3, rtol=1e-4
        ), name


def test_sp_anchor_contract(sp_statistics: dict) -> None:
    layer = _sp_layer(sp_statistics)
    assert layer.requires_decoder_forcing
    assert layer.normalizer_methods["z"] == "max" and layer.normalizer_methods["sp"] == "mean-std"
    assert layer.normalizer_methods["z_500"] == "min-max"
    with pytest.raises(RuntimeError, match="decoder"):
        layer(torch.randn(2, len(SP_NAME_TO_INDEX)))
    with pytest.raises(ValueError, match="decoder forcing"):
        _sp_layer(sp_statistics, decoder_forcing_names=["lsm"])
    with pytest.raises(ValueError, match="exclusive"):
        _sp_layer(sp_statistics, msl_anchor="msl", normalizer=NORMALIZER_SP | {"msl": "mean-std"})
    with pytest.raises(ValueError, match="orography"):
        _sp_layer(sp_statistics, orography_normalizer=None)
    with pytest.raises(ValueError, match="sp"):
        _sp_layer(sp_statistics, normalizer=NORMALIZER)
    assert not _layer(statistics=_plain_stats(sp_statistics)).requires_decoder_forcing


def _plain_stats(sp_statistics: dict) -> dict:
    """Map the sp fixture back onto the plain statistics index."""
    out = {k: np.zeros(len(NAMES)) for k in sp_statistics}
    for name, i in NAME_TO_INDEX_STATS.items():
        for k in out:
            out[k][i] = sp_statistics[k][SP_NAME_TO_INDEX_STATS[name]]
    return out


def test_builder_injects_decoder_forcing_names(sp_statistics: dict) -> None:
    from types import SimpleNamespace

    from anemoi.models.layers.bounding import _build_dataset_boundings

    model_config = SimpleNamespace(
        model=SimpleNamespace(
            bounding=[
                {"_target_": "anemoi.models.layers.bounding.ReluBounding", "variables": ["q_500"]},
                {
                    "_target_": "anemoi.models.layers.bounding.HydrostaticGeopotential",
                    "levels": LEVELS,
                    "normalizer": NORMALIZER_SP,
                    "sp_anchor": "sp",
                    "orography": "z",
                    "orography_normalizer": "max",
                },
            ]
        )
    )
    # decoder_forcing holds data-input indices; channels follow sorted index order.
    df_idx = torch.tensor(sorted(SP_NAME_TO_INDEX_STATS[n] for n in ("z", "lsm")), dtype=torch.int)
    data_indices = SimpleNamespace(
        model=SimpleNamespace(output=SimpleNamespace(name_to_index=SP_NAME_TO_INDEX)),
        data=SimpleNamespace(input=SimpleNamespace(name_to_index=SP_NAME_TO_INDEX_STATS, decoder_forcing=df_idx)),
    )
    relu, hydro = _build_dataset_boundings(model_config, data_indices, sp_statistics)
    expected_order = sorted(("z", "lsm"), key=SP_NAME_TO_INDEX_STATS.get)
    assert hydro.orography_position == expected_order.index("z")
    assert not getattr(relu, "requires_decoder_forcing", False)


def test_enc_proc_dec_passes_decoder_forcing_to_boundings(sp_statistics: dict) -> None:
    from types import SimpleNamespace

    from anemoi.models.models.encoder_processor_decoder import AnemoiModelEncProcDec

    layer = _sp_layer(sp_statistics)
    batch, time, ens, grid = 1, 1, 1, 4
    phys = _isothermal_dry(250.0) | {"sp": torch.full((batch, time, ens, grid), 85000.0, dtype=torch.float64)}
    x, df = _sp_inputs(sp_statistics, phys, torch.full((batch, time, ens, grid), 1500.0), (batch, time, ens, grid))
    n_vars = x.shape[-1]
    fake = SimpleNamespace(
        n_step_output=time,
        boundings={"data": [layer]},
        _internal_output_idx={"data": []},
        _internal_input_idx={"data": []},
    )
    x_flat = x.reshape(batch * ens * grid, time * n_vars)
    x_skip = torch.zeros(batch, time, ens, grid, 1)
    out = AnemoiModelEncProcDec._assemble_output(
        fake, x_flat, x_skip, batch, ens, torch.float32, "data", decoder_forcing=df
    )
    expected = 1500.0 * G0 + R_D * 250.0 * math.log(85000.0 / 50000.0)
    torch.testing.assert_close(
        _sp_read(sp_statistics, out, "z_500"),
        torch.full((batch, time, ens, grid), expected, dtype=torch.float64),
        rtol=1e-5,
        atol=1.0,
    )
    with pytest.raises(RuntimeError, match="decoder"):
        AnemoiModelEncProcDec._assemble_output(fake, x_flat, x_skip, batch, ens, torch.float32, "data")


def test_humidity_levels_integrates_dry_above(statistics: dict) -> None:
    # q only at 1000/850: the 850-500 layer uses T at 500 with q = 0 there, and q_500 is never read.
    n2i = {k: v for k, v in NAME_TO_INDEX.items() if k != "q_500"}
    layer = _layer(statistics, humidity_levels=[1000, 850], name_to_index=n2i)
    full = _layer(statistics)
    t = {1000: 285.0, 850: 275.0, 500: 255.0}
    phys = {f"t_{lvl}": torch.tensor(v) for lvl, v in t.items()}
    phys |= {"q_1000": torch.tensor(8e-3), "q_850": torch.tensor(5e-3), "q_500": torch.tensor(0.0)}
    phys["z_1000"] = torch.tensor(300.0)
    x = _make_x(statistics, phys, (3,))
    torch.testing.assert_close(layer(x.clone()), full(x.clone()), rtol=0, atol=1e-6)
    x[..., NAME_TO_INDEX["q_500"]] = 5.0  # ignored
    torch.testing.assert_close(
        layer(x.clone())[..., layer.data_index], full(_make_x(statistics, phys, (3,)))[..., full.data_index]
    )
    assert "q_500" not in layer.normalizer_methods
    with pytest.raises(ValueError, match="humidity_levels"):
        _layer(statistics, humidity_levels=[1000, 700])
    with pytest.raises(ValueError, match="humidity_levels"):
        _layer(statistics, humidity_levels=[])
    assert "q_position" not in layer.state_dict()
