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
NAMES = [f"{p}_{lvl}" for p in ("z", "t", "q") for lvl in LEVELS] + ["other"]
# Model-output order deliberately differs from ladder order and from the statistics order.
NAME_TO_INDEX = {
    n: i
    for i, n in enumerate(["other", "q_500", "z_500", "t_850", "z_1000", "q_1000", "t_500", "z_850", "t_1000", "q_850"])
}
NAME_TO_INDEX_STATS = {n: i for i, n in enumerate(reversed(NAMES))}
NORMALIZER = {"z": "min-max", "t": "mean-std", "q": "mean-std"}


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
