# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import datetime
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from hydra.utils import instantiate
from pydantic import ValidationError

from anemoi.models.data.layout import TensorLayout
from anemoi.models.data.sources import GriddedSource
from anemoi.models.layers.bounding import ForcingMaskBounding
from anemoi.models.layers.bounding import ReluBounding
from anemoi.models.layers.bounding import apply_boundings
from anemoi.models.layers.forcing_functions import ForcingCondition
from anemoi.models.layers.forcing_functions import NoSunlightInWindow
from anemoi.models.layers.forcing_functions import insolation
from anemoi.models.models.encoder_processor_decoder import AnemoiModelEncProcDec
from anemoi.models.schemas.bounding import ForcingMaskBoundingSchema

LAYOUT = TensorLayout(batch=0, time=1, ensemble=2, grid=3, variables=4)
TIME_FORCINGS = ["cos_julian_day", "sin_julian_day", "cos_local_time", "sin_local_time"]


def _source(values: torch.Tensor, variables: list[str], statistics=None, lat_deg=None) -> GriddedSource:
    grid = values.shape[3]
    lat = torch.zeros(grid, dtype=torch.float64) if lat_deg is None else torch.deg2rad(torch.as_tensor(lat_deg))
    return GriddedSource(
        name="data",
        values=values,
        variables=variables,
        statistics=statistics or {},
        coordinates=torch.stack([lat, torch.zeros_like(lat)], dim=-1),
        layout=LAYOUT,
    )


def _pred(value: float = 7.0, statistics=None) -> GriddedSource:
    return _source(torch.full((1, 1, 1, 4, 2), value), ["ssr", "t2m"], statistics)


def _time_forcings(date: datetime.datetime, lat_deg, lon_deg) -> GriddedSource:
    """The four time forcings at ``date``, defined as in earthkit-data."""
    lon = np.asarray(lon_deg, dtype=np.float64)
    day = 2 * np.pi * ((date - datetime.datetime(date.year, 1, 1)).total_seconds() / 86400) / 365.25
    local = 2 * np.pi * ((lon / 360 * 24 + date.hour) % 24) / 24
    columns = [np.full_like(lon, np.cos(day)), np.full_like(lon, np.sin(day)), np.cos(local), np.sin(local)]
    values = torch.from_numpy(np.stack(columns, axis=-1)).reshape(1, 1, 1, lon.size, 4)
    return _source(values, TIME_FORCINGS, lat_deg=np.broadcast_to(lat_deg, lon.shape).copy())


def test_mask_from_a_forcing_in_physical_units() -> None:
    # the forcing arrives normalised (mean 0.5, stdev 0.25) and ssr is normalised with mean 2, stdev 4
    physical = torch.tensor([0.0, 0.1, 0.3, 1.0])
    forcings = _source(
        ((physical - 0.5) / 0.25).reshape(1, 1, 1, 4, 1), ["lsm"], {"mean": np.array([0.5]), "stdev": np.array([0.25])}
    )
    mask = ForcingCondition("lsm", operator="lt", threshold=0.2, normalizer="mean-std")
    bounding = ForcingMaskBounding(variables=["ssr"], normalizer=["mean-std"], mask=mask, fill_value=10.0)
    pred = _pred(3.0, {"mean": np.array([2.0, 0.0]), "stdev": np.array([4.0, 1.0])})

    out = bounding(pred, target_forcings=forcings).values

    assert out[0, 0, 0, :, 0].tolist() == pytest.approx([2.0, 2.0, 3.0, 3.0])  # (10 - 2) / 4 where lsm < 0.2
    assert torch.all(out[..., 1] == 3.0)  # t2m untouched
    assert torch.all(pred.values == 3.0)  # the input is not modified


def test_no_sunlight_in_window() -> None:
    assert NoSunlightInWindow("1h").shifts == [0.0, -1.0]  # the two ends of a one-hour window
    assert len(NoSunlightInWindow("6h").shifts) == 7
    with pytest.raises(ValueError, match="whole number of hours"):
        NoSunlightInWindow("90m")

    # At 65N near the winter solstice the day is shorter than 6h: it can fall between the ends of a window.
    forcings = _time_forcings(datetime.datetime(2016, 12, 21, 12), 65.0, np.arange(-180.0, 180.0, 1.0))
    cj, sj, cl, sl = (forcings.values[..., i] for i in range(4))
    lat = forcings.coordinates[:, 0]
    ends_dark = (insolation(cj, sj, cl, sl, lat) == 0) & (insolation(cj, sj, cl, sl, lat, -6.0) == 0)
    dark = NoSunlightInWindow("6h")(forcings)

    assert bool((ends_dark & ~dark).any())  # sunlit only inside the window: the two ends would miss it
    assert not bool((dark & ~ends_dark).any())  # and never the other way round


def test_insolation_matches_earthkit() -> None:
    ekd = pytest.importorskip("earthkit.data")
    rng = np.random.default_rng(0)
    lat, lon = rng.uniform(-90, 90, 2000), rng.uniform(-180, 180, 2000)
    t = datetime.datetime(2016, 9, 30, 13)
    forcings = _time_forcings(t, lat, lon)
    cj, sj, cl, sl = (forcings.values[..., i] for i in range(4))

    for shift in (0, -1, -6):
        date = t + datetime.timedelta(hours=shift)
        expected = ekd.from_source("forcings", latitudes=lat, longitudes=lon, date=[date], param=["insolation"])
        computed = insolation(cj, sj, cl, sl, forcings.coordinates[:, 0], float(shift))[0, 0, 0].numpy()
        np.testing.assert_allclose(computed, expected[0].to_numpy(dtype=np.float64, flatten=True), atol=1e-12)


def test_config_and_schema() -> None:
    config = {
        "_target_": "anemoi.models.layers.bounding.ForcingMaskBounding",
        "variables": ["ssr"],
        "normalizer": ["none"],
        "mask": {"_target_": "anemoi.models.layers.forcing_functions.NoSunlightInWindow", "window": "6h"},
    }
    # the extra arguments are those injected by the model when building boundings
    bounding = instantiate(config, name_to_index={"ssr": 0}, statistics=None, name_to_index_stats={})
    assert isinstance(bounding.mask, NoSunlightInWindow)

    assert ForcingMaskBoundingSchema.model_validate(config).fill_value == 0.0
    with pytest.raises(ValidationError):
        ForcingMaskBoundingSchema.model_validate(config | {"normalizer": ["none", "none"]})
    with pytest.raises(ValueError, match="Unknown operator"):
        ForcingCondition("lsm", operator="eq")
    with pytest.raises(ValueError, match="target_forcings"):
        bounding(_pred())  # the forcings are required


def test_model_applies_boundings_with_the_target_forcings() -> None:
    relu = ReluBounding(variables=["t2m"])
    mask = ForcingMaskBounding(variables=["ssr"], normalizer=["none"], mask=ForcingCondition("insolation"))
    template = SimpleNamespace(unflatten=lambda x: _source(x, ["ssr", "t2m"]))
    forcings = _source(torch.tensor([0.0, 0.5]).reshape(1, 1, 1, 2, 1), ["insolation"])
    x_out = torch.tensor([[5.0, -1.0], [5.0, 2.0]]).reshape(1, 1, 1, 2, 2)

    for boundings in (torch.nn.Sequential(relu, mask), torch.nn.ModuleList([relu, mask])):
        model = SimpleNamespace(boundings={"data": boundings}, _add_residual=lambda pred, skip, name: pred)
        out = AnemoiModelEncProcDec._assemble_output(
            model, x_out, None, template, x_out.dtype, "data", target_forcings=forcings
        ).values
        assert out[0, 0, 0, :, 0].tolist() == [0.0, 5.0]  # ssr zeroed where insolation <= 0
        assert out[0, 0, 0, :, 1].tolist() == [0.0, 2.0]  # t2m clipped by the ReLU

    # a dataset without boundings, or with a single module, still works
    assert torch.equal(apply_boundings(torch.nn.Sequential(), _pred()).values, _pred().values)
    assert torch.equal(apply_boundings(torch.nn.Identity(), _pred()).values, _pred().values)
