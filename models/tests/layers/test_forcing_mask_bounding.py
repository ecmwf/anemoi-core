# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import datetime
import math
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
from anemoi.models.layers.forcing_functions import MaxInsolationOverWindow
from anemoi.models.layers.forcing_functions import insolation
from anemoi.models.models.encoder_processor_decoder import AnemoiModelEncProcDec
from anemoi.models.schemas.bounding import ForcingMaskBoundingSchema

LAYOUT = TensorLayout(batch=0, time=1, ensemble=2, grid=3, variables=4)
TIME_FORCINGS = ["cos_julian_day", "sin_julian_day", "cos_local_time", "sin_local_time"]


def _source(values: torch.Tensor, variables: list[str], statistics=None, lat_deg=None) -> GriddedSource:
    grid = values.shape[3]
    lat = torch.zeros(grid) if lat_deg is None else torch.deg2rad(torch.as_tensor(lat_deg, dtype=torch.float64))
    return GriddedSource(
        name="data",
        values=values,
        variables=variables,
        statistics=statistics or {},
        coordinates=torch.stack([lat, torch.zeros(grid, dtype=lat.dtype)], dim=-1),
        layout=LAYOUT,
    )


def _pred(value: float = 7.0, times: int = 1, grid: int = 4, statistics=None) -> GriddedSource:
    return _source(torch.full((1, times, 1, grid, 2), value), ["ssr", "t2m"], statistics)


def _time_forcings(dates: list[datetime.datetime], lat_deg, lon_deg) -> GriddedSource:
    """Forcings ``(1, time, 1, grid, 4)`` defined as in earthkit-data, as float64."""
    lon = np.asarray(lon_deg, dtype=np.float64)
    values = np.zeros((1, len(dates), 1, lon.size, 4))
    for t, date in enumerate(dates):
        delta = date - datetime.datetime(date.year, 1, 1)
        day = 2 * np.pi * (delta.days + delta.seconds / 86400) / 365.25
        hours = date.hour + date.minute / 60
        local = 2 * np.pi * ((lon / 360 * 24 + hours) % 24) / 24
        values[0, t, 0] = np.stack(
            [np.full_like(lon, np.cos(day)), np.full_like(lon, np.sin(day)), np.cos(local), np.sin(local)], axis=-1
        )
    return _source(torch.from_numpy(values), TIME_FORCINGS, lat_deg=np.broadcast_to(lat_deg, lon.shape).copy())


def _insolation_at(source: GriddedSource, shift_hours: float) -> torch.Tensor:
    cj, sj, cl, sl = (source.values[..., i] for i in range(4))
    return insolation(cj, sj, cl, sl, source.coordinates[:, 0], shift_hours)


def test_mask_by_forcing_name() -> None:
    forcings = _source(torch.tensor([0.0, 0.3, -0.1, 1.0]).reshape(1, 1, 1, 4, 1), ["insolation"])
    pred = _pred()
    bounding = ForcingMaskBounding(variables=["ssr"], normalizer=["none"], mask="insolation")

    out = bounding(pred, target_forcings=forcings).values

    assert out[0, 0, 0, :, 0].tolist() == [0.0, 7.0, 0.0, 7.0]
    assert torch.all(out[..., 1] == 7.0)  # t2m untouched
    assert torch.all(pred.values == 7.0)  # the input is not modified


def test_threshold_and_fill_value_are_physical() -> None:
    # the forcing arrives normalised (mean 0.5, stdev 0.25), the fill value is normalised for ssr (mean 2, stdev 4)
    physical = torch.tensor([0.0, 0.1, 0.3, 1.0])
    forcing_stats = {"mean": np.array([0.5]), "stdev": np.array([0.25])}
    forcings = _source(((physical - 0.5) / 0.25).reshape(1, 1, 1, 4, 1), ["insolation"], forcing_stats)
    pred_stats = {"mean": np.array([2.0, 0.0]), "stdev": np.array([4.0, 1.0])}
    bounding = ForcingMaskBounding(
        variables=["ssr"],
        normalizer=["mean-std"],
        mask="insolation",
        mask_normalizer="mean-std",
        operator="lt",
        threshold=0.2,
        fill_value=10.0,
    )

    out = bounding(_pred(3.0, statistics=pred_stats), target_forcings=forcings).values

    assert out[0, 0, 0, :, 0].tolist() == pytest.approx([2.0, 2.0, 3.0, 3.0])  # (10 - 2) / 4 where < 0.2


def test_window_sampling() -> None:
    assert MaxInsolationOverWindow("1h").shifts == [0.0, -1.0]  # the two ends of a one-hour window
    assert MaxInsolationOverWindow("6h").shifts == [0.0, -1.0, -2.0, -3.0, -4.0, -5.0, -6.0]
    assert MaxInsolationOverWindow("90m", step="1h").shifts == [0.0, -1.0, -1.5]
    with pytest.raises(ValueError, match="positive"):
        MaxInsolationOverWindow("0h")


def test_short_day_inside_a_6h_window_is_not_masked() -> None:
    """At 65N near the winter solstice the day is shorter than 6h: it can fall between the ends of a window."""
    t = datetime.datetime(2016, 12, 21, 12)
    forcings = _time_forcings([t], 65.0, np.arange(-180.0, 180.0, 1.0))

    window = MaxInsolationOverWindow("6h")(forcings)
    ends = torch.maximum(_insolation_at(forcings, 0.0), _insolation_at(forcings, -6.0))

    assert torch.all(window >= ends)
    assert bool(((ends == 0) & (window > 0)).any())  # sunlit only inside the window: the two ends would miss it


def test_insolation_matches_earthkit() -> None:
    ekd = pytest.importorskip("earthkit.data")
    rng = np.random.default_rng(0)
    lat, lon = rng.uniform(-90, 90, 2000), rng.uniform(-180, 180, 2000)
    t = datetime.datetime(2016, 9, 30, 13)
    forcings = _time_forcings([t], lat, lon)

    for shift in (0, -1, -6):
        expected = ekd.from_source(
            "forcings", latitudes=lat, longitudes=lon, date=[t + datetime.timedelta(hours=shift)], param=["insolation"]
        )[0].to_numpy(dtype=np.float64, flatten=True)
        computed = _insolation_at(forcings, float(shift))[0, 0, 0].numpy()
        np.testing.assert_allclose(computed, expected, atol=1e-12)


def test_errors() -> None:
    bounding = ForcingMaskBounding(variables=["ssr"], normalizer=["none"], mask="insolation")
    with pytest.raises(ValueError, match="target_forcings"):
        bounding(_pred())
    with pytest.raises(ValueError, match="not among the target forcings"):
        bounding(_pred(), target_forcings=_source(torch.zeros(1, 1, 1, 4, 1), ["lsm"]))
    with pytest.raises(ValueError, match="2 times"):
        bounding(_pred(times=2), target_forcings=_source(torch.zeros(1, 1, 1, 4, 1), ["insolation"]))
    with pytest.raises(ValueError, match="unknown operator"):
        ForcingMaskBounding(variables=["ssr"], normalizer=["none"], mask="insolation", operator="eq")
    with pytest.raises(TypeError, match="forcing name or a callable"):
        ForcingMaskBounding(variables=["ssr"], normalizer=["none"], mask=3)


def test_config_instantiation_and_schema() -> None:
    config = {
        "_target_": "anemoi.models.layers.bounding.ForcingMaskBounding",
        "variables": ["ssr"],
        "normalizer": ["none"],
        "mask": {"_target_": "anemoi.models.layers.forcing_functions.MaxInsolationOverWindow", "window": "6h"},
    }
    # the extra arguments are those injected by the model when building boundings
    bounding = instantiate(config, name_to_index={"ssr": 0}, statistics=None, name_to_index_stats={})
    assert isinstance(bounding.mask, MaxInsolationOverWindow)

    assert ForcingMaskBoundingSchema.model_validate(config).operator == "le"
    assert ForcingMaskBoundingSchema.model_validate(config | {"mask": "insolation"}).mask == "insolation"
    with pytest.raises(ValidationError):
        ForcingMaskBoundingSchema.model_validate(config | {"normalizer": ["none", "none"]})
    with pytest.raises(ValidationError):
        ForcingMaskBoundingSchema.model_validate(config | {"mask": {"window": "6h"}})  # no _target_


def test_model_passes_target_forcings_to_the_boundings() -> None:
    boundings = torch.nn.Sequential(
        ReluBounding(variables=["t2m"]),
        ForcingMaskBounding(variables=["ssr"], normalizer=["none"], mask="insolation"),
    )
    model = SimpleNamespace(boundings={"data": boundings}, _add_residual=lambda pred, skip, name: pred)
    template = SimpleNamespace(unflatten=lambda x: _source(x, ["ssr", "t2m"]))
    x_out = torch.tensor([[5.0, -1.0], [5.0, 2.0]]).reshape(1, 1, 1, 2, 2)
    forcings = _source(torch.tensor([0.0, 0.5]).reshape(1, 1, 1, 2, 1), ["insolation"])

    out = AnemoiModelEncProcDec._assemble_output(
        model, x_out, None, template, x_out.dtype, "data", target_forcings=forcings
    ).values

    assert out[0, 0, 0, :, 0].tolist() == [0.0, 5.0]  # ssr masked where insolation <= 0
    assert out[0, 0, 0, :, 1].tolist() == [0.0, 2.0]  # t2m bounded by the ReLU
    assert math.isclose(float(x_out[0, 0, 0, 0, 0]), 5.0)  # the decoder output is not modified


def test_apply_boundings_accepts_any_container() -> None:
    pred = _pred(-1.0)
    forcings = _source(torch.zeros(1, 1, 1, 4, 1), ["insolation"])
    relu = ReluBounding(variables=["t2m"])
    mask = ForcingMaskBounding(variables=["ssr"], normalizer=["none"], mask="insolation", fill_value=3.0)

    # the module shapes a dataset's boundings can take: none, a single module, a list, the built sequence
    assert torch.equal(apply_boundings(torch.nn.Sequential(), pred).values, pred.values)
    assert torch.equal(apply_boundings(torch.nn.Identity(), pred).values, pred.values)
    assert apply_boundings(torch.nn.ModuleList([relu]), pred).values[..., 1].max() == 0.0

    out = apply_boundings(torch.nn.Sequential(relu, mask), pred, forcings).values
    assert out[..., 0].min() == out[..., 0].max() == 3.0  # ssr filled where the forcing is <= 0
    assert out[..., 1].max() == 0.0  # t2m clipped by the ReLU, applied first
