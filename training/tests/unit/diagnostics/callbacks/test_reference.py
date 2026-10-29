# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import contextlib
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from omegaconf import DictConfig
from pytorch_lightning.utilities import rank_zero_only

from anemoi.models.data_indices.collection import IndexCollection
from anemoi.models.distributed.graph import gather_tensor
from anemoi.training.diagnostics.callbacks.reference import ReferenceComparisonPlot
from anemoi.training.diagnostics.callbacks.reference import regular_grid_spectra
from anemoi.training.tasks import DAForecaster

START = np.datetime64("2025-01-01T00:00", "s")
SIX_HOURS = np.timedelta64(6, "h")


def _grid() -> tuple[np.ndarray, np.ndarray]:
    """Small reduced-Gaussian-like grid: 16 latitude rows, more points near the equator."""
    lats, lons = [], []
    for lat in np.linspace(-80.0, 80.0, 16):
        n_lon = int(20 + 40 * np.cos(np.deg2rad(lat)))
        lats.extend([lat] * n_lon)
        lons.extend(np.linspace(0.0, 360.0, n_lon, endpoint=False))
    return np.asarray(lats), np.asarray(lons)


LAT, LON = _grid()
FIELD = {
    "z_500": 50000.0 + 500.0 * np.cos(np.deg2rad(LAT)) * np.cos(np.deg2rad(2 * LON)),
    "t_850": 270.0 + 10.0 * np.sin(np.deg2rad(LAT)),
}


ZONAL_LEVELS = (850, 500, 250)


def _zonal_fields() -> dict[str, np.ndarray]:
    """u, v, t on three levels with jets, two Hadley-like cells and poleward eddy fluxes."""
    phi, lam = np.deg2rad(LAT), np.deg2rad(LON)
    wave = np.cos(2 * lam)
    fields = {}
    for level, shear, overturning in zip(ZONAL_LEVELS, (0.2, 0.6, 1.0), (-1.0, 0.0, 1.0), strict=True):
        fields[f"u_{level}"] = 40.0 * shear * np.sin(2 * phi) ** 2 + 4.0 * wave * np.sin(phi)
        fields[f"v_{level}"] = overturning * np.sin(6 * phi) * (np.abs(LAT) < 60) + 3.0 * wave * np.cos(phi)
        fields[f"t_{level}"] = 300.0 - 40.0 * (1 - shear) - 30.0 * np.sin(phi) ** 2 + 2.0 * wave * np.sin(phi)
    return fields


class _Reference:
    """Minimal stand-in for an anemoi dataset: time-independent fields."""

    def __init__(self, lat: np.ndarray = LAT, lon: np.ndarray = LON, fields: dict | None = None) -> None:
        self.latitudes, self.longitudes = lat, lon
        self.dates = START + SIX_HOURS * np.arange(40)
        self.fields = FIELD if fields is None else fields
        self.name_to_index = {name: i for i, name in enumerate(self.fields)}
        self.reads = 0

    def __getitem__(self, index: int) -> np.ndarray:
        self.reads += 1
        return np.stack(list(self.fields.values()))[:, None, :]


def _setup(
    steps: list[str] | None = None,
    fields: dict[str, np.ndarray] | None = None,
    pred_offset: dict[str, float] | None = None,
    **kwargs: Any,
) -> tuple[ReferenceComparisonPlot, Any, Any]:
    fields = FIELD if fields is None else fields
    outputs = [name for name in fields if name != "sp"]
    name_to_index = {name: i for i, name in enumerate([*outputs, "cos_latitude"])}
    task = DAForecaster(
        multistep_input=2,
        multistep_output=1,
        timestep="6h",
        rollout={"start": 1, "maximum": 1},
        validation_rollout=1,
        da_cycles=2,
    )
    grid = len(LAT)
    reader = SimpleNamespace(dates=START + SIX_HOURS * np.arange(40))
    dataset = SimpleNamespace(
        data_readers={"data": reader},
        anchors=np.stack([np.zeros(30, dtype=int), np.arange(1, 31)], axis=1),
        get_sample=lambda _index: {
            "data": torch.zeros(len(task.get_offsets("validation")), 1, grid, len(name_to_index)),
        },
    )
    data_indices = {
        "data": IndexCollection(DictConfig({"forcing": ["cos_latitude"], "diagnostic": []}), name_to_index),
    }

    def _step(_batch: dict, validation_mode: bool) -> SimpleNamespace:
        assert validation_mode
        offset = pred_offset or {}
        pred = np.stack([fields[name] + offset.get(name, 0.0) for name in outputs], axis=-1)
        pred = torch.tensor(pred, dtype=torch.float64)
        preds = [{"data": pred.view(1, 1, 1, grid, len(outputs))} for _ in task.steps("validation")]
        return SimpleNamespace(predictions=preds)

    pl_module = SimpleNamespace(
        task=task,
        data_indices=data_indices,
        device=torch.device("cpu"),
        model=SimpleNamespace(
            model=SimpleNamespace(
                _graph_data={"data": SimpleNamespace(x=torch.tensor(np.deg2rad(np.stack([LAT, LON], 1))))},
            ),
            post_processors={"data": lambda x, **_kw: x},
        ),
        transfer_batch_to_device=lambda batch, _device: batch,
        on_after_batch_transfer=lambda batch, _idx: batch,
        _step=_step,
    )
    trainer = SimpleNamespace(
        datamodule=SimpleNamespace(ds_valid=dataset),
        precision_plugin=SimpleNamespace(forward_context=contextlib.nullcontext),
    )
    callback = ReferenceComparisonPlot(
        reference_dataset="unused",
        variables=kwargs.pop("variables", ["z_500", "t_850"]),
        dates=["2025-01-02T00:00"],
        steps=steps,
        high_k=5,
        **kwargs,
    )
    callback._reference = _Reference(fields=fields)
    return callback, trainer, pl_module


def test_valid_times_follow_task_offsets() -> None:
    callback, trainer, pl_module = _setup(steps=["dacycle0", "dacycle1", "rstep0"])
    result = callback.compute(trainer, pl_module)
    anchor = np.datetime64("2025-01-02T00:00", "s")
    assert [case["step"] for case in result["cases"]] == ["dacycle0", "dacycle1", "rstep0"]
    assert [case["valid_time"] for case in result["cases"]] == [anchor + SIX_HOURS * i for i in (1, 2, 3)]


def test_default_step_is_last_and_identical_fields_score_zero() -> None:
    callback, trainer, pl_module = _setup(scores=["rmse", "bias", "highk_ratio"])
    result = callback.compute(trainer, pl_module)
    assert [case["step"] for case in result["cases"]] == ["rstep0"]
    scores = callback.scores(result)
    assert scores["val_ref_rmse/z_500/rstep0"] == pytest.approx(0.0, abs=1e-2)
    assert scores["val_ref_bias/t_850/rstep0"] == pytest.approx(0.0, abs=1e-3)
    assert scores["val_ref_highk_ratio/z_500/rstep0"] == pytest.approx(1.0, rel=1e-3)


def test_grid_mismatch_raises() -> None:
    callback, trainer, pl_module = _setup()
    callback._reference = _Reference(lat=LAT + 1.0)
    with pytest.raises(ValueError, match="does not match"):
        callback.compute(trainer, pl_module)


def test_unknown_step_and_anchor_raise() -> None:
    callback, trainer, pl_module = _setup(steps=["rstep5"])
    with pytest.raises(ValueError, match="not in the validation task steps"):
        callback.compute(trainer, pl_module)
    callback, trainer, pl_module = _setup()
    callback.dates = [np.datetime64("2030-01-01T00:00", "s")]
    with pytest.raises(ValueError, match="not a valid validation sample anchor"):
        callback.compute(trainer, pl_module)


def test_spectra_separate_smooth_and_noisy_fields() -> None:
    rng = np.random.default_rng(0)
    smooth = FIELD["z_500"] - FIELD["z_500"].mean()
    noisy = smooth + rng.normal(0.0, 50.0, smooth.shape)
    p_smooth, p_noisy = regular_grid_spectra(LAT, LON, [smooth, noisy])
    assert p_smooth.shape == p_noisy.shape
    assert p_noisy[-5:].sum() > 10 * p_smooth[-5:].sum()


def test_plot_renders(tmp_path: Path) -> None:
    callback, trainer, pl_module = _setup(steps=["dacycle1", "rstep0"])
    callback.save_basedir = str(tmp_path)
    result = callback.compute(trainer, pl_module)
    callback._plot(SimpleNamespace(logger=None), pl_module, ["data"], epoch=0, result=result)
    # One figure per date holding every variable and step.
    assert len(list((tmp_path / "plots").glob("ref_*_epoch000.jpg"))) == 1


def test_spectral_grid_matches_scipy_griddata() -> None:
    from scipy.interpolate import griddata

    from anemoi.training.diagnostics.callbacks.reference import SpectralGrid

    grid = SpectralGrid(LAT, LON)
    field = FIELD["z_500"]
    ours = grid.interpolate(field)
    n_lat, n_lon = grid.shape
    mesh_lon, mesh_lat = np.meshgrid(
        np.linspace(0.0, 360.0, n_lon, endpoint=False),
        np.linspace(LAT.max(), LAT.min(), n_lat),
    )
    pad = (np.concatenate((LON - 360.0, LON, LON + 360.0)), np.concatenate((LAT, LAT, LAT)))
    reference = griddata(pad, np.concatenate((field, field, field)), (mesh_lon, mesh_lat), method="linear")
    np.testing.assert_allclose(ours, reference, rtol=1e-10)


def test_plot_thread_does_no_numerics(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Spectra are computed once in compute(); the plot executor thread only draws."""
    from anemoi.training.diagnostics.callbacks import reference

    callback, trainer, pl_module = _setup(steps=["rstep0"], mean_maps=True)
    callback.save_basedir = str(tmp_path)
    result = callback.compute(trainer, pl_module)

    def _fail(*_args: Any, **_kwargs: Any) -> None:
        msg = "numerics must not run on the plot thread"
        raise AssertionError(msg)

    monkeypatch.setattr(reference, "compute_spectra", _fail)
    monkeypatch.setattr(reference.SpectralGrid, "spectra", _fail)
    monkeypatch.setattr(reference.ZonalGrid, "mean", _fail)
    callback.scores(result)
    callback._plot(SimpleNamespace(logger=None), pl_module, ["data"], epoch=0, result=result)


def test_score_dates_are_scored_not_plotted(tmp_path: Path) -> None:
    callback, trainer, pl_module = _setup(
        steps=["rstep0"],
        score_dates=["2025-01-02T00:00", "2025-01-03T00:00", "2025-01-04T06:00"],
        scores=["rmse", "highk_ratio"],
    )
    # The plotted date is not repeated among the score-only dates.
    assert len(callback.score_dates) == 2
    callback.save_basedir = str(tmp_path)
    result = callback.compute(trainer, pl_module)
    assert [case["plot"] for case in result["cases"]] == [True, False, False]
    assert ["spectra" in case for case in result["cases"]] == [True, False, False]

    scores = callback.scores(result)
    assert scores["val_ref_rmse/z_500/rstep0"] == pytest.approx(0.0, abs=1e-2)
    assert scores["val_ref_highk_ratio/z_500/rstep0"] == pytest.approx(1.0, rel=1e-3)
    callback._plot(SimpleNamespace(logger=None), pl_module, ["data"], epoch=0, result=result)
    assert len(list((tmp_path / "plots").glob("ref_*_epoch000.jpg"))) == 1


def test_mean_maps_over_all_dates(tmp_path: Path) -> None:
    callback, trainer, pl_module = _setup(
        steps=["dacycle1", "rstep0"],
        score_dates=["2025-01-03T00:00", "2025-01-04T06:00"],
        mean_maps=True,
        pred_offset={"t_850": 1.5},
    )
    callback.save_basedir = str(tmp_path)
    result = callback.compute(trainer, pl_module)
    assert sorted(result["mean_maps"]) == ["dacycle1", "rstep0"]
    stats = result["mean_maps"]["rstep0"]["t_850"]
    assert stats["n_dates"] == 3
    np.testing.assert_allclose(stats["bias"], 1.5, rtol=1e-5)
    np.testing.assert_allclose(stats["rmse"], 1.5, rtol=1e-5)
    np.testing.assert_allclose(stats["lat_rmse"], 1.5, rtol=1e-5)
    assert stats["lat_bias"].shape == (16,)
    np.testing.assert_allclose(result["mean_maps"]["rstep0"]["z_500"]["rmse"], 0.0, atol=1e-2)

    callback._plot(SimpleNamespace(logger=None), pl_module, ["data"], epoch=0, result=result)
    plots = tmp_path / "plots"
    assert len(list(plots.glob("ref_mean_*_epoch000.jpg"))) == 2  # one per step
    assert len(list(plots.glob("ref_2025*_epoch000.jpg"))) == 1  # per-date maps for plotted dates only


def test_no_plotted_dates_keeps_scores_and_mean_figures(tmp_path: Path) -> None:
    callback, trainer, pl_module = _setup(steps=["rstep0"], mean_maps=True, scores=["rmse", "highk_ratio"])
    callback.dates = []
    callback.score_dates = [np.datetime64("2025-01-03T00:00", "s"), np.datetime64("2025-01-04T00:00", "s")]
    callback.save_basedir = str(tmp_path)
    result = callback.compute(trainer, pl_module)
    assert not any("spectra" in case for case in result["cases"])
    scores = callback.scores(result)
    assert "val_ref_rmse/z_500/rstep0" in scores
    assert not any(key.startswith("val_ref_highk_ratio") for key in scores)
    callback._plot(SimpleNamespace(logger=None), pl_module, ["data"], epoch=0, result=result)
    plots = tmp_path / "plots"
    assert len(list(plots.glob("ref_mean_*_epoch000.jpg"))) == 1
    assert not list(plots.glob("ref_2025*_epoch000.jpg"))


def test_mean_maps_off_by_default() -> None:
    callback, trainer, pl_module = _setup()
    assert "mean_maps" not in callback.compute(trainer, pl_module)


def test_band_scores() -> None:
    callback, _, _ = _setup(bands=True, scores=["rmse", "bias", "highk_ratio"])
    ref = FIELD["t_850"]
    pred = ref + np.where(LAT < -20.0, 2.0, 0.0)
    result = {"lat": LAT, "cases": [{"step": "rstep0", "fields": {"t_850": (pred, ref)}}]}

    scores = callback.scores(result)
    sh_fraction = np.mean(LAT < -20.0)
    assert scores["val_ref_rmse/sh/t_850/rstep0"] == pytest.approx(2.0)
    assert scores["val_ref_bias/sh/t_850/rstep0"] == pytest.approx(2.0)
    assert scores["val_ref_rmse/tropics/t_850/rstep0"] == pytest.approx(0.0)
    assert scores["val_ref_rmse/nh/t_850/rstep0"] == pytest.approx(0.0)
    assert scores["val_ref_rmse/t_850/rstep0"] == pytest.approx(2.0 * np.sqrt(sh_fraction))
    assert "val_ref_highk_ratio/t_850/rstep0" not in scores


def test_bands_off_by_default() -> None:
    callback, _, _ = _setup()
    ref = FIELD["t_850"]
    scores = callback.scores({"lat": LAT, "cases": [{"step": "rstep0", "fields": {"t_850": (ref, ref)}}]})
    assert set(scores) == {"val_ref_rmse/t_850/rstep0"}


def test_band_variables_limit_band_scores() -> None:
    callback, _, _ = _setup(bands=True, band_variables=["t_850"], scores=["rmse", "bias"])
    fields = {var: (FIELD[var], FIELD[var]) for var in ("z_500", "t_850")}
    scores = callback.scores({"lat": LAT, "cases": [{"step": "rstep0", "fields": fields}]})
    banded = sorted(key for key in scores if key.split("/")[1] in ("nh", "tropics", "sh"))
    assert banded == sorted(f"val_ref_{m}/{b}/t_850/rstep0" for m in ("rmse", "bias") for b in ("nh", "tropics", "sh"))
    assert "val_ref_rmse/z_500/rstep0" in scores


def test_unknown_band_variables_raise() -> None:
    with pytest.raises(ValueError, match="band_variables"):
        _setup(bands=True, band_variables=["q_700"])


def test_default_scores_are_rmse_only() -> None:
    callback, trainer, pl_module = _setup(bands=True)
    scores = callback.scores(callback.compute(trainer, pl_module))
    assert scores
    assert all(key.startswith("val_ref_rmse/") for key in scores)


def test_unknown_scores_and_zonal_options_raise() -> None:
    with pytest.raises(ValueError, match="unknown scores"):
        _setup(scores=["rmse", "mae"])
    with pytest.raises(ValueError, match="unknown zonal options"):
        _setup(zonal={"hadley": True})


def test_reference_read_once_per_date() -> None:
    callback, trainer, pl_module = _setup(steps=["dacycle1", "rstep0"], score_dates=["2025-01-03T00:00"])
    callback.compute(trainer, pl_module)
    assert callback.reference.reads == 4  # 2 dates x 2 steps, however many variables


def _zonal_setup(**kwargs: Any) -> tuple[ReferenceComparisonPlot, Any, Any]:
    fields = {**_zonal_fields(), "sp": np.full(LAT.shape, 1.0e5)}
    return _setup(steps=["rstep0"], fields=fields, variables=["t_850"], **kwargs)


ZONAL_KEYS = {
    "sections": ["u_rmse", "t_rmse", *(f"jet_{h}_{q}_err" for h in ("nh", "sh") for q in ("lat", "speed"))],
    "rmse_sections": ["u_point_rmse", "v_point_rmse", "t_point_rmse"],
    "thermal_wind": ["tw_residual_rmse"],
    "streamfunction": [f"hadley_{h}_{q}" for h in ("nh", "sh") for q in ("strength_ratio", "edge_err")],
    "eddy_fluxes": [f"{f}_{h}_ratio" for f in ("vt850", "uv250") for h in ("nh", "sh")],
}


def test_zonal_identical_fields_score_perfectly(tmp_path: Path) -> None:
    callback, trainer, pl_module = _zonal_setup(zonal={})
    callback.save_basedir = str(tmp_path)
    result = callback.compute(trainer, pl_module)
    summary = result["zonal"]["rstep0"]
    assert summary["levels"].tolist() == [250.0, 500.0, 850.0]
    assert "zonal" not in result["cases"][0]  # per-date moments are dropped after averaging

    scores = callback.scores(result)
    zonal = {key.split("/")[1]: value for key, value in scores.items() if key.startswith("val_ref_zonal/")}
    assert sorted(zonal) == sorted(key for keys in ZONAL_KEYS.values() for key in keys)
    for key, value in zonal.items():
        expected = 1.0 if key.endswith("ratio") else 0.0
        if key == "tw_residual_rmse":
            continue  # the synthetic winds are not in thermal-wind balance
        assert value == pytest.approx(expected, abs=1e-5), key  # float32 predictions
    # Sanity of the synthetic circulation: NH Hadley cell positive, SH negative, edges near +-30.
    assert summary["ref"]["hadley"]["nh"][0] > 0 > summary["ref"]["hadley"]["sh"][0]
    assert summary["ref"]["hadley"]["nh"][1] == pytest.approx(30.0, abs=6.0)
    assert summary["ref"]["hadley"]["sh"][1] == pytest.approx(-30.0, abs=6.0)

    callback._plot(SimpleNamespace(logger=None), pl_module, ["data"], epoch=0, result=result)
    assert len(list((tmp_path / "plots").glob("zonal_rstep0_epoch000.jpg"))) == 1


def test_zonal_offsets_show_in_errors() -> None:
    callback, trainer, pl_module = _zonal_setup(
        zonal={},
        pred_offset={f"u_{level}": 2.0 for level in ZONAL_LEVELS} | {f"t_{level}": 1.0 for level in ZONAL_LEVELS},
    )
    scores = callback.scores(callback.compute(trainer, pl_module))
    assert scores["val_ref_zonal/u_rmse/rstep0"] == pytest.approx(2.0)
    assert scores["val_ref_zonal/t_rmse/rstep0"] == pytest.approx(1.0)
    assert scores["val_ref_zonal/jet_nh_speed_err/rstep0"] == pytest.approx(2.0)
    assert scores["val_ref_zonal/jet_sh_lat_err/rstep0"] == pytest.approx(0.0, abs=1e-5)
    # Point-wise errors: a uniform offset gives the same RMSE on every level and latitude.
    assert scores["val_ref_zonal/u_point_rmse/rstep0"] == pytest.approx(2.0)
    assert scores["val_ref_zonal/t_point_rmse/rstep0"] == pytest.approx(1.0)
    assert scores["val_ref_zonal/v_point_rmse/rstep0"] == pytest.approx(0.0, abs=1e-5)


def test_zonal_rmse_sections_see_errors_the_zonal_mean_cancels() -> None:
    """A zonal-wave error leaves [u] unchanged but shows in the point-wise RMSE section."""
    wave = np.cos(np.deg2rad(3 * LON))
    callback, trainer, pl_module = _zonal_setup(zonal={})
    fields = dict(callback.reference.fields)  # the prediction keeps the unperturbed fields
    for level in ZONAL_LEVELS:
        fields[f"u_{level}"] = fields[f"u_{level}"] - 2.0 * wave
    callback._reference = type(callback.reference)(fields=fields)
    result = callback.compute(trainer, pl_module)
    scores = callback.scores(result)
    assert scores["val_ref_zonal/u_rmse/rstep0"] == pytest.approx(0.0, abs=1e-2)
    assert scores["val_ref_zonal/u_point_rmse/rstep0"] == pytest.approx(2.0 / np.sqrt(2.0), rel=1e-2)
    assert result["zonal"]["rstep0"]["err"]["u"].shape == (len(ZONAL_LEVELS), 16)


def test_zonal_flags_select_scores_and_rows(tmp_path: Path) -> None:
    callback, trainer, pl_module = _zonal_setup(zonal={"sections": False, "thermal_wind": False})
    callback.save_basedir = str(tmp_path)
    result = callback.compute(trainer, pl_module)
    zonal = {key.split("/")[1] for key in callback.scores(result) if key.startswith("val_ref_zonal/")}
    assert zonal == set(ZONAL_KEYS["rmse_sections"] + ZONAL_KEYS["streamfunction"] + ZONAL_KEYS["eddy_fluxes"])
    callback._plot(SimpleNamespace(logger=None), pl_module, ["data"], epoch=0, result=result)
    assert len(list((tmp_path / "plots").glob("zonal_rstep0_epoch000.jpg"))) == 1


def test_zonal_off_when_unset_or_all_flags_false() -> None:
    for zonal in (None, dict.fromkeys(ZONAL_KEYS, False)):
        callback, trainer, pl_module = _zonal_setup(zonal=zonal)
        result = callback.compute(trainer, pl_module)
        assert "zonal" not in result
        assert not any(key.startswith("val_ref_zonal/") for key in callback.scores(result))


def test_zonal_missing_level_raises() -> None:
    callback, trainer, pl_module = _zonal_setup(zonal={"vt_level": 700})
    with pytest.raises(ValueError, match="zonal diagnostics need u, v, t"):
        callback.compute(trainer, pl_module)


def test_zonal_below_ground_is_masked() -> None:
    callback, trainer, pl_module = _zonal_setup(zonal={})
    callback.reference.fields["sp"] = np.where(LAT < -60.0, 7.0e4, 1.0e5)  # 850 hPa underground south of 60S
    summary = callback.compute(trainer, pl_module)["zonal"]["rstep0"]
    south = summary["lat"] < -60.0
    assert np.all(np.isnan(summary["ref"]["u"][-1, south]))
    assert np.all(summary["ref"]["frac"][-1, south] == 0.0)
    assert np.all(np.isfinite(summary["ref"]["psi"]))


def _check_epoch_end_with_member_gather(rank: int, store_path: str) -> None:
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        init_method=f"file://{store_path}",
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=30),
    )
    rank_zero_only.rank = rank
    try:
        callback, trainer, pl_module = _setup()
        local_step = pl_module._step

        def _step(batch: dict, validation_mode: bool) -> SimpleNamespace:
            # Like DAEnsembleTraining._output_prediction: gather the members over the ensemble group.
            output = local_step(batch, validation_mode)
            output.predictions = [
                {name: gather_tensor(pred, dim=2, sizes=[1, 1], mgroup=dist.group.WORLD) for name, pred in step.items()}
                for step in output.predictions
            ]
            return output

        pl_module._step = _step
        trainer.sanity_checking, trainer.current_epoch, trainer.logger = False, 0, None
        plotted = []
        callback.plot = lambda *_args, **kwargs: plotted.append(kwargs["result"])
        callback.on_validation_epoch_end(trainer, pl_module)
        dist.barrier()  # the next collective every rank enters after validation
        assert len(plotted) == (1 if rank == 0 else 0)
    finally:
        dist.destroy_process_group()


def test_epoch_end_runs_the_forward_on_every_rank(tmp_path: Path) -> None:
    """The ensemble forward gathers members across ranks, so rank 0 cannot run it alone (control_ens hang)."""
    mp.spawn(_check_epoch_end_with_member_gather, args=(str(tmp_path / "process_group"),), nprocs=2, join=True)
