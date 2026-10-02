# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch
from omegaconf import DictConfig

from anemoi.models.data_indices.collection import IndexCollection
from anemoi.training.diagnostics.callbacks.reference import ReferenceComparisonPlot
from anemoi.training.diagnostics.callbacks.reference import regular_grid_spectra
from anemoi.training.tasks import DAForecaster

NAME_TO_INDEX = {"z_500": 0, "t_850": 1, "cos_latitude": 2}
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


class _Reference:
    """Minimal stand-in for an anemoi dataset: time-independent fields."""

    def __init__(self, lat: np.ndarray = LAT, lon: np.ndarray = LON) -> None:
        self.latitudes, self.longitudes = lat, lon
        self.dates = START + SIX_HOURS * np.arange(40)
        self.name_to_index = {"z_500": 0, "t_850": 1}

    def __getitem__(self, index: int) -> np.ndarray:
        return np.stack([FIELD["z_500"], FIELD["t_850"]])[:, None, :]


def _setup(steps: list[str] | None = None) -> tuple[ReferenceComparisonPlot, Any, Any]:
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
        get_sample=lambda _index: {"data": torch.zeros(len(task.get_offsets("validation")), 1, grid, 3)},
    )
    data_indices = {
        "data": IndexCollection(DictConfig({"forcing": ["cos_latitude"], "diagnostic": []}), NAME_TO_INDEX),
    }

    def _step(_batch: dict, validation_mode: bool) -> SimpleNamespace:
        assert validation_mode
        pred = torch.tensor(np.stack([FIELD["z_500"], FIELD["t_850"]], axis=-1), dtype=torch.float32)
        preds = [{"data": pred.view(1, 1, 1, grid, 2)} for _ in task.steps("validation")]
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
    trainer = SimpleNamespace(datamodule=SimpleNamespace(ds_valid=dataset))
    callback = ReferenceComparisonPlot(
        reference_dataset="unused",
        variables=["z_500", "t_850"],
        dates=["2025-01-02T00:00"],
        steps=steps,
        high_k=5,
    )
    callback._reference = _Reference()
    return callback, trainer, pl_module


def test_valid_times_follow_task_offsets() -> None:
    callback, trainer, pl_module = _setup(steps=["dacycle0", "dacycle1", "rstep0"])
    result = callback.compute(trainer, pl_module)
    anchor = np.datetime64("2025-01-02T00:00", "s")
    assert [case["step"] for case in result["cases"]] == ["dacycle0", "dacycle1", "rstep0"]
    assert [case["valid_time"] for case in result["cases"]] == [anchor + SIX_HOURS * i for i in (1, 2, 3)]


def test_default_step_is_last_and_identical_fields_score_zero() -> None:
    callback, trainer, pl_module = _setup()
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
