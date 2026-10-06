# (C) Copyright 2026 Anemoi contributors.

"""Tests for query diagnostic coordinate handling."""

from types import SimpleNamespace

import numpy as np

from anemoi.training.diagnostics.callbacks.query import QueryDiagnosticsPlot
from anemoi.training.diagnostics.callbacks.query import SOURCE_COLOURS
from anemoi.training.diagnostics.callbacks.query import _degrees
from anemoi.training.diagnostics.callbacks.query import _extent_mask
from anemoi.training.diagnostics.callbacks.query import _field_semantic_rank
from anemoi.training.diagnostics.callbacks.query import _regional_extent
from anemoi.training.diagnostics.callbacks.query import _target_equivalent


def test_degrees_wraps_zero_to_360_longitudes() -> None:
    coordinates = np.deg2rad(np.asarray([[60.0, 350.0], [61.0, 10.0]], dtype=np.float32))

    result = _degrees(coordinates)

    np.testing.assert_allclose(result[:, 0], [60.0, 61.0], atol=1e-5)
    np.testing.assert_allclose(result[:, 1], [-10.0, 10.0], atol=1e-5)


def test_degrees_can_preserve_native_zero_to_360_longitudes() -> None:
    coordinates = np.deg2rad(np.asarray([[60.0, 350.0], [61.0, 10.0]], dtype=np.float32))

    result = _degrees(coordinates, preserve_native_longitude=True)

    np.testing.assert_allclose(result[:, 0], [60.0, 61.0], atol=1e-5)
    np.testing.assert_allclose(result[:, 1], [350.0, 10.0], atol=1e-5)


def test_extent_mask_selects_regional_points_before_plot_sampling() -> None:
    coordinates = np.asarray(
        [
            [60.0, 10.0],
            [65.0, 20.0],
            [-20.0, 20.0],
            [65.0, 170.0],
        ],
    )

    selected = _extent_mask(coordinates, [-20.0, 40.0, 50.0, 75.0])

    np.testing.assert_array_equal(selected, [True, True, False, False])


def test_extent_mask_accepts_native_ifs_longitudes_across_zero() -> None:
    coordinates = np.asarray(
        [
            [60.0, 350.0],
            [65.0, 20.0],
            [65.0, 300.0],
            [65.0, 170.0],
        ],
    )

    selected = _extent_mask(coordinates, [-20.0, 40.0, 50.0, 75.0])

    np.testing.assert_array_equal(selected, [True, True, False, False])


def test_regional_extent_and_ifs_colour_are_stable() -> None:
    coordinates = np.asarray([[49.5, -18.0], [75.0, 54.0]])

    assert _regional_extent(coordinates) == [-18.0, 54.0, 49.5, 75.0]
    assert SOURCE_COLOURS["IFS"] == "tab:blue"


def test_rotating_validation_case_preserves_fixed_reference() -> None:
    callback = QueryDiagnosticsPlot(
        enabled=True,
        max_cases=2,
        fixed_validation_cases=[0],
        rotating_validation_case=True,
    )
    trainer = SimpleNamespace(
        current_epoch=2,
        sanity_checking=False,
        datamodule=SimpleNamespace(ds_valid=range(12)),
    )

    callback.on_train_epoch_start(trainer, None)

    assert callback.fixed_validation_cases == {0}
    assert callback._rotating_validation_index == 3


def test_semantic_input_fallback_prefers_ifs_10m_wind_for_100m_wind() -> None:
    target = {
        "variable": "u",
        "level_type": "height",
        "height_m": 100.0,
        "aggregation_type": "instantaneous",
        "temporal_aggregation_window_hours": None,
        "units": "m s-1",
    }
    terrain = {
        "variable": "z",
        "level_type": "surface",
        "aggregation_type": "instantaneous",
        "temporal_aggregation_window_hours": None,
        "units": "m2 s-2",
    }
    wind_10m = {
        "variable": "u",
        "level_type": "height",
        "height_m": 10.0,
        "aggregation_type": "instantaneous",
        "temporal_aggregation_window_hours": None,
        "units": "m s-1",
    }

    assert min((terrain, wind_10m), key=lambda field: _field_semantic_rank(field, target)) is wind_10m
    assert not _target_equivalent(wind_10m, target)


def test_exact_semantic_input_wins() -> None:
    target = {
        "variable": "t",
        "level_type": "height",
        "height_m": 2.0,
        "aggregation_type": "instantaneous",
        "temporal_aggregation_window_hours": None,
        "units": "K",
    }
    exact = dict(target)
    pressure_temperature = dict(target, level_type="pressure", height_m=None, pressure_pa=85000.0)

    assert min((pressure_temperature, exact), key=lambda field: _field_semantic_rank(field, target)) is exact
    assert _target_equivalent(exact, target)
