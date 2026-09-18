# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from __future__ import annotations

from pathlib import Path  # noqa: TC003

import numpy as np
import pytest

from anemoi.training.data.residual_statistics import load_residual_statistics


def _write_npz(path: Path, *, mean: dict, minimum: dict, maximum: dict, stdev: dict) -> None:
    np.savez(
        path,
        mean=np.array(mean),
        minimum=np.array(minimum),
        maximum=np.array(maximum),
        stdev=np.array(stdev),
    )


def test_load_residual_statistics_returns_arrays_ordered_like_variables(tmp_path: Path) -> None:
    path = tmp_path / "residuals.npz"
    _write_npz(
        path,
        mean={"a": 1.0, "b": 2.0, "c": 3.0},
        minimum={"a": -1.0, "b": -2.0, "c": -3.0},
        maximum={"a": 1.0, "b": 2.0, "c": 3.0},
        stdev={"a": 0.1, "b": 0.2, "c": 0.3},
    )

    statistics = load_residual_statistics(str(path), variables=["c", "a", "b"])

    assert list(statistics.keys()) == ["mean", "minimum", "maximum", "stdev"]
    np.testing.assert_allclose(statistics["mean"], [3.0, 1.0, 2.0])
    np.testing.assert_allclose(statistics["minimum"], [-3.0, -1.0, -2.0])
    np.testing.assert_allclose(statistics["maximum"], [3.0, 1.0, 2.0])
    np.testing.assert_allclose(statistics["stdev"], [0.3, 0.1, 0.2])
    assert all(arr.dtype == np.float32 for arr in statistics.values())


def test_load_residual_statistics_raises_key_error_for_missing_variable(tmp_path: Path) -> None:
    path = tmp_path / "residuals.npz"
    _write_npz(
        path,
        mean={"a": 1.0},
        minimum={"a": -1.0},
        maximum={"a": 1.0},
        stdev={"a": 0.1},
    )

    with pytest.raises(KeyError):
        load_residual_statistics(str(path), variables=["a", "missing"])


def test_load_residual_statistics_raises_value_error_for_non_finite_entry(tmp_path: Path) -> None:
    path = tmp_path / "residuals.npz"
    _write_npz(
        path,
        mean={"a": 1.0, "b": float("nan")},
        minimum={"a": -1.0, "b": -2.0},
        maximum={"a": 1.0, "b": 2.0},
        stdev={"a": 0.1, "b": 0.2},
    )

    with pytest.raises(ValueError, match="mean"):
        load_residual_statistics(str(path), variables=["a", "b"])
