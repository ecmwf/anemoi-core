# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from pathlib import Path

import numpy as np
import pytest

from anemoi.training.data.residual_statistics import RESIDUAL_STATISTICS_KEYS
from anemoi.training.data.residual_statistics import load_residual_statistics


def _write(path: Path, values: dict[str, dict[str, float]]) -> str:
    np.savez(path, **{key: np.array(mapping, dtype=object) for key, mapping in values.items()})
    return str(path)


def test_statistics_are_ordered_like_the_variables(tmp_path: Path) -> None:
    values = {key: {"a": 1.0 + offset, "b": 2.0 + offset} for offset, key in enumerate(RESIDUAL_STATISTICS_KEYS)}
    path = _write(tmp_path / "stats.npz", values)

    statistics = load_residual_statistics(path, variables=["b", "a"])

    assert set(statistics) == set(RESIDUAL_STATISTICS_KEYS)
    for key in RESIDUAL_STATISTICS_KEYS:
        np.testing.assert_array_equal(statistics[key], [values[key]["b"], values[key]["a"]])


def test_variables_without_an_entry_are_nan(tmp_path: Path) -> None:
    path = _write(tmp_path / "stats.npz", {key: {"a": 1.0} for key in RESIDUAL_STATISTICS_KEYS})

    statistics = load_residual_statistics(path, variables=["a", "force"])

    for key in RESIDUAL_STATISTICS_KEYS:
        np.testing.assert_array_equal(statistics[key], [1.0, np.nan])


def test_a_file_without_a_statistic_is_rejected(tmp_path: Path) -> None:
    path = _write(tmp_path / "stats.npz", {"mean": {"a": 1.0}, "stdev": {"a": 1.0}})

    with pytest.raises(KeyError, match=r"lack the entries \['minimum', 'maximum'\]"):
        load_residual_statistics(path, variables=["a"])
