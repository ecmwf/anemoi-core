# (C) Copyright 2024-2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import pytest

from anemoi.training.data.datamodule import AnemoiDatasetsDataModule

DATASET_NAME = "target_dataset"


def _make_datamodule(state_stdev: list[float]) -> AnemoiDatasetsDataModule:
    """Build a bare datamodule with just enough state to exercise residual-stats validation.

    ``_validate_residual_statistics`` only needs ``self.statistics[dataset_name]["stdev"]`` to
    determine the expected array length, so we set that directly rather than going through the
    full dataset-construction machinery (mirrors the ``__new__`` + attribute-injection pattern
    used elsewhere in test_datamodule.py).
    """
    datamodule = AnemoiDatasetsDataModule.__new__(AnemoiDatasetsDataModule)
    datamodule.statistics = {DATASET_NAME: {"stdev": state_stdev}}
    return datamodule


def _residual_stats(length: int, stdev_value: float = 1.0) -> dict:
    return {
        "minimum": [0.0] * length,
        "maximum": [1.0] * length,
        "mean": [0.5] * length,
        "stdev": [stdev_value] * length,
    }


def test_correct_length_residual_statistics_pass() -> None:
    datamodule = _make_datamodule(state_stdev=[1.0, 2.0, 3.0])

    validated = datamodule._validate_residual_statistics(_residual_stats(length=3), DATASET_NAME)

    assert len(validated["stdev"]) == 3


@pytest.mark.parametrize("field", ["minimum", "maximum", "mean", "stdev"])
def test_wrong_length_residual_statistics_raise(field: str) -> None:
    """A truncated or reordered stats file must fail loudly, naming dataset/field/lengths."""
    datamodule = _make_datamodule(state_stdev=[1.0, 2.0, 3.0])  # expected length 3

    stats = _residual_stats(length=3)
    stats[field] = stats[field][:-1]  # truncate this one field to length 2

    with pytest.raises(ValueError, match=field) as excinfo:
        datamodule._validate_residual_statistics(stats, DATASET_NAME)

    message = str(excinfo.value)
    assert DATASET_NAME in message
    assert field in message
    assert "2" in message
    assert "3" in message


def test_missing_field_raises() -> None:
    datamodule = _make_datamodule(state_stdev=[1.0, 2.0, 3.0])
    stats = _residual_stats(length=3)
    del stats["mean"]

    with pytest.raises(ValueError, match="mean"):
        datamodule._validate_residual_statistics(stats, DATASET_NAME)


def test_non_finite_values_raise() -> None:
    datamodule = _make_datamodule(state_stdev=[1.0, 2.0, 3.0])
    stats = _residual_stats(length=3)
    stats["mean"][1] = float("nan")

    with pytest.raises(ValueError, match="finite"):
        datamodule._validate_residual_statistics(stats, DATASET_NAME)


def test_non_positive_stdev_raises() -> None:
    datamodule = _make_datamodule(state_stdev=[1.0, 2.0, 3.0])
    stats = _residual_stats(length=3)
    stats["stdev"][0] = 0.0

    with pytest.raises(ValueError, match="positive"):
        datamodule._validate_residual_statistics(stats, DATASET_NAME)
