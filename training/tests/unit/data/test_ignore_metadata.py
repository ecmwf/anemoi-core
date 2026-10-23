# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""``dataloader.ignore_metadata`` must index a cloned dataset over all of its data.

Cloned datasets keep the metadata of the dataset they were cloned from: the ``dates``
array is only valid for the original time range (the rest is garbage) and can be shorter
than ``data``. Any start/end/frequency subsetting indexes those dates and fails.
"""

import datetime
from pathlib import Path

import numpy as np
import pytest
import zarr

from anemoi.training.data.data_reader import create_dataset
from anemoi.training.data.relative_time_indices import compute_relative_date_indices

N_DATES = 40
N_VALID_DATES = 8  # dates written by the original dataset, the rest are garbage
VARIABLES = ["10u", "2t", "msl"]
N_GRID = 16
CLONED_O800 = Path("/lus/h1aiws01/naco/cloned-hdd/cloned-o800-20kdates.zarr")


class SixHourlyTask:
    """Minimal forecaster-like task: two input steps and one output step at 6h."""

    def get_offsets(self, **_kwargs) -> list[datetime.timedelta]:
        return [datetime.timedelta(hours=h) for h in (-6, 0, 6)]


@pytest.fixture
def cloned_dataset(tmp_path: Path) -> str:
    """A small zarr laid out like a clone: dates shorter than data and mostly invalid."""
    path = tmp_path / "cloned.zarr"
    root = zarr.open_group(str(path), mode="w", zarr_format=2) if zarr.__version__ >= "3" else zarr.open(str(path), "w")

    data = np.arange(N_DATES * len(VARIABLES) * N_GRID, dtype="f4").reshape(N_DATES, len(VARIABLES), 1, N_GRID)
    root.create_dataset("data", data=data, chunks=(1, len(VARIABLES), 1, N_GRID))

    dates = np.zeros(N_DATES - 1, dtype="datetime64[s]")  # one short, like the o800 clone
    dates[:N_VALID_DATES] = np.datetime64("2023-01-01T00:00:00") + np.arange(N_VALID_DATES) * np.timedelta64(6, "h")
    dates[N_VALID_DATES + 4 :] = np.datetime64("2001-05-05T00:00:00")  # garbage after some 1970 zeros
    root.create_dataset("dates", data=dates)

    root.create_dataset("latitudes", data=np.linspace(-80, 80, N_GRID))
    root.create_dataset("longitudes", data=np.linspace(0, 350, N_GRID))
    for name in ("mean", "stdev", "maximum", "minimum", "sums", "squares", "count", "has_nans"):
        root.create_dataset(name, data=np.ones(len(VARIABLES)))

    root.attrs.update(
        {
            "frequency": "6h",
            "resolution": "o800",
            "start_date": "2023-01-01T00:00:00",
            "end_date": "2023-01-02T18:00:00",
            "variables": VARIABLES,
            "missing_dates": [],
            "ensemble_dimension": 1,
            "field_shape": [N_GRID],
            "flatten_grid": True,
            "version": "0.30",
        },
    )
    return str(path)


def _reader_config(dataset: str, frequency: str | None = "6h") -> dict:
    """Dataloader reader config as built from config/dataloader/native_grid.yaml."""
    dataset_config = {"dataset": dataset}
    if frequency is not None:
        dataset_config["frequency"] = frequency
    return {"dataset_config": dataset_config, "start": "2023-01-01", "end": "2023-02-01"}


@pytest.mark.parametrize("frequency", ["6h", None])
def test_ignore_metadata_reads_all_data(cloned_dataset: str, frequency: str | None) -> None:
    reader = create_dataset(_reader_config(cloned_dataset, frequency), ignore_metadata=True)

    assert len(reader.data) == N_DATES
    assert reader.sequence_length() == N_DATES - 1  # bounded by the (one short) dates array
    assert reader.frequency == datetime.timedelta(hours=6)
    # the last positions have no valid date but must still be readable
    data = np.arange(N_DATES * len(VARIABLES) * N_GRID, dtype="f4").reshape(N_DATES, len(VARIABLES), 1, N_GRID)
    expected = np.moveaxis(data[N_DATES - 2 :], 1, -1)  # dates ensemble gridpoints variables
    np.testing.assert_array_equal(reader.get_sample(0, slice(N_DATES - 2, N_DATES)).numpy(), expected)


def test_ignore_metadata_relative_date_indices(cloned_dataset: str) -> None:
    readers = {"data": create_dataset(_reader_config(cloned_dataset), ignore_metadata=True)}
    assert compute_relative_date_indices(SixHourlyTask(), readers) == {"data": [-1, 0, 1]}


def test_ignore_metadata_does_not_mutate_input(cloned_dataset: str) -> None:
    config = _reader_config(cloned_dataset)
    create_dataset(config, ignore_metadata=True)
    assert config == _reader_config(cloned_dataset)


def test_cloned_dataset_without_ignore_metadata_fails(cloned_dataset: str) -> None:
    """Guard that the fixture reproduces the clone problem ignore_metadata works around."""
    with pytest.raises(IndexError):
        create_dataset(_reader_config(cloned_dataset), ignore_metadata=False)


@pytest.mark.skipif(not CLONED_O800.exists(), reason=f"{CLONED_O800} not available on this machine")
def test_ignore_metadata_real_o800_clone() -> None:
    reader = create_dataset(_reader_config(str(CLONED_O800)), ignore_metadata=True)

    assert reader.data.shape[0] == 20000
    assert reader.sequence_length() == 19999  # the clone's dates array is one short
    assert reader.frequency == datetime.timedelta(hours=6)
    assert compute_relative_date_indices(SixHourlyTask(), {"data": reader}) == {"data": [-1, 0, 1]}
