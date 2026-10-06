# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import numpy as np
from pytest_mock import MockFixture

from anemoi.training.data.data_reader import GriddedDataReader
from anemoi.training.data.multidataset import MultiDataset


def test_multidataset_normalizes_relative_time_indices_to_slices(mocker: MockFixture) -> None:
    """Contiguous relative time indices are collapsed to slices; sparse ones are kept as lists."""
    reader = mocker.MagicMock()
    reader.missing = set()
    reader.dates = list(range(20))
    reader.has_trajectories = False
    reader.num_sequences = 1
    positions = np.arange(17, dtype=np.int64)
    reader.compute_anchors.return_value = np.stack([np.zeros_like(positions), positions], axis=1)

    ds = MultiDataset(
        data_readers={"a": reader, "b": reader},
        relative_date_indices={"a": [0, 1, 2], "b": [0, 2, 3]},
    )

    assert ds.relative_date_indices["a"] == slice(0, 3, 1)
    assert ds.relative_date_indices["b"] == [0, 2, 3]


def test_gridded_reader_passes_time_indices_through_to_dataset() -> None:
    """GriddedDataReader.get_data forwards time indices to the dataset's time axis unchanged."""

    class FakeDataset:
        def __init__(self) -> None:
            self.last_index = None

        def __getitem__(self, item: object) -> np.ndarray:
            self.last_index = item
            return np.zeros((3, 2, 4, 5), dtype=np.float32)

    reader = GriddedDataReader.__new__(GriddedDataReader)
    reader.data = FakeDataset()
    reader.grid_shard_slice = None

    full = (slice(None), slice(None), slice(None))

    reader.get_data(0, [4, 5, 7])
    assert reader.data.last_index == ([4, 5, 7], *full)

    reader.get_data(0, slice(4, 7, 1))
    assert reader.data.last_index == (slice(4, 7, 1), *full)

    reader.grid_shard_slice = slice(0, 2)
    x = reader.get_data(0, slice(4, 7, 1))
    assert reader.data.last_index == (slice(4, 7, 1), slice(None), slice(None), slice(0, 2))
    # (dates, variables, ensemble, grid) -> (dates, ensemble, grid, variables)
    assert tuple(x.shape) == (3, 4, 5, 2)
