# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import numpy as np

from anemoi.training.data.data_reader import NativeGridDataset


def test_get_sample_normalizes_time_indices_before_dataset_access() -> None:
    """Test that NativeGridDataset.get_sample normalizes time indices to slices before accessing the dataset."""

    class FakeDataset:
        def __init__(self) -> None:
            self.last_index = None

        def __getitem__(self, item: int) -> np.ndarray:
            self.last_index = item
            return np.zeros((3, 2, 4, 5), dtype=np.float32)

    dataset = NativeGridDataset.__new__(NativeGridDataset)
    dataset.data = FakeDataset()

    dataset.get_sample(time_indices=[4, 5, 6], grid_shard_indices=slice(0, 5))

    time_index = dataset.data.last_index[0]
    assert isinstance(time_index, list)

    dataset.get_sample(time_indices=slice(4, 7, 1), grid_shard_indices=slice(0, 5))

    time_index = dataset.data.last_index[0]
    assert isinstance(time_index, slice)
    assert (time_index.start, time_index.stop, time_index.step) == (4, 7, 1)
