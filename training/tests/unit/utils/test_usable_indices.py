# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import numpy as np

from anemoi.training.utils.usable_indices import get_usable_indices


def test_get_usable_indices() -> None:
    """Test get_usable_indices function."""
    # Test base case
    valid_indices = get_usable_indices(missing_indices=set(), series_length=10, relative_indices=np.array([0, 1]))
    expected_values = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8])
    assert np.allclose(valid_indices, expected_values)

    # Test 3 indices, either from rollout = 1 and multistep = 2, or rollout = 2 and multistep = 1
    valid_indices = get_usable_indices(missing_indices=set(), series_length=10, relative_indices=np.array([0, 1, 2]))
    expected_values = np.array([0, 1, 2, 3, 4, 5, 6, 7])
    assert np.allclose(valid_indices, expected_values)

    # With time increment
    valid_indices = get_usable_indices(missing_indices=set(), series_length=10, relative_indices=np.array([0, 2, 4]))
    expected_values = np.array([0, 1, 2, 3, 4, 5])
    assert np.allclose(valid_indices, expected_values)

    # Test missing indices with standard setup
    missing_indices = {7, 5, 14}
    valid_indices = get_usable_indices(
        missing_indices=missing_indices,
        series_length=20,
        relative_indices=np.array([0, 1, 2]),
    )
    expected_values = np.array([0, 1, 2, 8, 9, 10, 11, 15, 16, 17])
    assert np.allclose(valid_indices, expected_values)

    # Now verify that with a non-standard setup, missing indices can be "jumped" over.
    valid_indices = get_usable_indices(
        missing_indices=missing_indices,
        series_length=20,
        relative_indices=np.array([0, 5, 6, 7]),  # e.g making a 1 hour forecast based on -6, -1 and 0 hours.
    )
    expected_values = np.array([3, 4, 6, 10, 11, 12])
    assert np.allclose(valid_indices, expected_values)

    # Test functionality for avoiding using different model trajectories
    series_length = 70
    mr_start, mr_length = [4, 18]
    trajectory_ids = (np.arange(series_length, dtype=np.int64) - mr_start) // mr_length
    valid_indices = get_usable_indices(
        missing_indices=set(range(40, 58)) | {11},  # one model run of length 18 missing, and one sample of another run.
        series_length=series_length,
        relative_indices=np.array([0, 3, 6]),
        trajectory_ids=trajectory_ids,
    )
    expected_values = np.array([*[4, 6, 7, 9, 10, 12, 13, 14, 15], *list(range(22, 40 - 6)), *[58, 59, 60, 61, 62, 63]])
