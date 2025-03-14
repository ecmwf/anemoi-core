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
    valid_indices = get_usable_indices(missing_indices=None, series_length=10, rollout=1, multistep=1, timeincrement=1)
    expected_values = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8])
    assert np.allclose(valid_indices, expected_values)

    # Test multiple steps inputs
    valid_indices = get_usable_indices(missing_indices=None, series_length=10, rollout=1, multistep=2, timeincrement=1)
    expected_values = np.array([1, 2, 3, 4, 5, 6, 7, 8])
    assert np.allclose(valid_indices, expected_values)

    # Test roll out
    valid_indices = get_usable_indices(missing_indices=None, series_length=10, rollout=2, multistep=1, timeincrement=1)
    expected_values = np.array([0, 1, 2, 3, 4, 5, 6, 7])
    assert np.allclose(valid_indices, expected_values)

    # Test longer time increments
    valid_indices = get_usable_indices(missing_indices=None, series_length=10, rollout=1, multistep=2, timeincrement=2)
    expected_values = np.array([2, 3, 4, 5, 6, 7])
    assert np.allclose(valid_indices, expected_values)

    # Test missing indices
    missing_indices = {7, 5}
    valid_indices = get_usable_indices(
        missing_indices=missing_indices,
        series_length=10,
        rollout=1,
        multistep=2,
        timeincrement=1,
    )
    expected_values = np.array([1, 2, 3])
    assert np.allclose(valid_indices, expected_values)
