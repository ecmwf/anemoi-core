# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


from __future__ import annotations

import numpy as np


def get_usable_indices(
    missing_indices: set[int],
    series_length: int,
    relative_indices: np.ndarray,
    model_run_ids: np.ndarray | None = None,
) -> np.ndarray:
    """Get the usable indices of a series with missing indices.

    Parameters
    ----------
    missing_indices : set[int]
        Dataset to be used.
    series_length : int
        Length of the series.
    relative_indices: array[np.int64]
        Array of relative indices requested at each index i.
    model_run_ids: array[np.int64]
        Array of integers of length series length that contains the id of a model run.
        When training on analysis: None

    Returns
    -------
    usable_indices : np.array
        Array of usable indices.
    """
    usable_indices = np.arange(series_length - max(relative_indices))

    # Avoid crossing model runs by selecting only relative indices with the same model run id
    if model_run_ids is not None:
        rel_run = usable_indices[None] + relative_indices[:, None]
        include = (model_run_ids[rel_run] == model_run_ids[rel_run[0]]).all(axis=0)
        usable_indices = usable_indices[include]

    # Missing indices
    for i in missing_indices:
        rel_missing = i - relative_indices #indices which have their relative indices match the missing.
        usable_indices = usable_indices[np.all(usable_indices != rel_missing[:,np.newaxis], axis = 0)]

    return usable_indices