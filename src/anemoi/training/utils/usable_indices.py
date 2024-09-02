from __future__ import annotations

import numpy as np


def get_usable_indices(
    missing_indices: set[int] | None,
    series_length: int,
    rollout: int,
    multistep: int,
    timeincrement: int = 1,
) -> np.ndarray:
    """Get the usable indices of a series whit missing indices.
    Parameters
    ----------
    missing_indices : set[int]
        Dataset to be used.
    series_length : int
        Length of the series.
    rollout : int
        Number of steps to roll out.
    multistep : int
        Number of previous indices to include as predictors.
    timeincrement : int
        Time increment, by default 1.
    Returns
    -------
    usable_indices : np.array
        Array of usable indices.
    """
    prev_invalid_dates = (multistep - 1) * timeincrement
    next_invalid_dates = rollout * timeincrement

    usable_indices = np.arange(series_length)  # set of all indices

    # No missing indices
    if missing_indices is None:
        return usable_indices[prev_invalid_dates : series_length - next_invalid_dates]

    missing_indices |= {-1, len(missing_indices)}  # to filter initial and final indices

    # Missing indices
    for i in missing_indices:
        usable_indices = usable_indices[(usable_indices < i - next_invalid_dates) + (usable_indices > i + prev_invalid_dates)]

    return usable_indices
