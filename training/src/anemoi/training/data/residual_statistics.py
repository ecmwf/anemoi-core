# (C) Copyright 2025-2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from __future__ import annotations

import numpy as np

RESIDUAL_STATISTICS_KEYS = ("mean", "minimum", "maximum", "stdev")


def load_residual_statistics(path: str, variables: list[str]) -> dict[str, np.ndarray]:
    """Load precomputed residual normalization statistics from an ``.npz`` file.

    Parameters
    ----------
    path : str
        Path to an ``.npz`` file with ``mean``, ``minimum``, ``maximum`` and
        ``stdev`` entries, each a 0-d object array wrapping a
        ``{variable_name: float}`` mapping.
    variables : list[str]
        Variable names, ordered to match the target dataset's full variable index.

    Returns
    -------
    dict[str, np.ndarray]
        ``{"mean": ..., "minimum": ..., "maximum": ..., "stdev": ...}``, each an
        array of shape ``(len(variables),)`` ordered like ``variables``.

    Raises
    ------
    KeyError
        If a variable in ``variables`` is missing from one of the statistics.
    ValueError
        If any loaded value is not finite.
    """
    npz = np.load(path, allow_pickle=True)
    stats_by_name = {key: npz[key].item() for key in RESIDUAL_STATISTICS_KEYS}

    statistics = {
        key: np.array([stats_by_name[key][variable] for variable in variables], dtype=np.float32)
        for key in RESIDUAL_STATISTICS_KEYS
    }

    for key, values in statistics.items():
        non_finite = [variable for variable, ok in zip(variables, np.isfinite(values), strict=True) if not ok]
        if non_finite:
            msg = f"Residual statistics '{key}' loaded from {path} are not finite for variables: {non_finite}"
            raise ValueError(msg)

    return statistics
