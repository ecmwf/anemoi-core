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
    """Load the normalisation statistics of a dataset's residuals from an ``.npz`` file.

    A residual is a predicted variable minus its reference from another dataset (for example a
    high-resolution field minus the projected low-resolution one). The model normalises it with these
    statistics, the same for every output step.

    Parameters
    ----------
    path : str
        Path to an ``.npz`` file with ``mean``, ``minimum``, ``maximum`` and ``stdev`` entries, each a
        0-d object array holding a ``{variable_name: float}`` mapping.
    variables : list[str]
        The dataset's variables, in the order of its full variable index.

    Returns
    -------
    dict[str, np.ndarray]
        ``{"mean": ..., "minimum": ..., "maximum": ..., "stdev": ...}``, each of shape
        ``(len(variables),)`` and ordered like ``variables``. A variable without an entry in the file
        gets NaN: the model checks that the variables it learns as residuals have finite values.
    """
    with np.load(path, allow_pickle=True) as npz:
        missing_keys = [key for key in RESIDUAL_STATISTICS_KEYS if key not in npz.files]
        if missing_keys:
            msg = f"Residual statistics in {path} lack the entries {missing_keys}."
            raise KeyError(msg)
        stats_by_name = {key: npz[key].item() for key in RESIDUAL_STATISTICS_KEYS}

    return {
        key: np.array([stats_by_name[key].get(variable, np.nan) for variable in variables], dtype=np.float64)
        for key in RESIDUAL_STATISTICS_KEYS
    }
