# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Normalisation methods, shared by the normaliser and by everything that normalises physical values.

:class:`~anemoi.models.preprocessing.normalizer.InputNormalizer` normalises data with these methods
(:func:`normalization_coefficients`);
:class:`~anemoi.models.preprocessing.postprocessor.NormalizedReluPostprocessor` and
:class:`~anemoi.models.layers.bounding.NormalizedReluBounding` normalise their physical thresholds
with them (:func:`normalize_value`, after :func:`remapped_statistics`). A new method is added here:
in :data:`NORMALIZATION_STATISTICS`, :func:`normalization_coefficients` and :func:`normalize_value`.
"""

import warnings
from collections.abc import Mapping
from collections.abc import Sequence

import numpy as np
import torch

#: Statistics each normalisation method needs, by method name. The keys are those of
#: :attr:`~anemoi.models.data.sources.Source.statistics`, i.e. of the anemoi-datasets statistics.
NORMALIZATION_STATISTICS: dict[str, tuple[str, ...]] = {
    "mean-std": ("mean", "stdev"),
    "std": ("stdev",),
    "min-max": ("minimum", "maximum"),
    "max": ("maximum",),
    "none": (),
}

#: Scales (standard deviation, range, maximum) below this are not divided by.
EPS = 1e-8


def check_normalization_method(method: str, owner: str) -> None:
    """Raise a ValueError unless ``method`` is a known normalisation method.

    Parameters
    ----------
    method : str
        Normalisation method name.
    owner : str
        Name of the caller, for the error message.
    """
    if method not in NORMALIZATION_STATISTICS:
        msg = f"{owner}: unknown normalisation method {method!r}; expected one of {list(NORMALIZATION_STATISTICS)}."
        raise ValueError(msg)


def missing_statistics(methods: Sequence[str], statistics: Mapping) -> list[str]:
    """Return the statistics ``methods`` need that ``statistics`` lacks, sorted.

    Parameters
    ----------
    methods : Sequence[str]
        Normalisation method names.
    statistics : Mapping
        Available statistics, by key.

    Returns
    -------
    list[str]
        The missing statistic keys.
    """
    required = {key for method in methods for key in NORMALIZATION_STATISTICS[method]}
    return sorted(required - set(statistics))


def normalization_coefficients(method: str, statistics: Mapping, index: int, name: str = "") -> tuple[float, float]:
    """Return ``(mul, add)`` such that ``value * mul + add`` normalises a value of the variable at ``index``.

    A near-zero scale is not divided by (with a warning), as in the normaliser.

    Parameters
    ----------
    method : str
        Normalisation method name, see :data:`NORMALIZATION_STATISTICS`.
    statistics : Mapping
        Statistics by key, one value per variable (numpy arrays or tensors).
    index : int
        Position of the variable in the statistics.
    name : str, optional
        Variable name, for the warnings.

    Returns
    -------
    tuple[float, float]
        The scale and offset, in float64.
    """
    check_normalization_method(method, "normalization_coefficients")
    stat = {key: float(statistics[key][index]) for key in NORMALIZATION_STATISTICS[method]}

    if method == "mean-std":
        if stat["stdev"] < EPS:
            warnings.warn(f"Variable {name} has near-zero variance. Skipping scale adjustments.")
            return 1.0, 0.0
        return 1.0 / stat["stdev"], -stat["mean"] / stat["stdev"]
    if method == "std":
        if stat["stdev"] < EPS:
            warnings.warn(f"Variable {name} has near-zero variance. Skipping scale adjustments.")
            return 1.0, 0.0
        return 1.0 / stat["stdev"], 0.0
    if method == "min-max":
        value_range = stat["maximum"] - stat["minimum"]
        if value_range < EPS:
            warnings.warn(f"Variable {name} has a near-zero range. Skipping scale adjustments.")
            return 1.0, -stat["minimum"]
        return 1.0 / value_range, -stat["minimum"] / value_range
    if method == "max":
        if abs(stat["maximum"]) < EPS:
            warnings.warn(f"Variable {name} has a near-zero maximum. Skipping scale adjustments.")
            return 1.0, 0.0
        return 1.0 / stat["maximum"], 0.0
    return 1.0, 0.0


def normalize_value(method: str, value: float, statistics: Mapping, index: int) -> float:
    """Normalise one physical ``value`` of the variable at ``index``, e.g. a threshold.

    The formulas (``(value - mean) / stdev`` and so on, with no near-zero fallback) are those the
    thresholds have always used, so thresholds stay bit-identical to earlier releases.

    Parameters
    ----------
    method : str
        Normalisation method name, see :data:`NORMALIZATION_STATISTICS`.
    value : float
        Physical value.
    statistics : Mapping
        Statistics by key, one value per variable (numpy arrays or tensors).
    index : int
        Position of the variable in the statistics.

    Returns
    -------
    float
        The normalised value.
    """
    check_normalization_method(method, "normalize_value")
    stat = {key: statistics[key][index] for key in NORMALIZATION_STATISTICS[method]}
    if method == "mean-std":
        return (value - stat["mean"]) / stat["stdev"]
    if method == "std":
        return value / stat["stdev"]
    if method == "min-max":
        return (value - stat["minimum"]) / (stat["maximum"] - stat["minimum"])
    if method == "max":
        return value / stat["maximum"]
    return value


def remapped_statistics(
    statistics: Mapping, name_to_index: Mapping[str, int], remap: Mapping[str, str], owner: str
) -> dict:
    """Return a copy of ``statistics`` where each variable of ``remap`` takes the statistics of its source.

    This is the normaliser's ``remap`` (e.g. ``cp: tp``, to normalise ``cp`` with the statistics of
    ``tp``). Every remapped variable takes the source's original statistics, so the result does not
    depend on the order of the mapping. ``statistics`` itself is not changed.

    Parameters
    ----------
    statistics : Mapping
        Statistics by key, one value per variable of the view.
    name_to_index : Mapping[str, int]
        Positions of the view's variables in the statistics.
    remap : Mapping[str, str]
        Variable to source variable.
    owner : str
        Name of the caller, for the error message.

    Returns
    -------
    dict
        The statistics, remapped.
    """
    pairs = [(name, source) for name, source in remap.items() if name in name_to_index]
    if not pairs:
        return dict(statistics)
    for name, source in pairs:
        if source not in name_to_index:
            msg = f"{owner}: cannot use the statistics of {source!r} for {name!r}: {source!r} is not in this view."
            raise ValueError(msg)
    targets = [name_to_index[name] for name, _ in pairs]
    sources = [name_to_index[source] for _, source in pairs]
    remapped = {}
    for key, value in statistics.items():
        if getattr(value, "ndim", 0) != 1 or len(value) != len(name_to_index):
            remapped[key] = value  # not one value per variable: nothing to remap
            continue
        value = value.clone() if torch.is_tensor(value) else np.array(value, copy=True)
        value[targets] = value[sources]
        remapped[key] = value
    return remapped
