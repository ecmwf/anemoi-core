# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Per-dataset view of the supporting arrays stored beside the metadata.

Checkpoints store numpy arrays (coordinates and masks) next to the metadata
JSON, listed under ``supporting_arrays_paths``.  Files written for format 2.0
and later key them by dataset name (``{dataset: {latitudes: ..., ...}}``);
older single-dataset files use a flat mapping (``{latitudes: ..., ...}``),
which belongs to the one dataset of the checkpoint.

:class:`DatasetArrays` is the public, read-only view of the arrays of one
dataset.  Every array it returns is a copy, so changing it never changes the
:class:`~anemoi.metadata.interface.Metadata` it came from.  A dataset without
arrays (a tabular dataset, for example) gives ``None`` for each of them.
"""

import logging
from collections.abc import Iterator
from collections.abc import Mapping
from typing import Any
from typing import ClassVar

import numpy as np

LOG = logging.getLogger(__name__)


def _copy_arrays(arrays: Mapping[str, Any], prefix: str = "") -> dict[str, np.ndarray]:
    """Return a flat ``{name: array copy}`` of *arrays*; nested mappings get ``/``-joined names."""
    flat: dict[str, np.ndarray] = {}
    for key, value in arrays.items():
        name = f"{prefix}{key}"
        if isinstance(value, Mapping):
            flat.update(_copy_arrays(value, f"{name}/"))
        else:
            flat[name] = np.array(value, copy=True)
    return flat


def split_by_dataset(
    supporting_arrays: Mapping[str, Any] | None,
    dataset_names: list[str],
) -> dict[str, dict[str, np.ndarray]]:
    """Assign the supporting arrays of a checkpoint to its datasets.

    Parameters
    ----------
    supporting_arrays : Mapping[str, Any] | None
        Arrays as returned by ``load_metadata(path, supporting_arrays=True)``:
        keyed by dataset name, or a flat mapping of arrays for single-dataset
        checkpoints.  ``None`` means none.
    dataset_names : list[str]
        The dataset names of the checkpoint.

    Returns
    -------
    dict[str, dict[str, numpy.ndarray]]
        Copies of the arrays, by dataset name; a dataset without arrays is not
        in the result.  A flat mapping is assigned to the only dataset; with
        several datasets it cannot be assigned, so it is dropped with a warning
        (never given to the first dataset).
    """
    if not supporting_arrays:
        return {}

    by_name = {
        name: _copy_arrays(supporting_arrays[name])
        for name in dataset_names
        if isinstance(supporting_arrays.get(name), Mapping)
    }
    if by_name:
        return by_name

    if len(dataset_names) == 1:
        return {dataset_names[0]: _copy_arrays(supporting_arrays)}

    LOG.warning(
        "Supporting arrays %s are not keyed by dataset name and the checkpoint has %d datasets: ignoring them.",
        sorted(supporting_arrays),
        len(dataset_names),
    )
    return {}


def merge_supporting_arrays(
    current: Mapping[str, Mapping[str, np.ndarray]],
    overrides: Mapping[str, Mapping[str, np.ndarray]],
) -> dict[str, dict[str, np.ndarray]]:
    """Return *current* with the arrays of *overrides* replaced or added, per dataset.

    Parameters
    ----------
    current : Mapping[str, Mapping[str, numpy.ndarray]]
        Arrays by dataset name.
    overrides : Mapping[str, Mapping[str, numpy.ndarray]]
        Arrays to replace or add, by dataset name.  A dataset not in *current*
        gets exactly these arrays.

    Returns
    -------
    dict[str, dict[str, numpy.ndarray]]
        Copies; arrays and datasets not named in *overrides* are kept.
    """
    merged = {name: _copy_arrays(arrays) for name, arrays in current.items()}
    for name, arrays in overrides.items():
        merged.setdefault(name, {}).update(_copy_arrays(arrays))
    return merged


class DatasetArrays:
    """Read-only view of the supporting arrays of one dataset.

    Parameters
    ----------
    arrays : Mapping[str, numpy.ndarray] | None
        The arrays of the dataset by name; ``None`` or empty when it has none.
        They are held as given (the caller passes copies); the view hands out
        a new copy on every access.
    version : str | None, optional
        Version of the dataset's data component (for example ``"0.2"``), as
        read by :func:`~anemoi.metadata.components.base.requires` to gate
        members on it.  ``None`` (the default) when unknown: a gated member
        then always raises :class:`~anemoi.metadata.exceptions.VersionError`.
        :class:`~anemoi.metadata.Metadata` sets it.

    Attributes
    ----------
    component : str
        Name of the component the arrays belong to: ``"data"``.
    version : str | None
        The data component version that ``@requires`` checks.

    Notes
    -----
    ``latitudes``, ``longitudes``, ``grid_indices`` and ``output_mask`` are
    the arrays inference reads by name; every other array is a *named mask*
    (:attr:`masks`).  Each is ``None`` when the dataset does not have it.
    """

    component: ClassVar[str] = "data"

    def __init__(self, arrays: Mapping[str, np.ndarray] | None = None, *, version: str | None = None) -> None:
        self._arrays: dict[str, np.ndarray] = dict(arrays or {})
        self.version = version

    def get(self, name: str) -> np.ndarray | None:
        """Return a copy of the array *name*, or ``None`` if the dataset has none."""
        array = self._arrays.get(name)
        return None if array is None else array.copy()

    @property
    def names(self) -> list[str]:
        """Names of all arrays of the dataset."""
        return list(self._arrays)

    @property
    def latitudes(self) -> np.ndarray | None:
        """Latitudes of the grid points, or ``None``."""
        return self.get("latitudes")

    @property
    def longitudes(self) -> np.ndarray | None:
        """Longitudes of the grid points, or ``None``."""
        return self.get("longitudes")

    @property
    def grid_indices(self) -> np.ndarray | None:
        """Indices of the grid points kept from the source grid (limited-area models), or ``None``."""
        # To consider within the context of the deprecation of grid_indices
        return self.get("grid_indices")

    @property
    def output_mask(self) -> np.ndarray | None:
        """Mask of the grid points that are output, or ``None``."""
        return self.get("output_mask")

    @property
    def masks(self) -> dict[str, np.ndarray]:
        """The named masks: every array other than coordinates, ``grid_indices`` and ``output_mask``."""
        reserved = {"latitudes", "longitudes", "grid_indices", "output_mask"}
        return {name: array.copy() for name, array in self._arrays.items() if name not in reserved}

    def __contains__(self, name: object) -> bool:
        return name in self._arrays

    def __iter__(self) -> Iterator[str]:
        return iter(self._arrays)

    def __len__(self) -> int:
        return len(self._arrays)

    def __repr__(self) -> str:
        return f"DatasetArrays({self.names!r})"
