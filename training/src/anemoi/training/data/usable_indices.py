# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import logging
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from anemoi.training.data.data_reader import BaseAnemoiReader

LOGGER = logging.getLogger(__name__)


def compute_valid_data_indices(
    data_readers: dict[str, "BaseAnemoiReader"],
    relative_date_indices: dict[str, np.ndarray | list[int]],
    optional_datasets: list[str] | set[str] | None = None,
) -> np.ndarray:
    """Return valid date indices.

    A date t is valid if we can sample the elements t + i for every
    ``relative_date_index`` i across all *required* data readers.

    Optional readers (listed in ``optional_datasets`` and/or marked with
    ``reader.optional=True``) are excluded from the intersection: their missing
    slots do not remove dates from the training pool. Instead, samples that
    fall on their missing slots come back as NaN tensors and the training step
    is expected to auto-drop those datasets for that batch.

    Returns the intersection of valid indices from all required data readers.
    """
    optional_set = set(optional_datasets or [])
    for name, reader in data_readers.items():
        if getattr(reader, "optional", False):
            optional_set.add(name)

    valid_date_indices_intersection = None
    for dataset_name, ds in data_readers.items():
        valid_date_indices = get_usable_indices(
            ds.missing,
            len(ds.dates),
            relative_date_indices[dataset_name],
            ds.trajectory_ids if ds.has_trajectories else None,
        )

        if dataset_name in optional_set:
            # Compute for logging, but do NOT intersect with the required pool.
            n_missing = len(ds.missing)
            n_valid = len(valid_date_indices)
            LOGGER.info(
                "Optional data reader '%s' has %d valid indices (%d missing slots) — excluded from date intersection.",
                dataset_name,
                n_valid,
                n_missing,
            )
            continue

        if valid_date_indices_intersection is None:
            valid_date_indices_intersection = valid_date_indices
        else:
            valid_date_indices_intersection = np.intersect1d(valid_date_indices_intersection, valid_date_indices)

        if len(valid_date_indices) == 0:
            msg = f"No valid date indices found for data reader '{dataset_name}': {ds}"
            raise ValueError(msg)

        LOGGER.info("Data reader '%s' has %d valid indices", dataset_name, len(valid_date_indices))

    if valid_date_indices_intersection is None:
        msg = (
            "compute_valid_data_indices: no required datasets found — every reader is optional. "
            "At least one dataset must be non-optional to anchor the training date pool."
        )
        raise ValueError(msg)

    if len(valid_date_indices_intersection) == 0:
        msg = "No valid date indices found after intersection across all datasets."
        raise ValueError(msg)

    LOGGER.info("MultiDataset has %d valid indices after intersection.", len(valid_date_indices_intersection))

    return valid_date_indices_intersection


def get_usable_indices(
    missing_indices: set[int],
    series_length: int,
    relative_indices: np.ndarray | list[int],
    trajectory_ids: np.ndarray | None = None,
) -> np.ndarray:
    """Get the usable indices of a series with missing indices.

    Parameters
    ----------
    missing_indices : set[int]
        Set of missing indices in the series.
    series_length : int
        Length of the series.
    relative_indices: np.ndarray | list[int]
        Array of relative indices requested at each index i.
    trajectory_ids: np.ndarray | None
        Array of integers of length series length that indicates which forecast trajectory a time index belongs to.
        When training on analysis: None

    Returns
    -------
    usable_indices : np.array
        Array of usable indices.
    """
    if isinstance(relative_indices, list):
        relative_indices = np.array(relative_indices)

    usable_indices = np.arange(series_length)

    # Restrict to indices where all relative positions are within bounds
    max_offset = int(max(relative_indices))
    min_offset = int(min(relative_indices))
    usable_indices = usable_indices[(usable_indices + min_offset >= 0) & (usable_indices + max_offset < series_length)]

    # Avoid crossing model runs by selecting only relative indices with the same model run id
    if trajectory_ids is not None:
        rel_run = usable_indices[None] + relative_indices[:, None]
        include = (trajectory_ids[rel_run] == trajectory_ids[rel_run[0]]).all(axis=0)
        usable_indices = usable_indices[include]

    # Missing indices
    for i in missing_indices:
        rel_missing = i - relative_indices  # indices which have their relative indices match the missing.
        usable_indices = usable_indices[np.all(usable_indices != rel_missing[:, np.newaxis], axis=0)]

    return usable_indices
