# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import logging
from abc import ABC
from abc import abstractmethod
from collections.abc import Sequence
from functools import cached_property

import numpy as np

LOGGER = logging.getLogger(__name__)

ArrayIndex = slice | int | Sequence[int] | None


class BaseIndices(ABC):
    """Base class for custom indices."""

    @property
    @abstractmethod
    def supporting_arrays(self) -> dict:
        raise NotImplementedError

    @abstractmethod
    def get_shard_indices(self, grid_shard_indices: np.ndarray | None = None) -> ArrayIndex:
        """Get shard indices."""
        raise NotImplementedError


class FullGrid(BaseIndices):
    """Class for full indices."""

    @property
    def supporting_arrays(self) -> dict:
        return {}

    def get_shard_indices(self, grid_shard_indices: np.ndarray | None = None) -> ArrayIndex:
        """Get shard indices."""
        return grid_shard_indices


class MaskedGrid(BaseIndices):
    """Class for masked indices."""

    def __init__(
        self,
        latitudes: np.ndarray,
        longitudes: np.ndarray,
        mask: np.ndarray,
        mask_radius_km: float,
    ):
        self.mask_radius = mask_radius_km / 6371.0
        self.latitudes = latitudes
        self.longitudes = longitudes
        self.mask = mask

    @cached_property
    def coords_3d(self) -> np.ndarray:
        """3D coordinates on a unit sphere."""
        lat_rad = np.radians(self.latitudes)
        lon_rad = np.radians(self.longitudes)

        x = np.cos(lat_rad) * np.cos(lon_rad)
        y = np.cos(lat_rad) * np.sin(lon_rad)
        z = np.sin(lat_rad)

        return np.vstack((x, y, z)).T

    @cached_property
    def grid_indices(self) -> np.ndarray:
        from scipy.spatial import cKDTree

        # Check which points are within the radius of any mask point
        tree = cKDTree(self.coords_3d[self.mask])
        dists, _ = tree.query(self.coords_3d[~self.mask], k=1, distance_upper_bound=self.mask_radius)

        grid_mask = self.mask.copy()  # points inside the mask are always included
        grid_mask[~self.mask] = np.isfinite(dists)
        return np.where(grid_mask)[0].astype(np.int64)

    @property
    def supporting_arrays(self) -> dict:
        return {"grid_indices": self.grid_indices}

    def get_shard_indices(self, grid_shard_indices: np.ndarray | None = None) -> ArrayIndex:
        """Get shard indices."""
        return self.grid_indices[grid_shard_indices]
