# (C) Copyright 2024- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from __future__ import annotations

import logging
from typing import Tuple

import numpy as np
from torch_geometric.data.storage import NodeStorage

from anemoi.datasets import open_dataset
from anemoi.graphs.generate.transforms import cartesian_to_latlon_rad
from anemoi.graphs.generate.transforms import latlon_rad_to_cartesian
from anemoi.graphs.nodes.attributes.base_attributes import FloatBaseNodeAttribute
from anemoi.graphs.utils import haversine_distance

LOGGER = logging.getLogger(__name__)


class DistanceFromPointAttribute(FloatBaseNodeAttribute):
    """
    Attribute containing distance of each point from a specified point.
    """

    def __init__(self, dst_point: str | Tuple[float, float] = "lam_center") -> None:
        super().__init__(norm=None)

        if dst_point == "lam_center":
            self.from_lam = True

        else:
            self.from_lam = False
            self.dst_point = dst_point

    def central_point_on_sphere(self, latlons: np.ndarray) -> np.ndarray:
        """
        Compute the central point of a set of points on a sphere.

        Parameters
        ----------
        latlons : np.ndarray
            Array of shape (N, 2) containing latitude and longitude in radians.

        Returns
        -------
        np.ndarray
            Central point (latitude, longitude) in radians.
        """
        latlons_np = latlons.numpy()  # Convert torch tensor to numpy array
        xyz = latlon_rad_to_cartesian((latlons_np[:, 0], latlons_np[:, 1]))

        # Compute mean of Cartesian coordinates
        mean_xyz = xyz.mean(axis=0)

        # Normalize to project back onto the sphere
        mean_xyz /= np.linalg.norm(mean_xyz)

        # Convert back to lat-lon
        return cartesian_to_latlon_rad(mean_xyz[np.newaxis, :])[0]

    def get_raw_values(self, nodes: NodeStorage, **kwargs) -> np.ndarray:

        distances = np.zeros(nodes.x.shape[0])

        if self.from_lam:
            assert isinstance(nodes["_dataset"], dict), "The 'dataset' attribute must be a dictionary."
            assert "cutout" in nodes["_dataset"], "The 'dataset' attribute must contain a 'cutout' key."
            num_lam, _ = open_dataset(nodes["_dataset"]).grids

            coords = nodes.x[num_lam:, :]
            dst_point = self.central_point_on_sphere(nodes.x[:num_lam])

        else:
            coords = nodes.x
            dst_point = self.dst_point

        # Compute distances for all points from the central point
        for i, p in enumerate(coords):
            distances[num_lam + i] = haversine_distance(dst_point, p)

        return distances
