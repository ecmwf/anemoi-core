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

import numpy as np
import torch
from scipy.spatial import ConvexHull
from scipy.spatial import SphericalVoronoi
from scipy.spatial import Voronoi
from torch_geometric.data.storage import NodeStorage

from anemoi.graphs.generate.transforms import latlon_rad_to_cartesian
from anemoi.graphs.nodes.attributes.base_attributes import BaseNodeAttribute

LOGGER = logging.getLogger(__name__)


class UniformWeights(BaseNodeAttribute):
    """Implements a uniform weight for the nodes.

    Methods
    -------
    compute(self, graph, nodes_name)
        Compute the area attributes for each node.
    """

    def get_raw_values(self, nodes: NodeStorage, **kwargs) -> torch.Tensor:
        """Compute the weights.

        Parameters
        ----------
        nodes : NodeStorage
            Nodes of the graph.
        kwargs : dict
            Additional keyword arguments.

        Returns
        -------
        torch.Tensor
            Ones.
        """
        return torch.ones((nodes.num_nodes,))


class AreaWeights(BaseNodeAttribute):
    """Implements the area of the nodes as the weights.

    Attributes
    ----------
    flat: bool
        If True, the area is computed in 2D, otherwise in 3D.
    **other: Any
        Additional keyword arguments, see PlanarAreaWeights and SphericalAreaWeights
        for details.

    Methods
    -------
    compute(self, graph, nodes_name)
        Compute the area attributes for each node.
    """

    def __new__(cls, flat: bool = False, **kwargs):
        logging.warning(
            "Creating %s with flat=%s and kwargs=%s. In a future release, AreaWeights will be deprecated: please use directly PlanarAreaWeights or SphericalAreaWeights.",
            cls.__name__,
            flat,
            kwargs,
        )
        if flat:
            return PlanarAreaWeights(**kwargs)
        return SphericalAreaWeights(**kwargs)


class PlanarAreaWeights(BaseNodeAttribute):
    """Implements the 2D area of the nodes as the weights.

    Attributes
    ----------
    norm : str
        Normalisation of the weights.

    Methods
    -------
    compute(self, graph, nodes_name)
        Compute the area attributes for each node.
    """

    def _compute_mean_nearest_distance(self, points: np.ndarray) -> float:
        """Compute mean distance to nearest neighbor for each point.

        Parameters
        ----------
        points : np.ndarray
            Array of point coordinates (N x 2)

        Returns
        -------
        float
            Mean nearest neighbor distance
        """
        from scipy.spatial import cKDTree

        tree = cKDTree(points)
        distances, _ = tree.query(points, k=2)
        return float(distances[:, 1].mean())

    def _get_boundary_ring(self, points: np.ndarray, resolution: float) -> np.ndarray:
        """Add a ring of boundary points around the input points.

        Parameters
        ----------
        points : np.ndarray
            Original point coordinates
        resolution : float
            Approximate spacing between points

        Returns
        -------
        np.ndarray
            Array including original and boundary points
        """
        # Get convex hull vertices
        hull = ConvexHull(points)
        hull_points = points[hull.vertices]

        # Expand hull slightly (resolution) outward
        centroid = np.mean(hull_points, axis=0)
        vectors = hull_points - centroid
        expanded_hull = hull_points + vectors * (resolution / np.linalg.norm(vectors, axis=1)[:, np.newaxis])

        # Create points along each hull edge
        boundary_points = []
        n = len(expanded_hull)
        for i in range(n):
            p1 = expanded_hull[i]
            p2 = expanded_hull[(i + 1) % n]

            # Calculate number of points needed along this edge
            edge_length = np.linalg.norm(p2 - p1)
            num_points = max(2, int(edge_length / resolution))

            # Create evenly spaced points along the edge
            t = np.linspace(0, 1, num_points)[:-1]  # Exclude last point to avoid duplicates
            edge_points = p1[None, :] + t[:, None] * (p2 - p1)[None, :]
            boundary_points.append(edge_points)

        boundary_points = np.vstack(boundary_points)

    def get_raw_values(self, nodes: NodeStorage, **kwargs) -> torch.Tensor:
        points = nodes.x.cpu().numpy()
        resolution = self._compute_mean_nearest_distance(points)
        extended_points = self._add_boundary_ring(points, resolution)

        v = Voronoi(extended_points, qhull_options="QJ Pp")
        areas = []
        for r in v.regions:
            area = ConvexHull(v.vertices[r, :]).volume
            areas.append(area)
        result = torch.from_numpy(np.asarray(areas))
        return result


class SphericalAreaWeights(BaseNodeAttribute):
    """Implements the 3D area of the nodes as the weights.

    Attributes
    ----------
    norm : str
        Normalisation of the weights.
    radius : float
        Radius of the sphere.
    centre : np.ndarray
        Centre of the sphere.
    fill_value : float
        Value to fill the empty regions.

    Methods
    -------
    compute(self, graph, nodes_name)
        Compute the area attributes for each node.
    """

    def __init__(
        self,
        norm: str | None = None,
        radius: float = 1.0,
        centre: np.ndarray = np.array([0, 0, 0]),
        fill_value: float = 0.0,
        dtype: str = "float32",
    ) -> None:
        assert isinstance(fill_value, float | int), f"fill_value must be float or nan but it is {type(fill_value)}"
        assert isinstance(radius, float | int) and radius > 0, f"radius must be a positive value, but radius={radius}"
        super().__init__(norm, dtype)
        self.radius = radius
        self.centre = centre
        self.fill_value = fill_value

    def get_raw_values(self, nodes: NodeStorage, **kwargs) -> np.ndarray:
        """Compute the area associated to each node.

        It uses Voronoi diagrams to compute the area of each node.

        Parameters
        ----------
        nodes : NodeStorage
            Nodes of the graph.
        kwargs : dict
            Additional keyword arguments.

        Returns
        -------
        np.ndarray
            Attributes.
        """
        points = latlon_rad_to_cartesian(nodes.x.cpu())
        sv = SphericalVoronoi(points, self.radius, self.centre)
        mask = np.array([bool(i) for i in sv.regions])
        sv.regions = [region for region in sv.regions if region]
        # compute the area weight without empty regions
        area_weights = sv.calculate_areas()
        if (null_nodes := (~mask).sum()) > 0:
            LOGGER.warning(
                "%s is filling %d (%.2f%%) nodes with value %f",
                self.__class__.__name__,
                null_nodes,
                100 * null_nodes / len(mask),
                self.fill_value,
            )
        result = np.ones(points.shape[0]) * self.fill_value
        result[mask] = area_weights
        LOGGER.debug(
            "There are %d of weights, which (unscaled) add up a total weight of %.2f.",
            len(result),
            result.sum(),
        )
        return torch.from_numpy(result)
