# (C) Copyright 2024- Anemoi contributors.
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

import numpy as np
import torch
from scipy.spatial import ConvexHull
from scipy.spatial import SphericalVoronoi
from scipy.spatial import Voronoi
from torch_geometric.data.storage import NodeStorage

from anemoi.graphs import EARTH_RADIUS
from anemoi.graphs.generate.transforms import latlon_rad_to_cartesian_np
from anemoi.graphs.nodes.attributes.base_attributes import BaseNodeAttribute

LOGGER = logging.getLogger(__name__)


class BaseAreaWeights(BaseNodeAttribute, ABC):
    """Base class for area weights of the nodes."""

    def get_latlon_coordinates(self, nodes: NodeStorage) -> torch.Tensor:
        return nodes.x.to(torch.float64)

    @abstractmethod
    def compute_area_weights(self, latlons: np.ndarray) -> np.ndarray: ...

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
            Area weights for the nodes.
        """
        latlons = self.get_latlon_coordinates(nodes).cpu().numpy()
        area_weights = self.compute_area_weights(latlons)
        return torch.from_numpy(area_weights)


class UniformWeights(BaseAreaWeights):
    """Implements a uniform weight for the nodes.

    Methods
    -------
    compute(self, graph, nodes_name)
        Compute the area attributes for each node.
    """

    def compute_area_weights(self, latlons: np.ndarray) -> np.ndarray:
        """Compute area weights.

        Parameters
        ----------
        latlons : np.ndarray
            2D array of shape (N, 2) with latitude and longitude coordinates of the
            nodes of the graph.

        Returns
        -------
        np.ndarray
            Ones.
        """
        return np.ones(latlons.shape[0])


class PlanarAreaWeights(BaseAreaWeights):
    """Planar area weights

    It computes the area in a 2D plane asociated to each node.

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

        # Expand hull outward
        centroid = np.mean(hull_points, axis=0)
        vectors = hull_points - centroid
        expanded_hull = hull_points + vectors * (2**0.5 * resolution / np.linalg.norm(vectors, axis=1)[:, np.newaxis])

        # Create points along each hull edge
        boundary_points = []
        p1 = expanded_hull
        p2 = np.roll(expanded_hull, 1, axis=0)

        # Calculate number of points needed along this edge
        edge_length = np.linalg.norm(p2 - p1, axis=1)
        num_points = np.ceil(edge_length / resolution).astype(int)

        for i in np.where(num_points > 2)[0]:
            # Create evenly spaced points along the edge
            t = np.linspace(0, 1, num_points[i])[1:-1][:, None]  # Exclude last point to avoid duplicates
            edge_points = p1[i] + t * (p2[i] - p1[i])
            boundary_points.append(edge_points)

        return np.concatenate([expanded_hull, np.vstack(boundary_points)])

    def compute_area_weights(self, latlons: np.ndarray) -> np.ndarray:
        """Compute area weights.

        Parameters
        ----------
        latlons : np.ndarray
            2D array of shape (N, 2) with latitude and longitude coordinates of the
            nodes of the graph.

        Returns
        -------
        np.ndarray
            Planar area weights.
        """
        resolution = self._compute_mean_nearest_distance(latlons)
        boundary_points = self._get_boundary_ring(latlons, resolution)

        # Compute convex hull over all points (boundary ring included)
        extended_points = np.vstack([latlons, boundary_points])
        v = Voronoi(extended_points, qhull_options="QJ Pp")

        # Compute the area of each node's region, excluding those in the boundary ring
        areas = []
        for idx in range(len(latlons)):
            p_idx = v.point_region[idx]
            r = v.regions[p_idx]
            poly_coords = v.vertices[r]
            area = ConvexHull(poly_coords).volume
            areas.append(area)

        return np.array(areas)


class MaskedPlanarAreaWeights(PlanarAreaWeights):
    """Masked planar area weights

    It computes the area in a 2D plane asociated to each node.

    Attributes
    ----------
    mask_node_attr_name : str
        Name of a node attribute to use as a mask for the computing the area weights.
        It sets to 0 values outside this masked region.
    norm : str, optional
        Normalisation of the weights.

    Methods
    -------
    compute(self, graph, nodes_name)
        Compute the area attributes for each node.
    """

    def __init__(
        self,
        mask_node_attr_name: str,
        norm: str | None = None,
        dtype: str = "float32",
    ) -> None:
        super().__init__(norm, dtype)
        assert isinstance(
            mask_node_attr_name, str
        ), f"{self.__class__.__name__} requires a string for 'mask_node_attr_name' variable."
        self.mask_node_attr_name = mask_node_attr_name

    def get_raw_values(self, nodes: NodeStorage, **kwargs) -> torch.Tensor:
        assert self.mask_node_attr_name in nodes, f"Node attribute '{self.mask_node_attr_name}' not found in nodes."
        attr_values = super().get_raw_values(nodes, **kwargs).to(self.device)
        mask = nodes[self.mask_node_attr_name].squeeze()
        return attr_values * mask


class SphericalAreaWeights(BaseAreaWeights):
    """Spherical area weights

    It computes the area of a unit radius sphere asociated to each node.

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
        assert isinstance(fill_value, float) or isinstance(
            fill_value, int
        ), f"fill_value must be float or nan but it is {type(fill_value)}"
        assert (
            isinstance(radius, float) or isinstance(radius, int)
        ) and radius > 0, f"radius must be a positive value, but radius={radius}"
        super().__init__(norm, dtype)
        self.radius = radius
        self.centre = centre
        self.fill_value = fill_value

    def compute_area_weights(self, latlons: np.ndarray) -> np.ndarray:
        """Compute the area associated to each node.

        It uses Voronoi diagrams to compute the area of each node.

        Parameters
        ----------
        latlons : np.ndarray
            2D array of shape (N, 2) with latitude and longitude coordinates of the
            nodes of the graph.

        Returns
        -------
        np.ndarray
            Spherical area weights.
        """
        points = latlon_rad_to_cartesian_np(latlons)
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
        return result


class BaseLatWeightedAttribute(BaseAreaWeights, ABC):
    """Base class for latitude-weigthed area weights."""

    @abstractmethod
    def compute_latitude_weight(self, latitudes: np.ndarray) -> np.ndarray: ...

    def compute_area_weights(self, latlons: np.ndarray) -> np.ndarray:
        return self.compute_latitude_weight(latlons[:, 0])


class CosineLatWeightedAttribute(BaseLatWeightedAttribute):
    """Latitude-weighting of the node attributes for rectilinear grids.

    Attributes
    ----------
    min_value : float
        Minimum value of the weights when the latitude is -pi/2 or pi/2 radians.
    max_value : float
        Maximum value of the weights when the latitude is 0 radians.
    norm : str
        Normalisation of the weights.

    Methods
    -------
    compute(self, graph, nodes_name)
        Compute the area attributes for each node.
    """

    def __init__(
        self,
        min_value: float = 1e-3,
        max_value: float = 1,
        norm: str | None = None,
        dtype: str = "float32",
    ) -> None:
        super().__init__(norm, dtype)
        self.min_value = min_value
        self.max_value = max_value

    def compute_latitude_weight(self, latitudes: np.ndarray) -> np.ndarray:
        return (self.max_value - self.min_value) * np.cos(latitudes) + self.min_value


class IsolatitudeAreaWeights(BaseLatWeightedAttribute):
    r"""Latitude-weighted area weights for rectilinear grids.

    Attributes
    ----------
    norm : str
        Normalisation of the weights.

    Methods
    -------
    compute(self, graph, nodes_name)
        Compute the area attributes for each node.

    Notes
    ------
    The area of a latitude band is
    .. math::
        A = 2\pi R(\sin(lat_2) - \sin(lat_1))
    where R is the earth radius and lat_1, lat_2 are in radians.
    """

    def compute_latitude_weight(self, latitudes: np.ndarray) -> np.ndarray:
        # Get the latitudes defining the bands
        unique_lats = np.sort(np.unique(latitudes))
        divisory_lats = (unique_lats[1:] + unique_lats[:-1]) / 2
        divisory_lats = np.concatenate([[-np.pi / 2], divisory_lats, [np.pi / 2]])

        # Compute the latitude band area
        lat_1 = divisory_lats[1:]
        lat_2 = divisory_lats[:-1]
        assert np.all(lat_1 >= lat_2), "Nodes should be sorted by latitude."
        ring_area_km = 2 * np.pi * EARTH_RADIUS * (np.sin(lat_1) - np.sin(lat_2))

        # Compute the number of points/nodes at each latitude band
        lat_to_ring = {lat: idx for idx, lat in enumerate(unique_lats)}
        lat_rings = np.array([lat_to_ring[lat] for lat in latitudes])
        lat_counts = np.bincount(lat_rings, minlength=len(unique_lats))

        # Compute the area of each node
        area_km = dict(zip(unique_lats, ring_area_km / lat_counts))
        return np.array([area_km[lat] for lat in latitudes])
