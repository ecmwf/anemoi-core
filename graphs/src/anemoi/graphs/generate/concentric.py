# (C) Copyright 2024 Anemoi contributors.
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

import networkx as nx
import numpy as np
import math

from anemoi.graphs.generate.masks import KNNAreaMaskBuilder
from anemoi.graphs.generate.transforms import latlon_rad_to_cartesian, cartesian_to_latlon_rad
from anemoi.graphs.generate.tri_icosahedron import create_nx_graph_from_tri_coords
from anemoi.graphs.generate.tri_icosahedron import get_latlon_coords_icosphere
from anemoi.graphs.generate.utils import get_coordinates_ordering

LOGGER = logging.getLogger(__name__)

import numpy as np

def central_point_on_sphere(latlons: np.ndarray) -> np.ndarray:
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
    # Convert lat-lon to Cartesian
    xyz = latlon_rad_to_cartesian((latlons[:, 0], latlons[:, 1]))
    
    # Compute mean of Cartesian coordinates
    mean_xyz = xyz.mean(axis=0)
    
    # Normalize to project back onto the sphere
    mean_xyz /= np.linalg.norm(mean_xyz)
    
    # Convert back to lat-lon
    return cartesian_to_latlon_rad(mean_xyz[np.newaxis, :])[0]

def get_latlon_coords_concentric(
    center_coords: Tuple[float, float],
    n_circles: int,
    base_dist: float,
    min_n_points: int,
    max_n_points: int,
    sphere_radius: float = 6371.0  # default Earth's radius in km
) -> np.ndarray:
    """
    Generate concentric geodesic circles (lat, lon in radians) around a given center on Earth.

    Uses the spherical destination formulas:
      lat2 = arcsin( sin(lat1)*cos(d/R) + cos(lat1)*sin(d/R)*cos(theta) )
      lon2 = lon1 + atan2( sin(theta)*sin(d/R)*cos(lat1),
                           cos(d/R) - sin(lat1)*sin(lat2) )
    
    Parameters
    ----------
    center_coords : Tuple[float, float]
        Center point (lat, lon) in radians.
    n_circles : int
        Number of concentric circles to generate.
    base_dist : float
        Distance (in km) for the smallest circle.
    min_n_points : int
        Minimum number of points on each circle.
    max_n_points : int
        Maximum number of points on each circle.
    sphere_radius : float, optional
        Radius of the sphere (default is 6371.0 km for Earth).

    Returns
    -------
    np.ndarray
        Array of shape (N, 2) containing (lat, lon) points in radians.
    """
    # Limit the outer circle to a reasonable distance (e.g., 2000 km) so the rings remain local.
    max_distance_km = 20000.0
    distances = np.geomspace(base_dist, max_distance_km, n_circles)[:-1]
    num_points_arr = np.linspace(min_n_points, max_n_points, n_circles)[::-1][:-1]

    # Assume center_coords are already in radians.
    lat_center, lon_center = center_coords

    result_points = []
    prev_r = 0.0
    for i, (r, n_pts) in enumerate(zip(distances, num_points_arr)):
        n_pts = int(n_pts)
        d_over_R = r / sphere_radius  # Angular distance in radians.
        for j in range(n_pts):
            theta = 2.0 * math.pi * j / n_pts  # Bearing in radians.
            lat2 = math.asin(math.sin(lat_center) * math.cos(d_over_R) +
                             math.cos(lat_center) * math.sin(d_over_R) * math.cos(theta))
            lon2 = lon_center + math.atan2(
                math.sin(theta) * math.sin(d_over_R) * math.cos(lat_center),
                math.cos(d_over_R) - math.sin(lat_center) * math.sin(lat2)
            )
            # Normalize longitude to the range [-pi, pi]
            lon2 = (lon2 + math.pi) % (2 * math.pi) - math.pi
            result_points.append((lat2, lon2))
        LOGGER.info(f"Circle {i}: radius={r:.2f} km, Δ from prev={r - prev_r:.2f} km, points={n_pts}")
        prev_r = r

    return np.array(result_points)


def create_concentric_mesh(
    center_coords: Tuple[int, int],
    n_circles: int = 200,
    base_dist: float = 0.1,
    min_n_points: int = 64,
    max_n_points: int = 1024,
    area_mask_builder: KNNAreaMaskBuilder | None = None,
) -> tuple[nx.DiGraph, np.ndarray, list[int]]:
    """Creates a global mesh with concentric circles around a given lat-lon point.

    Parameters
    ----------
    center_coords: tuple(int, int)
        Latitude and Longitude of center.

    n_circles: int
        Number of circles to generate around the center.

    base_dist: float = 0.1
        Distance of the first circle fromn the center in km.

    min_n_points: int = 64,
        Minimum number of points in the further away circle.

    max_n_points: int = 1024
        Maximum number of points in the innermost circle.

    area_mask_builder : KNNAreaMaskBuilder
        KNNAreaMaskBuilder with the cloud of points to limit the mesh area, by default None.

    Returns
    -------
    graph : networkx.Graph
        The specified graph (only nodes) sorted by latitude and longitude.
    coords_rad : np.ndarray
        The node coordinates (not ordered) in radians.
    node_ordering : list[int]
        Order of the node coordinates to be sorted by latitude and longitude.
    """
    coords_rad = get_latlon_coords_concentric(center_coords, n_circles, base_dist, min_n_points, max_n_points)

    node_ordering = get_coordinates_ordering(coords_rad)

    if area_mask_builder is not None:
        area_mask = area_mask_builder.get_mask(coords_rad)
        node_ordering = node_ordering[area_mask[node_ordering]]

    # Creates the graph, with the nodes sorted by latitude and longitude.
    nx_graph = create_nx_graph_from_tri_coords(coords_rad, node_ordering)

    return nx_graph, coords_rad, list(node_ordering)


def create_stretched_concentric(
    n_circles: int = 200,
    base_dist: float = 0.1,
    min_n_points: int = 64,
    max_n_points: int = 1024,
    lam_resolution: int = 10,
    area_mask_builder: KNNAreaMaskBuilder | None = None,
) -> tuple[nx.DiGraph, np.ndarray, list[int]]:
    """
    Creates a global mesh with 2 levels of resolution.

    The nodes outside the Area Of Interest (AOI) are generated as a concentric mesh,
    while the lam_resolution is used to define the nodes inside the AOI.

    Parameters
    ---------
    n_circles: int
        Number of circles to generate around the center.

    base_dist: float = 0.1
        Distance of the first circle fromn the center in km.

    min_n_points: int = 64,
        Minimum number of points in the further away circle.

    max_n_points: int = 1024
        Maximum number of points in the innermost circle.

    lam_resolution : int
        Local resolution level.
    area_mask_builder : KNNAreaMaskBuilder
        NearestNeighbors with the cloud of points to limit the mesh area.

    Returns
    -------
    nx_graph : nx.DiGraph
        The graph with the added nodes.
    coords_rad : np.ndarray
        The node coordinates (not ordered) in radians.
    node_ordering : list[int]
        Order of the node coordinates to be sorted by latitude and longitude.
    """
    assert area_mask_builder is not None, "AOI mask builder must be provided to build refined grid."

    # Get the high resolution nodes inside the AOI
    lam_coords_rad = get_latlon_coords_icosphere(lam_resolution)
    lam_area_mask = area_mask_builder.get_mask(lam_coords_rad)

    # Compute center of AOI
    center_coords = central_point_on_sphere(lam_coords_rad[lam_area_mask])
    # LOGGER.info("Computed Centre of LAM (in rads): ", center_coords)

    # Get the low resolution nodes outside the AOI
    base_coords_rad = get_latlon_coords_concentric(center_coords, n_circles, base_dist, min_n_points, max_n_points)
    base_area_mask = ~area_mask_builder.get_mask(base_coords_rad)

    coords_rad = np.concatenate([base_coords_rad[base_area_mask], lam_coords_rad[lam_area_mask]])

    node_ordering = get_coordinates_ordering(coords_rad)

    # Creates the graph, with the nodes sorted by latitude and longitude.
    nx_graph = create_nx_graph_from_tri_coords(coords_rad, node_ordering)

    return nx_graph, coords_rad, list(node_ordering)
