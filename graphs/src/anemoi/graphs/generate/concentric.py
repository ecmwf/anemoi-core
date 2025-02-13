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
from geopy.distance import geodesic

from anemoi.graphs.generate.masks import KNNAreaMaskBuilder
from anemoi.graphs.generate.transforms import cartesian_to_latlon_rad
from anemoi.graphs.generate.transforms import latlon_rad_to_cartesian
from anemoi.graphs.generate.tri_icosahedron import create_nx_graph_from_tri_coords
from anemoi.graphs.generate.tri_icosahedron import get_latlon_coords_icosphere
from anemoi.graphs.generate.utils import get_coordinates_ordering

LOGGER = logging.getLogger(__name__)


def find_geographical_center(coords):
    """
    Compute the central point (geometric centroid) of a set of latitude and longitude coordinates.

    Parameters:
        cords : list of (lat, lon) tuples

    Returns:
        (lat, lon) tuple representing the central point.
    """
    cartesian_coords = np.array([latlon_rad_to_cartesian(coord) for coord in coords])

    # Compute mean Cartesian coordinates
    center = cartesian_coords.mean(axis=0)

    # Normalize the vector to project back onto the sphere
    center /= np.linalg.norm(center)

    # Convert back to latitude and longitude
    return cartesian_to_latlon_rad(*center)


def get_latlon_coords_concentric(
    center_coords: Tuple[int, int], n_circles: int, base_dist: float, min_n_points: int, max_n_points: int
) -> np.ndarray:

    result = []
    max_distance_km = 20000  # Half of Earth's circumference

    distances = np.geomspace(base_dist, max_distance_km, n_circles)[:-1]

    points = np.linspace(min_n_points, max_n_points, n_circles)[::-1][:-1]

    prev_r = 0
    for i, (r, num_points) in enumerate(zip(distances, points)):
        num_points = int(num_points)

        for j in range(num_points):
            angle = 360 * j / num_points  # Equally spaced points
            new_point = geodesic(kilometers=r).destination(center_coords, angle)
            result.append((new_point.latitude, new_point.longitude))

        LOGGER.info(f"\nCircle {i} has a distance from the center of: {r}.")
        LOGGER.info(f"Distance from previous circle: {r-prev_r} km.")
        LOGGER.info(f"Number of pointss: {num_points}.")

        prev_r = r

    return np.array(result)


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
    center_coords = find_geographical_center(lam_coords_rad)

    # Get the low resolution nodes outside the AOI
    base_coords_rad = get_latlon_coords_concentric(center_coords, n_circles, base_dist, min_n_points, max_n_points)
    base_area_mask = ~area_mask_builder.get_mask(base_coords_rad)

    coords_rad = np.concatenate([base_coords_rad[base_area_mask], lam_coords_rad[lam_area_mask]])

    node_ordering = get_coordinates_ordering(coords_rad)

    # Creates the graph, with the nodes sorted by latitude and longitude.
    nx_graph = create_nx_graph_from_tri_coords(coords_rad, node_ordering)

    return nx_graph, coords_rad, list(node_ordering)
