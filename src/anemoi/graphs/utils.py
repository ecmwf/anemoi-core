# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from __future__ import annotations

import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors


def get_nearest_neighbour(coords_rad: torch.Tensor, mask: torch.Tensor | None = None) -> NearestNeighbors:
    """Get NearestNeighbour object fitted to coordinates.

    Parameters
    ----------
    coords_rad : torch.Tensor
        corrdinates in radians
    mask : Optional[torch.Tensor], optional
        mask to remove nodes, by default None

    Returns
    -------
    NearestNeighbors
        fitted NearestNeighbour object
    """
    assert mask is None or mask.shape == (
        coords_rad.shape[0],
        1,
    ), "Mask must have the same shape as the number of nodes."

    nearest_neighbour = NearestNeighbors(metric="haversine", n_jobs=4)

    nearest_neighbour.fit(coords_rad)

    return nearest_neighbour


def get_grid_reference_distance(coords_rad: torch.Tensor, mask: torch.Tensor | None = None) -> float:
    """Get the reference distance of the grid.

    It is the maximum distance of a node in the mesh with respect to its nearest neighbour.

    Parameters
    ----------
    coords_rad : torch.Tensor
        corrdinates in radians
    mask : Optional[torch.Tensor], optional
        mask to remove nodes, by default None

    Returns
    -------
    float
        The reference distance of the grid.
    """
    nearest_neighbours = get_nearest_neighbour(coords_rad, mask)
    dists, _ = nearest_neighbours.kneighbors(coords_rad, n_neighbors=2, return_distance=True)
    return dists[dists > 0].max()


def concat_edges(edge_indices1: torch.Tensor, edge_indices2: torch.Tensor) -> torch.Tensor:
    """Concat edges

    Parameters
    ----------
    edge_indices1: torch.Tensor
        Edge indices of the first set of edges. Shape: (2, num_edges1)
    edge_indices2: torch.Tensor
        Edge indices of the second set of edges. Shape: (2, num_edges2)

    Returns
    -------
    torch.Tensor
        Concatenated edge indices.
    """
    return torch.unique(torch.cat([edge_indices1, edge_indices2], axis=1), dim=1)


def haversine_distance(source_coords: np.ndarray, target_coords: np.ndarray) -> np.ndarray:
    """Haversine distance.

    Parameters
    ----------
    source_coords : np.ndarray of shape (N, 2)
        Source coordinates in radians.
    target_coords : np.ndarray of shape (N, 2)
        Destination coordinates in radians.

    Returns
    -------
    np.ndarray of shape (N,)
        Haversine distance between source and destination coordinates.
    """
    dlat = target_coords[:, 0] - source_coords[:, 0]
    dlon = target_coords[:, 1] - source_coords[:, 1]
    a = np.sin(dlat / 2) ** 2 + np.cos(source_coords[:, 0]) * np.cos(target_coords[:, 0]) * np.sin(dlon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return c
