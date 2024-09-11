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


def add_margin(lats: np.ndarray, lons: np.ndarray, margin: float) -> tuple[np.ndarray, np.ndarray]:
    """Add a margin to the convex hull of the points considered.

    For each point (lat, lon) add 8 points around it, each at a distance of `margin` from the original point.

    Arguments
    ---------
    lats : np.ndarray
        Latitudes of the points considered.
    lons : np.ndarray
        Longitudes of the points considered.
    margin : float
        The margin to add to the convex hull.

    Returns
    -------
    latitudes : np.ndarray
        Latitudes of the points considered, including the margin.
    longitudes : np.ndarray
        Longitudes of the points considered, including the margin.
    """
    assert margin >= 0, "Margin must be non-negative"
    if margin == 0:
        return lats, lons

    latitudes, longitudes = [], []
    for lat_sign in [-1, 0, 1]:
        for lon_sign in [-1, 0, 1]:
            latitudes.append(lats + lat_sign * margin)
            longitudes.append(lons + lon_sign * margin)

    return np.concatenate(latitudes), np.concatenate(longitudes)


def get_index_in_outer_join(vector: torch.Tensor, tensor: torch.Tensor) -> int:
    """Index position of vector.

    Get the index position of a vector in a matrix.

    Parameters
    ----------
    vector : torch.Tensor of shape (N, )
        Vector to get its position in the matrix.
    tensor : torch.Tensor of shape (M, N,)
        Tensor in which the position is searched.

    Returns
    -------
    int
        Index position of `vector` in `tensor`. -1 if `vector` is not in `tensor`.
    """
    mask = torch.all(tensor == vector, axis=1)
    if mask.any():
        return int(torch.where(mask)[0])
    return -1


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
