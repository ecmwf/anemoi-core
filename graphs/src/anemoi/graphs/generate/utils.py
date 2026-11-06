# (C) Copyright 2024-2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import numpy as np
import scipy
from scipy.spatial import cKDTree
from typeguard import typechecked

from anemoi.graphs import EARTH_RADIUS
from anemoi.graphs.generate.transforms import latlon_rad_to_cartesian_np


def get_coordinates_ordering(coords: np.ndarray) -> np.ndarray:
    """Sort node coordinates by latitude and longitude.

    Parameters
    ----------
    coords : np.ndarray of shape (N, 2)
        The node coordinates, with the latitude in the first column and the
        longitude in the second column.

    Returns
    -------
    np.ndarray
        The order of the node coordinates to be sorted by latitude and longitude.
    """
    # Get indices to sort points by lon & lat in radians.
    index_latitude = np.argsort(coords[:, 1])
    index_longitude = np.argsort(coords[index_latitude][:, 0])[::-1]
    node_ordering = np.arange(coords.shape[0])[index_latitude][index_longitude]
    return node_ordering


@typechecked
def convert_list_to_adjacency_matrix(list_matrix: np.ndarray, ncols: int = 0) -> scipy.sparse.csr_matrix:
    """Convert an edge list into an adjacency matrix.

    Parameters
    ----------
    list_matrix : np.ndarray
        boolean matrix given by list of column indices for each row.
    ncols : int
        number of columns in result matrix.

    Returns
    -------
    scipy.sparse.csr_matrix
        sparse matrix [nrows, ncols]
    """
    nrows, ncols_per_row = list_matrix.shape
    indptr = np.arange(ncols_per_row * (nrows + 1), step=ncols_per_row)
    indices = list_matrix.ravel()
    return scipy.sparse.csr_matrix((np.ones(nrows * ncols_per_row), indices, indptr), dtype=bool, shape=(nrows, ncols))


@typechecked
def convert_adjacency_matrix_to_list(
    adj_matrix: scipy.sparse.csr_matrix,
    ncols_per_row: int,
    remove_duplicates: bool = True,
) -> np.ndarray:
    """Convert an adjacency matrix into an edge list.

    Parameters
    ----------
    adj_matrix : scipy.sparse.csr_matrix
        sparse (boolean) adjacency matrix
    ncols_per_row : int
        number of nonzero entries per row
    remove_duplicates : bool
        logical flag: remove duplicate rows.

    Returns
    -------
    np.ndarray
        boolean matrix given by list of column indices for each row.
    """
    if remove_duplicates:
        # The edges-vertex adjacency matrix may have duplicate rows, remove
        # them by selecting the rows that are unique:
        nrows = int(adj_matrix.nnz // ncols_per_row)
        mat = adj_matrix.indices.reshape((nrows, ncols_per_row))
        return np.unique(mat, axis=0)

    nrows = adj_matrix.shape[0]
    return adj_matrix.indices.reshape((nrows, ncols_per_row))


@typechecked
def selection_matrix(idx: np.ndarray, num_diagonals: int) -> scipy.sparse.csr_matrix:
    """Create a diagonal selection matrix.

    Parameters
    ----------
    idx : np.ndarray
        integer array of indices
    num_diagonals : int
        size of (square) selection matrix

    Returns
    -------
    scipy.sparse.csr_matrix
        diagonal matrix with ones at selected indices (idx,idx).
    """
    return scipy.sparse.csr_matrix((np.ones(len(idx)), (idx, idx)), dtype=bool, shape=(num_diagonals, num_diagonals))


def compute_field_gradient(
    source_coords: np.ndarray,
    values: np.ndarray,
    query_coords: np.ndarray,
    num_neighbours: int = 12,
    chunk_size: int = 100_000,
) -> np.ndarray:
    """Gradient magnitude of a scattered field, per kilometre, at arbitrary coordinates.

    For each query point, fits a plane to the ``num_neighbours`` nearest source points
    in the local tangent plane. Source points with missing values are ignored. A locally
    constant field has a gradient of exactly 0, and so do query points outside the
    source points, where the fit would be an extrapolation.

    Parameters
    ----------
    source_coords : np.ndarray of shape (N, 2)
        Latitude and longitude of the source points, in radians.
    values : np.ndarray of shape (N,)
        Field values at the source points.
    query_coords : np.ndarray of shape (M, 2)
        Latitude and longitude of the query points, in radians.
    num_neighbours : int, optional
        Number of source points used in each fit, by default 12.
    chunk_size : int, optional
        Number of query points processed at once, by default 100000.

    Returns
    -------
    np.ndarray of shape (M,)
        Gradient magnitude of the field, in field units per kilometre.
    """
    values = np.asarray(values, dtype=np.float64)
    valid = np.isfinite(values)
    assert valid.sum() >= 3, "At least 3 source points with valid values are needed to compute a gradient."
    source_xyz = latlon_rad_to_cartesian_np(np.asarray(source_coords, dtype=np.float64)[valid], radius=EARTH_RADIUS)
    values = values[valid]

    query_coords = np.asarray(query_coords, dtype=np.float64)
    query_xyz = latlon_rad_to_cartesian_np(query_coords, radius=EARTH_RADIUS)
    lat, lon = query_coords[:, 0], query_coords[:, 1]
    east = np.stack([-np.sin(lon), np.cos(lon), np.zeros_like(lon)], axis=-1)
    north = np.stack([-np.sin(lat) * np.cos(lon), -np.sin(lat) * np.sin(lon), np.cos(lat)], axis=-1)

    num_neighbours = min(num_neighbours, len(values))
    _, neighbours = cKDTree(source_xyz).query(query_xyz, k=num_neighbours, workers=-1)
    neighbours = neighbours.reshape(len(query_coords), num_neighbours)

    gradient = np.empty(len(query_coords))
    for start in range(0, len(query_coords), chunk_size):
        chunk = slice(start, start + chunk_size)
        offsets = source_xyz[neighbours[chunk]] - query_xyz[chunk][:, None]
        design = np.stack(
            [
                np.ones(offsets.shape[:2]),
                np.einsum("nkj,nj->nk", offsets, east[chunk]),
                np.einsum("nkj,nj->nk", offsets, north[chunk]),
            ],
            axis=-1,
        )
        local_values = values[neighbours[chunk]]
        # Relative to one neighbour rather than the mean, so a constant field gives exactly 0.
        local_values = local_values - local_values[:, :1]
        coefficients = np.einsum("nij,nj->ni", np.linalg.pinv(design), local_values)
        gradient[chunk] = np.hypot(coefficients[:, 1], coefficients[:, 2])

        # The fit is meaningless when extrapolating. On the edge of the data the centroid of a
        # half-disc of neighbours is near 0.42 of its radius; farther out it tends to 1.
        centroid_distance = np.linalg.norm(offsets.mean(axis=1), axis=-1)
        gradient[chunk][centroid_distance > 0.6 * np.linalg.norm(offsets, axis=-1).max(axis=1)] = 0.0
    return gradient
