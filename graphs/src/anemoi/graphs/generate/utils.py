# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import numpy as np
import scipy
from typeguard import typechecked


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
