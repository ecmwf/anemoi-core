# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import logging

import numpy as np
import scipy
from typeguard import typechecked

LOGGER = logging.getLogger(__name__)

_EARTH_RADIUS_M = 6_371_000.0
_OROG_VAR_NAMES = ("orog", "z", "orography", "hsurf", "topography", "fi")


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


def compute_orography_gradient(orography_path: str, coords_rad: np.ndarray) -> np.ndarray:
    """Interpolate orographic slope magnitude (m/m) to arbitrary node coordinates.

    Loads the orography from a NetCDF file, computes the 2-D gradient on the
    regular lat/lon grid using finite differences, then linearly interpolates
    the gradient magnitude to the requested coordinates.

    Parameters
    ----------
    orography_path : str
        Path to a NetCDF file containing a 2-D orography field.  The variable
        must be named one of: orog, z, orography, hsurf, topography, fi
        (case-insensitive).  Geopotential values (> 10 000 J/kg) are
        automatically converted to metres using g = 9.806 65 m/s².
    coords_rad : np.ndarray, shape (N, 2)
        Node coordinates [lat, lon] in radians.

    Returns
    -------
    np.ndarray, shape (N,)
        Gradient magnitude (m/m) at each node coordinate.

    Raises
    ------
    ValueError
        If the orography variable or its lat/lon dimensions cannot be found.
    """
    import xarray as xr
    from scipy.interpolate import RegularGridInterpolator

    ds = xr.open_dataset(orography_path)
    orog_var = next((v for v in ds.data_vars if v.lower() in _OROG_VAR_NAMES), None)
    if orog_var is None:
        raise ValueError(
            f"Cannot find orography variable in {orography_path}. "
            f"Expected one of {_OROG_VAR_NAMES}. Found: {list(ds.data_vars)}"
        )
    orog_da = ds[orog_var].squeeze()

    lat_dim = next((d for d in orog_da.dims if "lat" in d.lower()), None)
    lon_dim = next((d for d in orog_da.dims if "lon" in d.lower()), None)
    if lat_dim is None or lon_dim is None:
        raise ValueError(
            f"Cannot identify lat/lon dimensions in {orog_var}. Dims: {orog_da.dims}"
        )

    lat_vals = orog_da[lat_dim].values.astype(float)
    lon_vals = orog_da[lon_dim].values.astype(float)
    orog_vals = orog_da.values.astype(float)

    if lat_vals[0] > lat_vals[-1]:
        lat_vals = lat_vals[::-1]
        orog_vals = orog_vals[::-1, :]

    if np.abs(orog_vals).max() > 10_000:
        LOGGER.debug("Geopotential detected (max=%.0f J/kg), converting to metres.", orog_vals.max())
        orog_vals = orog_vals / 9.80665

    dlat_m = np.deg2rad(np.gradient(lat_vals)) * _EARTH_RADIUS_M
    cos_lat = np.cos(np.deg2rad(lat_vals))
    dlon_m = (
        np.deg2rad(np.gradient(lon_vals))[np.newaxis, :]
        * _EARTH_RADIUS_M
        * cos_lat[:, np.newaxis]
    )

    grad_lat = np.gradient(orog_vals, axis=0)
    grad_lon = np.gradient(orog_vals, axis=1)

    eps = 1e-10
    grad_magnitude = np.sqrt(
        (grad_lat / (np.abs(dlat_m[:, np.newaxis]) + eps)) ** 2
        + (grad_lon / (np.abs(dlon_m) + eps)) ** 2
    )

    interp = RegularGridInterpolator(
        (lat_vals, lon_vals),
        grad_magnitude,
        method="linear",
        bounds_error=False,
        fill_value=0.0,
    )
    return interp(np.rad2deg(coords_rad))


def compute_orography_elevation(orography_path: str, coords_rad: np.ndarray) -> np.ndarray:
    """Interpolate raw orographic elevation (metres) to arbitrary node coordinates.

    Loads the orography from a NetCDF file and linearly interpolates the elevation
    field to the requested coordinates.  Geopotential values are automatically
    converted to metres.

    Parameters
    ----------
    orography_path : str
        Path to a NetCDF file containing a 2-D orography field.  The variable
        must be named one of: orog, z, orography, hsurf, topography, fi.
        Geopotential values (> 10 000 J/kg) are converted to metres using
        g = 9.806 65 m/s².
    coords_rad : np.ndarray, shape (N, 2)
        Node coordinates [lat, lon] in radians.

    Returns
    -------
    np.ndarray, shape (N,)
        Elevation (metres) at each node coordinate.

    Raises
    ------
    ValueError
        If the orography variable or its lat/lon dimensions cannot be found.
    """
    import xarray as xr
    from scipy.interpolate import RegularGridInterpolator

    ds = xr.open_dataset(orography_path)
    orog_var = next((v for v in ds.data_vars if v.lower() in _OROG_VAR_NAMES), None)
    if orog_var is None:
        raise ValueError(
            f"Cannot find orography variable in {orography_path}. "
            f"Expected one of {_OROG_VAR_NAMES}. Found: {list(ds.data_vars)}"
        )
    orog_da = ds[orog_var].squeeze()

    lat_dim = next((d for d in orog_da.dims if "lat" in d.lower()), None)
    lon_dim = next((d for d in orog_da.dims if "lon" in d.lower()), None)
    if lat_dim is None or lon_dim is None:
        raise ValueError(
            f"Cannot identify lat/lon dimensions in {orog_var}. Dims: {orog_da.dims}"
        )

    lat_vals = orog_da[lat_dim].values.astype(float)
    lon_vals = orog_da[lon_dim].values.astype(float)
    orog_vals = orog_da.values.astype(float)

    if lat_vals[0] > lat_vals[-1]:
        lat_vals = lat_vals[::-1]
        orog_vals = orog_vals[::-1, :]

    if np.abs(orog_vals).max() > 10_000:
        LOGGER.debug("Geopotential detected (max=%.0f J/kg), converting to metres.", orog_vals.max())
        orog_vals = orog_vals / 9.80665

    interp = RegularGridInterpolator(
        (lat_vals, lon_vals),
        orog_vals,
        method="linear",
        bounds_error=False,
        fill_value=0.0,
    )
    return interp(np.rad2deg(coords_rad))
