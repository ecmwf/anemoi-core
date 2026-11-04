# (C) Copyright 2024-2026 Anemoi contributors.
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
_GEOPOTENTIAL_THRESHOLD = 10_000.0
_GEOPOTENTIAL_VAR_NAMES = ("z", "fi")
_GEOPOTENTIAL_UNITS = ("m**2 s**-2", "m2 s-2", "m^2 s^-2", "m2/s2", "m**2/s**2", "j/kg", "j kg-1", "j kg**-1")
_METRE_UNITS = ("m", "metre", "metres", "meter", "meters")
_STANDARD_GRAVITY = 9.80665


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


def load_orography_grid(orography_path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load a regular lat/lon orography field from a NetCDF file.

    Parameters
    ----------
    orography_path : str
        Path to a NetCDF file containing a 2-D orography field. The variable must
        be named one of: orog, z, orography, hsurf, topography, fi (case-insensitive).
        Geopotential values (absolute maximum above 10 000) are converted to metres
        using g = 9.80665 m/s².

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        Ascending latitudes (degrees), longitudes (degrees) and the elevation
        field (metres) with shape (num_lats, num_lons).

    Raises
    ------
    ValueError
        If the orography variable or its lat/lon dimensions cannot be found.
    """
    import xarray as xr

    with xr.open_dataset(orography_path) as ds:
        orog_var = next((v for v in ds.data_vars if str(v).lower() in _OROG_VAR_NAMES), None)
        if orog_var is None:
            raise ValueError(
                f"Cannot find orography variable in {orography_path}. "
                f"Expected one of {_OROG_VAR_NAMES}. Found: {list(ds.data_vars)}"
            )
        orog_da = ds[orog_var].squeeze()

        lat_dim = next((d for d in orog_da.dims if "lat" in str(d).lower()), None)
        lon_dim = next((d for d in orog_da.dims if "lon" in str(d).lower()), None)
        if lat_dim is None or lon_dim is None or orog_da.ndim != 2:
            raise ValueError(f"Expected a 2-D lat/lon field for {orog_var}. Dims: {orog_da.dims}")

        orog_da = orog_da.transpose(lat_dim, lon_dim)
        units = str(orog_da.attrs.get("units", "")).strip().lower()
        lat_vals = orog_da[lat_dim].values.astype(float)
        lon_vals = orog_da[lon_dim].values.astype(float)
        orog_vals = orog_da.values.astype(float)

    if lat_vals[0] > lat_vals[-1]:
        lat_vals = lat_vals[::-1]
        orog_vals = orog_vals[::-1, :]

    if _is_geopotential(str(orog_var).lower(), units, orog_vals):
        LOGGER.info("Orography variable %s is geopotential, converting to metres.", orog_var)
        orog_vals = orog_vals / _STANDARD_GRAVITY

    return lat_vals, lon_vals, orog_vals


def _is_geopotential(var_name: str, units: str, values: np.ndarray) -> bool:
    """Decide whether an orography field is geopotential (m2/s2) rather than height (m)."""
    if units in _GEOPOTENTIAL_UNITS:
        return True
    if units in _METRE_UNITS:
        return False
    if var_name in _GEOPOTENTIAL_VAR_NAMES:
        return True
    # No usable metadata: no surface on Earth is higher than 10 km.
    return bool(np.abs(values).max() > _GEOPOTENTIAL_THRESHOLD)


def _interpolate_to_coords(
    lat_vals: np.ndarray, lon_vals: np.ndarray, field: np.ndarray, coords_rad: np.ndarray
) -> np.ndarray:
    """Linearly interpolate a lat/lon field to node coordinates, 0 outside the grid."""
    from scipy.interpolate import RegularGridInterpolator

    query = np.rad2deg(np.asarray(coords_rad, dtype=float)).reshape(-1, 2)
    # Match the longitude convention of the grid ([0, 360) or [-180, 180)).
    if lon_vals.max() > 180.0:
        query[:, 1] = np.mod(query[:, 1], 360.0)
    else:
        query[:, 1] = (query[:, 1] + 180.0) % 360.0 - 180.0

    interp = RegularGridInterpolator((lat_vals, lon_vals), field, method="linear", bounds_error=False, fill_value=0.0)
    return interp(query)


def compute_orography_gradient(orography_path: str, coords_rad: np.ndarray) -> np.ndarray:
    """Interpolate the orographic slope magnitude (m/m) to node coordinates.

    The gradient is computed with finite differences on the regular lat/lon grid of
    the orography file and then linearly interpolated to ``coords_rad``. Nodes
    outside the orography grid get a gradient of 0.

    Parameters
    ----------
    orography_path : str
        Path to a NetCDF orography file, see :func:`load_orography_grid`.
    coords_rad : np.ndarray, shape (N, 2)
        Node coordinates [lat, lon] in radians.

    Returns
    -------
    np.ndarray, shape (N,)
        Gradient magnitude (m/m) at each node coordinate.
    """
    lat_vals, lon_vals, orog_vals = load_orography_grid(orography_path)

    dlat_m = np.deg2rad(np.gradient(lat_vals)) * _EARTH_RADIUS_M
    dlon_m = np.deg2rad(np.gradient(lon_vals))[np.newaxis, :] * _EARTH_RADIUS_M * np.cos(np.deg2rad(lat_vals))[:, None]

    eps = 1e-10
    grad_lat = np.gradient(orog_vals, axis=0) / (np.abs(dlat_m[:, np.newaxis]) + eps)
    grad_lon = np.gradient(orog_vals, axis=1) / (np.abs(dlon_m) + eps)
    grad_magnitude = np.hypot(grad_lat, grad_lon)

    return _interpolate_to_coords(lat_vals, lon_vals, grad_magnitude, coords_rad)


def compute_orography_elevation(orography_path: str, coords_rad: np.ndarray) -> np.ndarray:
    """Interpolate the orographic elevation (metres) to node coordinates.

    Nodes outside the orography grid get an elevation of 0.

    Parameters
    ----------
    orography_path : str
        Path to a NetCDF orography file, see :func:`load_orography_grid`.
    coords_rad : np.ndarray, shape (N, 2)
        Node coordinates [lat, lon] in radians.

    Returns
    -------
    np.ndarray, shape (N,)
        Elevation (metres) at each node coordinate.
    """
    lat_vals, lon_vals, orog_vals = load_orography_grid(orography_path)
    return _interpolate_to_coords(lat_vals, lon_vals, orog_vals, coords_rad)
