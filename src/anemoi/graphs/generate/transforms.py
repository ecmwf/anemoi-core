# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import numpy as np


def cartesian_to_latlon_degrees(xyz: np.ndarray) -> np.ndarray:
    """3D to lat-lon (in degrees) conversion.

    Convert 3D coordinates of points to the (lat, lon) on the sphere containing
    them.

    Parameters
    ----------
    xyz : np.ndarray
        The 3D coordinates of points.

    Returns
    -------
    np.ndarray
        A 2D array of lat-lon coordinates of shape (N, 2).
    """
    lat = np.arcsin(xyz[..., 2] / (xyz**2).sum(axis=1)) * 180.0 / np.pi
    lon = np.arctan2(xyz[..., 1], xyz[..., 0]) * 180.0 / np.pi
    return np.array((lat, lon), dtype=np.float32).transpose()


def cartesian_to_latlon_rad(xyz: np.ndarray) -> np.ndarray:
    """3D to lat-lon (in radians) conversion.

    Convert 3D coordinates of points to its coordinates on the sphere containing
    them.

    Parameters
    ----------
    xyz : np.ndarray
        The 3D coordinates of points.

    Returns
    -------
    np.ndarray
        A 2D array of the coordinates of shape (N, 2) in radians.
    """
    lat = np.arcsin(xyz[..., 2] / (xyz**2).sum(axis=1))
    lon = np.arctan2(xyz[..., 1], xyz[..., 0])
    return np.array((lat, lon), dtype=np.float32).transpose()


def sincos_to_latlon_rad(sincos: np.ndarray) -> np.ndarray:
    """Sine & cosine components to lat-lon coordinates.

    Parameters
    ----------
    sincos : np.ndarray
        The sine and cosine componenets of the latitude and longitude. Shape: (N, 4).
        The dimensions correspond to: sin(lat), cos(lat), sin(lon) and cos(lon).

    Returns
    -------
    np.ndarray
        A 2D array of the coordinates of shape (N, 2) in radians.
    """
    latitudes = np.arctan2(sincos[:, 0], sincos[:, 1])
    longitudes = np.arctan2(sincos[:, 2], sincos[:, 3])
    return np.stack([latitudes, longitudes], axis=-1)


def sincos_to_latlon_degrees(sincos: np.ndarray) -> np.ndarray:
    """Sine & cosine components to lat-lon coordinates.

    Parameters
    ----------
    sincos : np.ndarray
        The sine and cosine componenets of the latitude and longitude. Shape: (N, 4).
        The dimensions correspond to: sin(lat), cos(lat), sin(lon) and cos(lon).

    Returns
    -------
    np.ndarray
        A 2D array of the coordinates of shape (N, 2) in degrees.
    """
    return np.rad2deg(sincos_to_latlon_rad(sincos))


def latlon_rad_to_cartesian(loc: tuple[np.ndarray, np.ndarray], radius: float = 1) -> np.ndarray:
    """Convert planar coordinates to 3D coordinates in a sphere.

    Parameters
    ----------
    loc : np.ndarray
        The 2D coordinates of the points, in radians.
    radius : float, optional
        The radius of the sphere containing los points. Defaults to the unit sphere.

    Returns
    -------
    np.array of shape (3, num_points)
        3D coordinates of the points in the sphere.
    """
    latr, lonr = loc[0], loc[1]
    x = radius * np.cos(latr) * np.cos(lonr)
    y = radius * np.cos(latr) * np.sin(lonr)
    z = radius * np.sin(latr)
    return np.array((x, y, z)).T


def direction_vec(points: np.ndarray, reference: np.ndarray, epsilon: float = 10e-11) -> np.ndarray:
    """Direction vector computation.

    Compute the direction vector of a set of points with respect to a reference
    vector.

    Parameters
    ----------
    points : np.array of shape (num_points, 3)
        The points to compute the direction vector.
    reference : np.array of shape (3, )
        The reference vector.
    epsilon : float, optional
        The value to add to the first vector to avoid division by zero. Defaults to 10e-11.

    Returns
    -------
    np.array of shape (3, num_points)
        The direction vector of the cross product of the two vectors.
    """
    v = np.cross(points, reference)
    vnorm1 = np.power(v, 2).sum(axis=-1)
    redo_idx = np.where(vnorm1 < epsilon)[0]
    if len(redo_idx) > 0:
        points[redo_idx] += epsilon
        v = np.cross(points, reference)
        vnorm1 = np.power(v, 2).sum(axis=-1)
    return v.T / np.sqrt(vnorm1)
