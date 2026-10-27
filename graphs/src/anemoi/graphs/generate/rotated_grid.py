# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import torch

from anemoi.graphs.generate.transforms import latlon_rad_to_cartesian


def rotation_matrix(centre_lat: float, centre_lon: float, rotation: float = 0.0) -> torch.Tensor:
    """Get the rotation that moves the centre to latitude 0 and longitude 0.

    With ``rotation`` set to zero, rotated north is true north along the meridian through the
    centre. Otherwise, the rotated axes are turned anticlockwise about the centre by ``rotation``
    degrees, so that rotated east points ``rotation`` degrees north of true east at the centre.

    Parameters
    ----------
    centre_lat : float
        Latitude of the centre, in degrees.
    centre_lon : float
        Longitude of the centre, in degrees.
    rotation : float, optional
        Anticlockwise angle of the rotated axes about the centre, in degrees. Defaults to 0.

    Returns
    -------
    torch.Tensor of shape (3, 3)
        Rotation matrix acting on cartesian coordinates, in float64.
    """
    lat = torch.deg2rad(torch.tensor(centre_lat, dtype=torch.float64))
    lon = torch.deg2rad(torch.tensor(centre_lon, dtype=torch.float64))
    angle = torch.deg2rad(torch.tensor(rotation, dtype=torch.float64))
    zero, one = torch.zeros((), dtype=torch.float64), torch.ones((), dtype=torch.float64)
    rot_z = torch.stack(
        [
            torch.stack([torch.cos(lon), torch.sin(lon), zero]),
            torch.stack([-torch.sin(lon), torch.cos(lon), zero]),
            torch.stack([zero, zero, one]),
        ]
    )
    rot_y = torch.stack(
        [
            torch.stack([torch.cos(lat), zero, torch.sin(lat)]),
            torch.stack([zero, one, zero]),
            torch.stack([-torch.sin(lat), zero, torch.cos(lat)]),
        ]
    )
    rot_x = torch.stack(
        [
            torch.stack([one, zero, zero]),
            torch.stack([zero, torch.cos(angle), torch.sin(angle)]),
            torch.stack([zero, -torch.sin(angle), torch.cos(angle)]),
        ]
    )
    return rot_x @ rot_y @ rot_z


def _rotate(coords_rad: torch.Tensor, rotation: torch.Tensor) -> torch.Tensor:
    """Rotate geographic coordinates using the given rotation matrix.

    Parameters
    ----------
    coords_rad : torch.Tensor
        Tensor of shape (..., 2) containing latitude and longitude in radians.
    rotation : torch.Tensor
        Rotation matrix of shape (3, 3) acting on cartesian coordinates.

    Returns
    -------
    torch.Tensor of shape (..., 2)
        Rotated latitude and longitude in radians.
    """
    xyz = latlon_rad_to_cartesian(coords_rad.to(torch.float64)) @ rotation.T
    lat = torch.asin(xyz[..., 2].clamp(-1, 1))
    lon = torch.atan2(xyz[..., 1], xyz[..., 0])
    return torch.stack([lat, lon], dim=-1)


def geographic_to_rotated(coords_rad: torch.Tensor, rotation: torch.Tensor) -> torch.Tensor:
    """Map geographic (lat, lon) coordinates, in radians, to the rotated frame.

    Parameters
    ----------
    coords_rad : torch.Tensor
        Tensor of shape (..., 2) containing latitude and longitude in radians.
    rotation : torch.Tensor
        Rotation matrix of shape (3, 3) acting on cartesian coordinates.

    Returns
    -------
    torch.Tensor of shape (..., 2)
        Rotated latitude and longitude in radians.
    """
    return _rotate(coords_rad, rotation)


def rotated_to_geographic(coords_rad: torch.Tensor, rotation: torch.Tensor) -> torch.Tensor:
    """Map rotated (lat, lon) coordinates, in radians, back to geographic coordinates.

    Parameters
    ----------
    coords_rad : torch.Tensor
        Tensor of shape (..., 2) containing latitude and longitude in radians.
    rotation : torch.Tensor
        Rotation matrix of shape (3, 3) acting on cartesian coordinates.

    Returns
    -------
    torch.Tensor of shape (..., 2)
        Geographic latitude and longitude in radians.
    """
    return _rotate(coords_rad, rotation.T)
