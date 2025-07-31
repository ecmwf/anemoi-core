# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import torch

from anemoi.graphs.generate.transforms import latlon_rad_to_cartesian

NORTH_POLE = [0, 0, 1]  # North pole in 3D coordinates


def direction_vec(points: torch.Tensor, reference: torch.Tensor, epsilon: float = 1e-12) -> torch.Tensor:
    """Compute the unit direction vector from b to a in torch."""
    v = torch.cross(points, reference, dim=-1)
    vnorm1 = torch.pow(v, 2).sum(dim=-1)
    redo_idx = torch.nonzero(vnorm1 < epsilon, as_tuple=False).squeeze()

    if redo_idx.numel() > 0:
        points[redo_idx] += epsilon
        v = torch.cross(points, reference, dim=-1)
        vnorm1 = torch.pow(v, 2).sum(dim=-1)

    return v / torch.sqrt(vnorm1).unsqueeze(-1)  # normalize across last dimension


def rotate_vectors(v: torch.Tensor, axis: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
    """Rotate points v around axis by the angle using Rodrigues' rotation formula in torch.

    Parameters
    ----------
    v : torch.Tensor
        A tensor of shape (N, 3) representing N vectors to be rotated.
    axis : torch.Tensor
        A tensor of shape (N, 3) representing the rotation axis.
    angle : torch.Tensor
        A tensor of shape (N,) representing the rotation angles.

    Returns
    -------
    torch.Tensor
        A tensor of shape (N, 3) representing the rotated locations.

    Notes
    -----
    - https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula
    """
    axis = axis / torch.norm(axis, dim=-1, keepdim=True)  # Ensure the axis is a unit vector
    cos_theta = torch.cos(angle).unsqueeze(-1)
    sin_theta = torch.sin(angle).unsqueeze(-1)

    s1 = v * cos_theta
    s2 = torch.cross(axis, v, dim=-1) * sin_theta
    s3 = axis * torch.sum(v * axis, dim=-1, keepdim=True) * (1 - cos_theta)
    v_rot = s1 + s2 + s3

    return v_rot


def compute_directions(source_coords: torch.Tensor, target_coords: torch.Tensor) -> torch.Tensor:
    """Compute the direction of the edge.

    Parameters
    ----------
    source_coords : torch.Tensor of shape (N, 2)
        A tensor of shape (N, 2) representing the lat-lon location of the N head nodes.
    target_coords : torch.Tensor of shape (N, 2)
        A tensor of shape (N, 2) representing the lat-lon location of the N tail nodes.

    Returns
    -------
    torch.Tensor of shape (N, 2)
        The direction of the edge.
    """
    north_pole = torch.tensor([NORTH_POLE], dtype=source_coords.dtype).to(device=source_coords.device)
    source_coords_xyz = latlon_rad_to_cartesian(source_coords, 1.0)
    target_coords_xyz = latlon_rad_to_cartesian(target_coords, 1.0)

    # Compute the unit direction vector & the angle theta between target coords and the north pole.
    v_unit = direction_vec(target_coords_xyz, north_pole)
    theta = torch.acos(
        torch.clamp(torch.sum(target_coords_xyz * north_pole, dim=1), -1.0, 1.0)
    )  # Clamp for numerical stability

    # Rotate source coords by angle theta around v_unit axis.
    rotated_source_coords_xyz = rotate_vectors(source_coords_xyz, v_unit, theta)

    # Compute the direction from the rotated vector to the north pole.
    direction = direction_vec(rotated_source_coords_xyz, north_pole)

    # All 3rd components should be 0s
    assert torch.all(direction[:, 2] == 0), "Rotation should be aligned with the north pole"
    return direction[:, :2]
