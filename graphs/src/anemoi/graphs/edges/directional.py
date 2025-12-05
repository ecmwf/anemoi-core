# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from __future__ import annotations

import torch

from anemoi.graphs.generate.transforms import latlon_rad_to_cartesian

NORTH_POLE = [0.0, 0.0, 1.0]  # North pole in 3D coordinates


def rotate_vectors(v: torch.Tensor, axis: torch.Tensor, angle: torch.Tensor, eps=1e-8) -> torch.Tensor:
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
    axis_norm = torch.norm(axis, dim=-1, keepdim=True)  # Ensure the axis is a unit vector
    axis = axis / torch.clamp(axis_norm, min=eps)

    cos_theta = torch.cos(angle).unsqueeze(-1)
    sin_theta = torch.sin(angle).unsqueeze(-1)

    s1 = v * cos_theta
    s2 = torch.cross(axis, v, dim=-1) * sin_theta
    s3 = axis * torch.sum(v * axis, dim=-1, keepdim=True) * (1 - cos_theta)
    v_rot = s1 + s2 + s3

    return v_rot


def compute_directions(source_coords: torch.Tensor, target_coords: torch.Tensor) -> torch.Tensor:
    """Compute the direction of the edge on the tangent plane.

    1. Rotate coordinate system so target_coords is at north pole (0, 0, 1)
    2. Rotate source_coords by the same rotation
    3. Project rotated source onto tangent plane at north pole (xy-plane)
    4. Direction = (x, y) components pointing toward rotated source

    Parameters
    ----------
    source_coords : torch.Tensor
        Source coordinates in lat/lon (radians), shape (N, 2)
    target_coords : torch.Tensor
        Target coordinates in lat/lon (radians), shape (N, 2)

    Returns
    -------
    torch.Tensor
        Unit direction vectors on tangent plane, shape (N, 2)
    """
    epsilon = 1e-8

    source_coords_xyz = latlon_rad_to_cartesian(source_coords, 1.0)
    target_coords_xyz = latlon_rad_to_cartesian(target_coords, 1.0)
    north_pole = torch.tensor([NORTH_POLE], dtype=source_coords.dtype, device=source_coords.device)

    # Compute dot product with north pole -> pole cases
    dot = (target_coords_xyz * north_pole).sum(dim=1, keepdim=True)

    # masks
    at_north = (dot > 1.0 - epsilon).squeeze(1)
    at_south = (dot < -1.0 + epsilon).squeeze(1)
    normal = ~(at_north | at_south)

    # init output
    direction = torch.zeros(
        (source_coords_xyz.shape[0], 2), dtype=source_coords_xyz.dtype, device=source_coords_xyz.device
    )

    # target at north pole - no rotation needed
    if at_north.any():
        xy = source_coords_xyz[at_north, :2]
        direction[at_north] = xy / torch.clamp(torch.linalg.norm(xy, dim=1, keepdim=True), min=epsilon)

    # target at south pole - rotate 180° around x-axis
    if at_south.any():
        rotated_south = source_coords_xyz[at_south].clone()
        # flip y and z
        rotated_south[:, 1] *= -1
        rotated_south[:, 2] *= -1
        xy = rotated_south[:, :2]
        direction[at_south] = xy / torch.clamp(torch.linalg.norm(xy, dim=1, keepdim=True), min=epsilon)

    # rotation via cross product
    if normal.any():
        target_normal = target_coords_xyz[normal]
        source_normal = source_coords_xyz[normal]

        axis = torch.cross(target_normal, north_pole.expand_as(target_normal), dim=1)
        sin_theta = torch.linalg.norm(axis, dim=1, keepdim=True)  # ||target x north_pole||
        cos_theta = torch.clamp((target_normal * north_pole).sum(dim=1, keepdim=True), -1.0, 1.0)
        theta = torch.atan2(sin_theta, cos_theta).squeeze(1)

        axis_unit = axis / torch.clamp(sin_theta, min=epsilon)

        rotated_normal = rotate_vectors(source_normal, axis_unit, theta)
        xy = rotated_normal[:, :2]
        direction[normal] = xy / torch.clamp(torch.linalg.norm(xy, dim=1, keepdim=True), min=epsilon)

    return direction
