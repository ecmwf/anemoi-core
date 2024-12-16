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
from scipy.spatial.transform import Rotation

from anemoi.graphs.generate.transforms import direction_vec
from anemoi.graphs.generate.transforms import latlon_rad_to_cartesian


def get_rotation_from_unit_vecs(points: np.ndarray, reference: np.ndarray) -> Rotation:
    """Compute rotation matrix of a set of points with respect to a reference vector.

    Parameters
    ----------
    points : np.ndarray of shape (num_points, 3)
        The points to compute the direction vector.
    reference : np.ndarray of shape (3, )
        The reference vector.

    Returns
    -------
    Rotation
        The rotation matrix that aligns the points with the reference vector.
    """
    assert points.shape[1] == 3, "Points must be in 3D"
    v_unit = direction_vec(points, reference)
    theta = np.arccos(np.dot(points, reference))
    return Rotation.from_rotvec(np.transpose(v_unit * theta))


def compute_directions(loc1: np.ndarray, loc2: np.ndarray, pole_vec: np.ndarray | None = None) -> np.ndarray:
    """Compute the direction of the edge joining the nodes considered.

    Parameters
    ----------
    loc1 : np.ndarray of shape (2, num_points)
        Location of the head nodes.
    loc2 : np.ndarray of shape (2, num_points)
        Location of the tail nodes.
    pole_vec : np.ndarray, optional
        The pole vector to rotate the points to. Defaults to the north pole.

    Returns
    -------
    np.ndarray of shape (3, num_points)
        The direction of the edge after rotating the north pole.
    """
    if pole_vec is None:
        pole_vec = np.array([0, 0, 1])

    # all will be rotated relative to destination node
    loc1_xyz = latlon_rad_to_cartesian(loc1, 1.0)
    loc2_xyz = latlon_rad_to_cartesian(loc2, 1.0)
    r = get_rotation_from_unit_vecs(loc2_xyz, pole_vec)
    direction = direction_vec(r.apply(loc1_xyz), pole_vec)
    return direction / np.sqrt(np.power(direction, 2).sum(axis=0))


def directional_edge_features(
    loc1: np.ndarray, loc2: np.ndarray, relative_to_rotated_target: bool = True
) -> np.ndarray:
    """Compute features of the edge joining the nodes considered.

    It computes the direction of the edge after rotating the north pole.

    Parameters
    ----------
    loc1 : np.ndarray of shpae (2, num_points)
        Location of the head node.
    loc2 : np.ndarray of shape (2, num_points)
        Location of the tail node.
    relative_to_rotated_target : bool, optional
        Whether to rotate the north pole to the target node. Defaults to True.

    Returns
    -------
    np.ndarray of shape of (2, num_points)
        Direction of the edge after rotation the north pole.
    """
    if relative_to_rotated_target:
        rotation = compute_directions(loc1, loc2)
        assert np.allclose(rotation[2], 0), "Rotation should be aligned with the north pole"
        return rotation[:2]

    return loc2 - loc1
