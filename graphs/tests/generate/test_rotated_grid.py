# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import pytest
import torch

from anemoi.graphs.generate.rotated_grid import geographic_to_rotated
from anemoi.graphs.generate.rotated_grid import rotated_to_geographic
from anemoi.graphs.generate.rotated_grid import rotation_matrix
from anemoi.graphs.generate.transforms import latlon_rad_to_cartesian

CENTRES = [(0.0, 0.0), (45.0, 10.0), (90.0, 0.0), (-60.0, 180.0), (30.0, -170.0)]


@pytest.mark.parametrize("centre", CENTRES)
@pytest.mark.parametrize("rotation", [0.0, 30.0, -120.0])
def test_centre_maps_to_origin(centre: tuple[float, float], rotation: float):
    coords = torch.deg2rad(torch.tensor([centre], dtype=torch.float64))
    rotated = geographic_to_rotated(coords, rotation_matrix(*centre, rotation))
    torch.testing.assert_close(rotated, torch.zeros_like(rotated), atol=1e-12, rtol=0)


@pytest.mark.parametrize("centre", CENTRES)
@pytest.mark.parametrize("rotation", [0.0, 30.0, -120.0])
def test_round_trip(centre: tuple[float, float], rotation: float):
    coords = torch.stack([torch.rand(100) * torch.pi - torch.pi / 2, torch.rand(100) * 2 * torch.pi], dim=-1)
    rotation = rotation_matrix(*centre, rotation)
    back = rotated_to_geographic(geographic_to_rotated(coords, rotation), rotation)
    torch.testing.assert_close(latlon_rad_to_cartesian(back), latlon_rad_to_cartesian(coords.to(torch.float64)))


def test_rotated_north_is_true_north():
    """Along the meridian through the centre, a point to the north has a positive rotated latitude."""
    rotation = rotation_matrix(40.0, 20.0)
    coords = torch.deg2rad(torch.tensor([[45.0, 20.0], [40.0, 25.0]], dtype=torch.float64))
    rotated = torch.rad2deg(geographic_to_rotated(coords, rotation))
    torch.testing.assert_close(rotated[0], torch.tensor([5.0, 0.0], dtype=torch.float64), atol=1e-9, rtol=0)
    assert rotated[1, 1] > 0


def test_rotation_turns_axes_anticlockwise():
    """Turned by 90 degrees, a point due north of the centre lies on the rotated east axis."""
    rotation = rotation_matrix(40.0, 20.0, 90.0)
    coords = torch.deg2rad(torch.tensor([[45.0, 20.0]], dtype=torch.float64))
    rotated = torch.rad2deg(geographic_to_rotated(coords, rotation))
    torch.testing.assert_close(rotated[0], torch.tensor([0.0, 5.0], dtype=torch.float64), atol=1e-9, rtol=0)
