# (C) Copyright 2025- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import torch

from anemoi.graphs.generate.transforms import latlon_rad_to_sincos
from anemoi.graphs.generate.transforms import sincos_to_latlon_rad

NUM_POINTS = 40


def test_latlon_rad_to_sincos():
    """Test conversion from lat-lon in radians to sin-cos representation."""
    coords = torch.rand(NUM_POINTS, 2) * torch.tensor([torch.pi, 2 * torch.pi]) - torch.tensor([torch.pi / 2, 0])
    # lat in [-pi/2, pi/2], lon in [0, 2 * pi]

    sincos = latlon_rad_to_sincos(coords)
    assert sincos.shape == (NUM_POINTS, 4)
    assert sincos.dtype == coords.dtype
    assert sincos.device == coords.device
    assert sincos.requires_grad == coords.requires_grad

    assert torch.all((sincos[:, :2] >= -1) & (sincos[:, :2] <= 1))  # sin(lat), sin(lon)
    assert torch.all((sincos[:, 2] >= 0) & (sincos[:, 2] <= 1))  # cos(lat)
    assert torch.all((sincos[:, 3] >= -1) & (sincos[:, 3] <= 1))  # cos(lon)


def test_sincos_to_latlon_rad():
    """Test conversion from sin-cos representation to lat-lon in radians."""
    sincos = torch.rand(NUM_POINTS, 4) * torch.tensor([2, 2, 1, 2]) - torch.tensor([1, 1, 0, 1])
    # sin(lat), sin(lon), cos(lon) in [-1, 1], cos(lat) in [0, 1]

    coords = sincos_to_latlon_rad(sincos)
    assert coords.shape == (NUM_POINTS, 2)
    assert coords.dtype == sincos.dtype
    assert coords.device == sincos.device
    assert coords.requires_grad == sincos.requires_grad

    assert torch.all((coords[:, 0] >= -torch.pi / 2) & (coords[:, 0] <= torch.pi / 2))  # lat in [-pi/2, pi/2]
    assert torch.all((coords[:, 1] >= 0) & (coords[:, 1] <= 2 * torch.pi))  # lon in [0, 2 * pi]


def test_latlon_rad_to_sincos_and_back():
    """Test conversion from lat-lon in radians to sin-cos representation and back."""
    coords = torch.rand(NUM_POINTS, 2) * torch.tensor([torch.pi, 2 * torch.pi]) - torch.tensor([torch.pi / 2, 0])

    sincos = latlon_rad_to_sincos(coords)
    assert sincos.shape == (NUM_POINTS, 4)

    recovered = sincos_to_latlon_rad(sincos)
    assert recovered.shape == (NUM_POINTS, 2)

    torch.testing.assert_close(recovered, coords)
