# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import numpy as np
import pytest
import torch

from anemoi.graphs.plotting.prepare import edge_list_from_coordinates


def test_edge_list_from_coordinates_converts_edge_index_to_line_segments() -> None:
    source_coords = torch.tensor([[0.0, 0.0], [np.pi / 4, np.pi / 2]])
    target_coords = torch.tensor([[np.pi / 2, np.pi], [-np.pi / 4, 3 * np.pi / 2]])
    edge_index = torch.tensor([[0, 1], [1, 0]])

    latitudes, longitudes = edge_list_from_coordinates(source_coords, target_coords, edge_index)

    np.testing.assert_allclose(latitudes, np.array([0.0, -45.0, np.nan, 45.0, 90.0, np.nan]))
    np.testing.assert_allclose(longitudes, np.array([0.0, 270.0, np.nan, 90.0, 180.0, np.nan]))


def test_edge_list_from_coordinates_validates_coordinate_shape() -> None:
    source_coords = torch.zeros((2, 3))
    target_coords = torch.zeros((2, 2))
    edge_index = torch.tensor([[0], [0]])

    with pytest.raises(ValueError, match="source_coords must have shape"):
        edge_list_from_coordinates(source_coords, target_coords, edge_index)


def test_edge_list_from_coordinates_validates_edge_index_bounds() -> None:
    source_coords = torch.zeros((2, 2))
    target_coords = torch.zeros((2, 2))
    edge_index = torch.tensor([[2], [0]])

    with pytest.raises(IndexError, match="source indices"):
        edge_list_from_coordinates(source_coords, target_coords, edge_index)
