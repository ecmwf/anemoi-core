# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Points of octahedral reduced Gaussian grids for tests."""

import math

import numpy as np
import torch
from torch_geometric.data import HeteroData


def octahedral_coords(n: int) -> torch.Tensor:
    """Latitude and longitude in radians of the points of the octahedral grid O``n``, row by row."""
    lengths = [4 * i + 20 for i in range(n)]
    lengths = lengths + lengths[::-1]
    latitudes = np.arcsin(np.polynomial.legendre.leggauss(2 * n)[0])[::-1]
    points = [(latitudes[row], 2 * math.pi * j / length) for row, length in enumerate(lengths) for j in range(length)]
    return torch.tensor(points, dtype=torch.float32)


def octahedral_graph(n: int, nodes_name: str = "data") -> HeteroData:
    """A graph whose ``nodes_name`` nodes are the points of the octahedral grid O``n``."""
    graph = HeteroData()
    graph[nodes_name].x = octahedral_coords(n)
    return graph
