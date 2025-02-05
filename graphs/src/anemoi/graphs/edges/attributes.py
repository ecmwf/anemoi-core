# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from __future__ import annotations

import logging

import torch
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.typing import Adj
from torch_geometric.typing import PairTensor
from torch_geometric.typing import Size

from anemoi.graphs.edges.directional import compute_directions
from anemoi.graphs.normalise import NormaliserMixin
from anemoi.graphs.utils import haversine_distance

LOGGER = logging.getLogger(__name__)


class BaseEdgeAttributeBuilder(MessagePassing, NormaliserMixin):
    """Base class for edge attribute builders."""

    def __init__(self, norm: str | None = None, dtype: str = "float32") -> None:
        super().__init__()
        self._idx_lat = 0
        self._idx_lon = 1
        self.norm = norm
        self.dtype = dtype

    def forward(self, x: PairTensor, edge_index: Adj, size: Size = None) -> torch.Tensor:
        return self.propagate(edge_index, x=x, size=size)

    def compute(self, x_i: torch.Tensor, x_j: torch.Tensor) -> torch.Tensor:
        """Compute edge features.

        Parameters
        ----------
        x_i : torch.Tensor
            Coordinates of the source nodes.
        x_j : torch.Tensor
            Coordinates of the target nodes.

        Returns
        -------
        torch.Tensor
            Edge features.
        """
        raise NotImplementedError("Method `compute` must be implemented.")

    def message(self, x_i: torch.Tensor, x_j: torch.Tensor) -> torch.Tensor:
        edge_features = self.compute(x_i, x_j)

        if edge_features.ndim == 1:
            edge_features = edge_features.unsqueeze(-1)

        return self.normalise(edge_features)

    def aggregate(self, edge_features: torch.Tensor) -> torch.Tensor:
        return edge_features

class EdgeLength(BaseEdgeAttributeBuilder):
    """Computes edge length for bipartite graphs."""

    def compute(self, x_i: torch.Tensor, x_j: torch.Tensor) -> torch.Tensor:
        edge_length = haversine_distance(x_i, x_j)
        return edge_length


class EdgeDirection(BaseEdgeAttributeBuilder):
    """Computes edge direction for bipartite graphs."""

    def compute(self, x_i: torch.Tensor, x_j: torch.Tensor) -> torch.Tensor:
        edge_dirs = compute_directions(x_i, x_j)
        return edge_dirs


class Azimuth(BaseEdgeAttributeBuilder):
    """Compute the azimuth of the edge.

    Attributes
    ----------
    norm : Optional[str]
        Normalisation method. Options: None, "l1", "l2", "unit-max", "unit-range", "unit-std".
    invert : bool
        Whether to invert the edge lengths, i.e. 1 - edge_length. Defaults to False.

    Methods
    -------
    compute(graph, source_name, target_name)
        Compute edge lengths attributes.

    References
    ----------
    - https://www.movable-type.co.uk/scripts/latlong.html
    """

    def compute(self, x_i: torch.Tensor, x_j: torch.Tensor) -> torch.Tensor:
        # Forward bearing. x_i, x_j must bez radians.
        a11 = torch.cos(x_i[:, self._idx_lat]) * torch.sin(x_j[:, self._idx_lat])
        a12 = (
            torch.sin(x_i[:, self._idx_lat])
            * torch.cos(x_j[:, self._idx_lat])
            * torch.cos(x_j[..., self._idx_lon] - x_i[..., self._idx_lon])
        )
        a1 = a11 - a12
        a2 = torch.sin(x_j[..., self._idx_lon] - x_i[..., self._idx_lon]) * torch.cos(x_j[:, self._idx_lat])
        edge_dirs = torch.atan2(a2, a1)

        return edge_dirs
