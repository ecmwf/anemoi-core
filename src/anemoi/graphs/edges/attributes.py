# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from __future__ import annotations

from abc import ABC
from abc import abstractmethod

import numpy as np
import torch
from torch_geometric.data import HeteroData

from anemoi.graphs.edges.directional import directional_edge_features
from anemoi.graphs.normalise import NormaliserMixin
from anemoi.graphs.utils import haversine_distance


class BaseEdgeAttribute(ABC, NormaliserMixin):
    """Base class for edge attributes."""

    def __init__(self, norm: str | None = None) -> None:
        self.norm = norm

    @abstractmethod
    def get_raw_values(self, graph: HeteroData, source_name: str, target_name: str, *args, **kwargs) -> np.ndarray: ...

    def post_process(self, values: np.ndarray) -> torch.Tensor:
        """Post-process the values."""
        if values.ndim == 1:
            values = values[:, np.newaxis]

        normed_values = self.normalise(values)

        return torch.tensor(normed_values, dtype=torch.float32)

    def compute(self, graph: HeteroData, edges_name: tuple[str, str, str], *args, **kwargs) -> torch.Tensor:
        """Compute the edge attributes."""
        source_name, _, target_name = edges_name
        assert (
            source_name in graph.node_types
        ), f"Node \"{source_name}\" not found in graph. Optional nodes are {', '.join(graph.node_types)}."
        assert (
            target_name in graph.node_types
        ), f"Node \"{target_name}\" not found in graph. Optional nodes are {', '.join(graph.node_types)}."

        values = self.get_raw_values(graph, source_name, target_name, *args, **kwargs)
        return self.post_process(values)


class EdgeDirection(BaseEdgeAttribute):
    """Edge direction feature.

    This class calculates the direction of an edge using either:
    1. Rotated features: The target nodes are rotated to the north pole to compute the edge direction.
    2. Non-rotated features: The direction is computed as the difference in latitude and longitude between the source
    and target nodes.

    The resulting direction is represented as a unit vector starting at (0, 0), with X and Y components.

    Attributes
    ----------
    norm : Optional[str]
        Normalisation method. Options: None, "l1", "l2", "unit-max", "unit-range", "unit-std".
    luse_rotated_features : bool
        Whether to use rotated features.

    Methods
    -------
    compute(graph, source_name, target_name)
        Compute direction of all edges.
    """

    def __init__(self, norm: str | None = None, luse_rotated_features: bool = True) -> None:
        super().__init__(norm)
        self.luse_rotated_features = luse_rotated_features

    def get_raw_values(self, graph: HeteroData, source_name: str, target_name: str) -> np.ndarray:
        """Compute directional features for edges.

        Parameters
        ----------
        graph : HeteroData
            The graph.
        source_name : str
            The name of the source nodes.
        target_name : str
            The name of the target nodes.

        Returns
        -------
        np.ndarray
            The directional features.
        """
        edge_index = graph[(source_name, "to", target_name)].edge_index
        source_coords = graph[source_name].x.numpy()[edge_index[0]].T
        target_coords = graph[target_name].x.numpy()[edge_index[1]].T
        edge_dirs = directional_edge_features(source_coords, target_coords, self.luse_rotated_features).T
        return edge_dirs


class EdgeLength(BaseEdgeAttribute):
    """Edge length feature.

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
    """

    def __init__(self, norm: str | None = None, invert: bool = False) -> None:
        super().__init__(norm)
        self.invert = invert

    def get_raw_values(self, graph: HeteroData, source_name: str, target_name: str) -> np.ndarray:
        """Compute haversine distance (in kilometers) between nodes connected by edges.

        Parameters
        ----------
        graph : HeteroData
            The graph.
        source_name : str
            The name of the source nodes.
        target_name : str
            The name of the target nodes.

        Returns
        -------
        np.ndarray
            The edge lengths.
        """
        edge_index = graph[(source_name, "to", target_name)].edge_index
        source_coords = graph[source_name].x.numpy()[edge_index[0]]
        target_coords = graph[target_name].x.numpy()[edge_index[1]]
        edge_lengths = haversine_distance(source_coords, target_coords)
        return edge_lengths

    def post_process(self, values: np.ndarray) -> torch.Tensor:
        """Post-process edge lengths."""
        values = super().post_process(values)

        if self.invert:
            values = 1 - values

        return values
