# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import logging
from abc import ABC
from abc import abstractmethod

import torch
from torch_geometric.data.storage import NodeStorage
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.typing import Adj
from torch_geometric.typing import PairTensor
from torch_geometric.typing import Size

from anemoi.graphs.edges.directional import compute_directions
from anemoi.graphs.normalise import NormaliserMixin
from anemoi.graphs.utils import NodesAxis
from anemoi.graphs.utils import get_distributed_device
from anemoi.graphs.utils import haversine_distance

LOGGER = logging.getLogger(__name__)


class BaseEdgeAttributeBuilder(MessagePassing, NormaliserMixin, ABC):
    """Base class for edge attribute builders."""

    node_attr_name: str = None
    norm_by_group: bool = False

    def __init__(self, norm: str | None = None, dtype: str = "float32") -> None:
        super().__init__()
        self.norm = norm
        self.dtype = dtype
        self.device = get_distributed_device()
        if self.node_attr_name is None:
            error_msg = f"Class {self.__class__.__name__} must define 'node_attr_name' either as a class attribute or in __init__"
            raise TypeError(error_msg)

    def subset_node_information(self, source_nodes: NodeStorage, target_nodes: NodeStorage) -> PairTensor:
        if self.node_attr_name in source_nodes:
            source_nodes_data = source_nodes[self.node_attr_name].to(self.device)
        else:
            source_nodes_data = None
            LOGGER.warning("The attribute %s is not in the source nodes.", self.node_attr_name)

        if self.node_attr_name in target_nodes:
            target_nodes_data = target_nodes[self.node_attr_name].to(self.device)
        else:
            target_nodes_data = None
            LOGGER.warning("The attribute %s is not in the target nodes.", self.node_attr_name)

        return source_nodes_data, target_nodes_data

    def forward(self, x: tuple[NodeStorage, NodeStorage], edge_index: Adj, size: Size = None) -> torch.Tensor:
        x = self.subset_node_information(*x)
        return self.propagate(edge_index, x=x, size=size)

    @abstractmethod
    def compute(self, x_i: torch.Tensor, x_j: torch.Tensor) -> torch.Tensor: ...

    def message(self, x_i: torch.Tensor, x_j: torch.Tensor) -> torch.Tensor:
        edge_features = self.compute(x_i, x_j)

        if edge_features.ndim == 1:
            edge_features = edge_features.unsqueeze(-1)

        return edge_features

    def aggregate(self, edge_features: torch.Tensor, index: torch.Tensor, ptr=None, dim_size=None) -> torch.Tensor:
        return self.normalise(edge_features, index, dim_size)


class BasePositionalBuilder(BaseEdgeAttributeBuilder, ABC):
    node_attr_name: str = "x"
    _idx_lat: int = 0
    _idx_lon: int = 1


class EdgeLength(BasePositionalBuilder):
    """Computes edge length for bipartite graphs."""

    def compute(self, x_i: torch.Tensor, x_j: torch.Tensor) -> torch.Tensor:
        edge_length = haversine_distance(x_i, x_j)
        return edge_length


class EdgeDirection(BasePositionalBuilder):
    """Computes edge direction for bipartite graphs."""

    def compute(self, x_i: torch.Tensor, x_j: torch.Tensor) -> torch.Tensor:
        edge_dirs = compute_directions(x_i, x_j)
        return edge_dirs


class Azimuth(BasePositionalBuilder):
    """Compute the azimuth of the edge.

    Attributes
    ----------
    norm : str | None
        Normalisation method. Options: None, "l1", "l2", "unit-max", "unit-range", "unit-std".
    invert : bool
        Whether to invert the edge lengths, i.e. 1 - edge_length. Defaults to False.

    Methods
    -------
    compute(x_i, x_j)
        Compute edge lengths attributes.

    References
    ----------
    - https://www.movable-type.co.uk/scripts/latlong.html
    """

    def compute(self, x_i: torch.Tensor, x_j: torch.Tensor) -> torch.Tensor:
        # Forward bearing. x_i, x_j must be radians.
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


class BaseBooleanEdgeAttributeBuilder(BaseEdgeAttributeBuilder, ABC):
    """Base class for boolean edge attributes."""

    def __init__(self) -> None:
        super().__init__(norm=None, dtype="bool")


class BaseEdgeAttributeFromNodeBuilder(BaseBooleanEdgeAttributeBuilder, ABC):
    """Base class for propagating an attribute from the nodes to the edges."""

    nodes_axis: NodesAxis | None = None

    def __init__(self, node_attr_name: str) -> None:
        self.node_attr_name = node_attr_name
        super().__init__()
        if self.nodes_axis is None:
            raise AttributeError(f"{self.__class__.__name__} class must set 'nodes_axis' attribute.")

    def compute(self, x_i: torch.Tensor, x_j: torch.Tensor) -> torch.Tensor:
        node_attr = (x_j, x_i)[self.nodes_axis.value]
        assert (
            node_attr is not None
        ), f"The node attribute specified for {self.node_attr_name} cannot be found in the nodes."
        return node_attr


class AttributeFromSourceNode(BaseEdgeAttributeFromNodeBuilder):
    """Copy an attribute of the source node to the edge."""

    nodes_axis = NodesAxis.SOURCE


class AttributeFromTargetNode(BaseEdgeAttributeFromNodeBuilder):
    """Copy an attribute of the target node to the edge."""

    nodes_axis = NodesAxis.TARGET


class GaussianDistanceWeights(EdgeLength):
    """Gaussian distance weights."""

    norm_by_group: bool = True  # normalise the gaussian weights by target node

    def __init__(self, sigma: float = 1.0, norm: str = "l1", **kwargs) -> None:
        self.sigma = sigma
        super().__init__(norm=norm)

    def compute(self, x_i: torch.Tensor, x_j: torch.Tensor) -> torch.Tensor:
        dists = super().compute(x_i, x_j)
        gaussian_weights = torch.exp(-(dists**2) / (2 * self.sigma**2))
        return gaussian_weights
