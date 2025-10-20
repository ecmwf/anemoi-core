# (C) Copyright 2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


from abc import ABC
from abc import abstractmethod
from typing import Optional

import numpy as np
import torch
from torch import Tensor
from torch import nn
from torch_geometric.data import HeteroData
from torch_geometric.typing import Adj

from anemoi.models.layers.graph import TrainableTensor


def create_graph_provider(
    sub_graph: Optional[HeteroData] = None,
    sub_graph_edge_attributes: Optional[list[str]] = None,
    src_grid_size: Optional[int] = None,
    dst_grid_size: Optional[int] = None,
    trainable_size: int = 0,
    edge_dim: Optional[int] = None,
) -> "BaseGraphProvider":
    """Factory function to create appropriate graph provider.

    Parameters
    ----------
    sub_graph : HeteroData, optional
        Sub graph of the full structure (for static mode)
    sub_graph_edge_attributes : list[str], optional
        Edge attributes to use (for static mode)
    src_grid_size : int, optional
        Source grid size (for static mode)
    dst_grid_size : int, optional
        Destination grid size (for static mode)
    trainable_size : int, optional
        Trainable tensor size, by default 0
    edge_dim : int, optional
        Edge dimension (for dynamic mode)

    Returns
    -------
    BaseGraphProvider
        Appropriate graph provider instance (StaticGraphProvider or DynamicGraphProvider)
    """
    if sub_graph is not None:
        # Static mode
        assert sub_graph_edge_attributes is not None, "sub_graph_edge_attributes required for static mode"
        assert src_grid_size is not None, "src_grid_size required for static mode"
        assert dst_grid_size is not None, "dst_grid_size required for static mode"
        return StaticGraphProvider(
            sub_graph=sub_graph,
            edge_attributes=sub_graph_edge_attributes,
            src_size=src_grid_size,
            dst_size=dst_grid_size,
            trainable_size=trainable_size,
        )
    else:
        # Dynamic mode
        assert trainable_size == 0, "Dynamic mode does not support trainable edge parameters (trainable_size must be 0)"
        assert edge_dim is not None, "edge_dim required for dynamic mode"
        return DynamicGraphProvider(edge_dim=edge_dim)


class BaseGraphProvider(nn.Module, ABC):
    """Base class for graph edge providers.

    Graph providers encapsulate the logic for supplying edge indices and attributes
    to mapper and processor layers. This allows for different strategies (static, dynamic, etc.)
    to be implemented and swapped without modifying the mapper classes.
    """

    @abstractmethod
    def get_edges(self, batch_size: Optional[int] = None) -> tuple[Tensor, Adj]:
        """Get edge information.

        Parameters
        ----------
        batch_size : int, optional
            Number of times to expand the edge index (used by static mode)

        Returns
        -------
        tuple[Tensor, Adj]
            Edge attributes and edge index
        """
        pass

    @property
    @abstractmethod
    def edge_dim(self) -> int:
        """Return the edge dimension."""
        pass


class StaticGraphProvider(BaseGraphProvider):
    """Provider for static graphs with fixed edge structure.

    This provider owns all graph-related state including edge attributes,
    edge indices, and trainable parameters.
    """

    def __init__(
        self,
        sub_graph: HeteroData,
        edge_attributes: list[str],
        src_size: int,
        dst_size: int,
        trainable_size: int,
    ) -> None:
        """Initialize StaticGraphProvider.

        Parameters
        ----------
        sub_graph : HeteroData
            Sub graph of the full structure
        edge_attributes : list[str]
            Edge attributes to use
        src_size : int
            Source grid size
        dst_size : int
            Destination grid size
        trainable_size : int
            Size of trainable edge parameters
        """
        super().__init__()

        assert sub_graph, "StaticGraphProvider needs a valid sub_graph to register edges."
        assert edge_attributes is not None, "Edge attributes must be provided"

        edge_attr_tensor = torch.cat([sub_graph[attr] for attr in edge_attributes], axis=1)

        self.register_buffer("edge_attr", edge_attr_tensor, persistent=False)
        self.register_buffer("edge_index_base", sub_graph.edge_index, persistent=False)
        self.register_buffer(
            "edge_inc", torch.from_numpy(np.asarray([[src_size], [dst_size]], dtype=np.int64)), persistent=True
        )

        self.trainable = TrainableTensor(trainable_size=trainable_size, tensor_size=edge_attr_tensor.shape[0])

        self._edge_dim = edge_attr_tensor.shape[1] + trainable_size

    @property
    def edge_dim(self) -> int:
        """Return the edge dimension."""
        return self._edge_dim

    def _expand_edges(self, edge_index: Adj, edge_inc: Tensor, batch_size: int) -> Adj:
        """Expand edge index.

        Parameters
        ----------
        edge_index : Adj
            Edge index to start
        edge_inc : Tensor
            Edge increment to use
        batch_size : int
            Number of times to expand the edge index

        Returns
        -------
        Adj
            Expanded edge index
        """
        edge_index = torch.cat(
            [edge_index + i * edge_inc for i in range(batch_size)],
            dim=1,
        )
        return edge_index

    def get_edges(
        self,
        batch_size: int,
    ) -> tuple[Tensor, Adj]:
        """Get edge attributes and expanded edge index for static graph.

        Parameters
        ----------
        batch_size : int
            Number of times to expand the edge index

        Returns
        -------
        tuple[Tensor, Adj]
            Edge attributes and expanded edge index
        """
        edge_attr = self.trainable(self.edge_attr, batch_size)
        edge_index = self._expand_edges(self.edge_index_base, self.edge_inc, batch_size)
        return edge_attr, edge_index


class DynamicGraphProvider(BaseGraphProvider):
    """Provider for dynamic graphs where edges are supplied at runtime.

    Current implementation requires edge indices and attributes to be passed in during
    forward pass via get_edges(). Does not support trainable edge parameters.

    Future implementation will support on-the-fly graph construction via build_graph()
    (e.g., k-NN graphs, radius graphs, adaptive connectivity).
    """

    def __init__(self, edge_dim: int) -> None:
        """Initialize DynamicGraphProvider.

        Parameters
        ----------
        edge_dim : int
            Expected dimension of edge attributes
        """
        super().__init__()
        self._edge_dim = edge_dim

    @property
    def edge_dim(self) -> int:
        """Return the edge dimension."""
        return self._edge_dim

    def build_graph(self, src_nodes: Tensor, dst_nodes: Tensor, **kwargs) -> tuple[Tensor, Adj]:
        """Build graph dynamically from source and destination nodes.

        This method will be implemented in the future to support on-the-fly
        graph construction (e.g., k-NN graphs, radius graphs, etc.).

        Parameters
        ----------
        src_nodes : Tensor
            Source node features/positions
        dst_nodes : Tensor
            Destination node features/positions
        **kwargs
            Additional parameters for graph construction algorithm

        Returns
        -------
        tuple[Tensor, Adj]
            Edge attributes and edge index

        Raises
        ------
        NotImplementedError
            This functionality is not yet implemented
        """
        raise NotImplementedError("Dynamic graph construction is not yet implemented. ")

    def get_edges(
        self,
        batch_size: Optional[int] = None,
    ) -> tuple[Tensor, Adj]:
        """Get dynamic edges.

        This method will be implemented in the future to return edges
        constructed on-the-fly via build_graph().

        Parameters
        ----------
        batch_size : int, optional
            Batch size (currently unused, reserved for future implementation)

        Returns
        -------
        tuple[Tensor, Adj]
            Edge attributes and edge index

        Raises
        ------
        NotImplementedError
            This functionality is not yet implemented
        """
        raise NotImplementedError("Dynamic graph edge retrieval is not yet implemented. ")
