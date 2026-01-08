# (C) Copyright 2025 Anemoi contributors.
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
from pathlib import Path
from typing import Optional
from typing import Union

import numpy as np
import torch
from torch import Tensor
from torch import nn
from torch.distributed.distributed_c10d import ProcessGroup
from torch.utils.checkpoint import checkpoint
from torch_geometric.data import HeteroData
from torch_geometric.typing import Adj

from anemoi.models.distributed.khop_edges import shard_edges_1hop
from anemoi.models.layers.graph import TrainableTensor

LOGGER = logging.getLogger(__name__)


def create_graph_provider(
    graph: Optional[HeteroData] = None,
    edge_attributes: Optional[list[str]] = None,
    src_size: Optional[int] = None,
    dst_size: Optional[int] = None,
    trainable_size: int = 0,
) -> "BaseGraphProvider":
    """Factory function to create appropriate graph provider.

    Returns StaticGraphProvider if graph has edges,
    otherwise returns NoOpGraphProvider for edge-less architectures.

    Parameters
    ----------
    graph : HeteroData, optional
        Graph containing edges (for static mode)
    edge_attributes : list[str], optional
        Edge attributes to use (for static mode)
    src_size : int, optional
        Source grid size (for static mode)
    dst_size : int, optional
        Destination grid size (for static mode)
    trainable_size : int, optional
        Trainable tensor size, by default 0

    Returns
    -------
    BaseGraphProvider
        Appropriate graph provider instance
    """
    if graph:
        return StaticGraphProvider(
            graph=graph,
            edge_attributes=edge_attributes,
            src_size=src_size,
            dst_size=dst_size,
            trainable_size=trainable_size,
        )
    else:
        return NoOpGraphProvider()


class BaseGraphProvider(nn.Module, ABC):
    """Base class for graph edge providers.

    Graph providers encapsulate the logic for supplying edge indices and attributes
    to mapper and processor layers. This allows for different strategies (static, dynamic, etc.).
    """

    @abstractmethod
    def get_edges(
        self,
        batch_size: Optional[int] = None,
        src_coords: Optional[Tensor] = None,
        dst_coords: Optional[Tensor] = None,
        model_comm_group: Optional[ProcessGroup] = None,
        shard_edges: bool = True,
    ) -> Union[tuple[Tensor, Adj, Optional[tuple[list, list]]], Tensor]:
        """Get edge information.

        Parameters
        ----------
        batch_size : int, optional
            Number of times to expand the edge index (used by static mode)
        src_coords : Tensor, optional
            Source node coordinates (used by dynamic mode for k-NN, radius graphs, etc.)
        dst_coords : Tensor, optional
            Destination node coordinates (used by dynamic mode for k-NN, radius graphs, etc.)
        model_comm_group : ProcessGroup, optional
            Model communication group
        shard_edges : bool, optional
            Whether to shard edges, by default True

        Returns
        -------
        Union[tuple[Tensor, Adj, Optional[tuple[list, list]]], Tensor]
            For standard providers: (edge_attr, edge_index, edge_shard_shapes) tuple
            For sparse providers: sparse projection matrix
        """
        pass

    @property
    @abstractmethod
    def edge_dim(self) -> int:
        """Return the edge dimension."""
        pass

    @property
    def is_sparse(self) -> bool:
        """Whether this provider returns sparse matrices."""
        return False


class StaticGraphProvider(BaseGraphProvider):
    """Provider for static graphs with fixed edge structure.

    This provider owns all graph-related state including edge attributes,
    edge indices, and trainable parameters.
    """

    def __init__(
        self,
        graph: HeteroData,
        edge_attributes: list[str],
        src_size: int,
        dst_size: int,
        trainable_size: int,
    ) -> None:
        """Initialize StaticGraphProvider.

        Parameters
        ----------
        graph : HeteroData
            Graph containing edges
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

        assert graph, "StaticGraphProvider needs a valid graph to register edges."
        assert edge_attributes is not None, "Edge attributes must be provided"

        edge_attr_tensor = torch.cat([graph[attr] for attr in edge_attributes], axis=1)

        self.register_buffer("edge_attr", edge_attr_tensor, persistent=False)
        self.register_buffer("edge_index_base", graph.edge_index, persistent=False)
        self.register_buffer(
            "edge_inc", torch.from_numpy(np.asarray([[src_size], [dst_size]], dtype=np.int64)), persistent=False
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

    def _get_edges_impl(
        self,
        batch_size: int,
        shard_edges: bool,
        model_comm_group: Optional[ProcessGroup],
    ) -> tuple[Tensor, Adj, Optional[tuple[list, list]]]:
        """Implementation of get_edges."""
        edge_attr = self.trainable(self.edge_attr, batch_size)
        edge_index = self._expand_edges(self.edge_index_base, self.edge_inc, batch_size)

        if shard_edges:
            src_size, dst_size = self.edge_inc[:, 0].tolist()
            return shard_edges_1hop(
                edge_attr, edge_index, src_size * batch_size, dst_size * batch_size, model_comm_group
            )

        return edge_attr, edge_index, None

    def get_edges(
        self,
        batch_size: int,
        src_coords: Optional[Tensor] = None,
        dst_coords: Optional[Tensor] = None,
        model_comm_group: Optional[ProcessGroup] = None,
        shard_edges: bool = True,
        act_checkpoint: bool = True,
    ) -> tuple[Tensor, Adj, Optional[tuple[list, list]]]:
        """Get edge attributes and expanded edge index for static graph.

        Parameters
        ----------
        batch_size : int
            Number of times to expand the edge index
        src_coords : Tensor, optional
            Source node coordinates (ignored for static graphs)
        dst_coords : Tensor, optional
            Destination node coordinates (ignored for static graphs)
        model_comm_group : ProcessGroup, optional
            Model communication group
        shard_edges : bool, optional
            Whether to shard edges, by default True.
        act_checkpoint : bool, optional
            Whether to use gradient checkpointing, by default True.

        Returns
        -------
        tuple[Tensor, Adj, Optional[tuple[list, list]]]
            Edge attributes, expanded edge index, and optional edge_shard_shapes.
            edge_shard_shapes is (shapes_edge_attr, shapes_edge_idx) when shard_edges=True,
            otherwise None.
        """
        if act_checkpoint:
            return checkpoint(self._get_edges_impl, batch_size, shard_edges, model_comm_group, use_reentrant=False)
        return self._get_edges_impl(batch_size, shard_edges, model_comm_group)


class NoOpGraphProvider(BaseGraphProvider):
    """Provider for edge-less architectures (e.g., Transformers).

    Returns None for edges and has edge_dim=0. Used when the mapper/processor
    does not require graph structure (e.g., pure attention-based models).
    """

    def __init__(self) -> None:
        """Initialize NoOpGraphProvider."""
        super().__init__()

    @property
    def edge_dim(self) -> int:
        """Return the edge dimension (0 for no edges)."""
        return 0

    def get_edges(
        self,
        batch_size: Optional[int] = None,
        src_coords: Optional[Tensor] = None,
        dst_coords: Optional[Tensor] = None,
        model_comm_group: Optional[ProcessGroup] = None,
        shard_edges: bool = True,
    ) -> tuple[None, None, None]:
        """Return None for edge attributes, edge index, and edge_shard_shapes.

        Parameters
        ----------
        batch_size : int, optional
            Unused
        src_coords : Tensor, optional
            Unused
        dst_coords : Tensor, optional
            Unused
        model_comm_group : ProcessGroup, optional
            Unused
        shard_edges : bool, optional
            Unused

        Returns
        -------
        tuple[None, None, None]
            No edges
        """
        return None, None, None


class DynamicGraphProvider(BaseGraphProvider):
    """Provider for dynamic graphs where edges are supplied at runtime.

    Does not support trainable edge parameters.

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

    def _get_edges_impl(
        self,
        src_coords: Tensor,
        dst_coords: Tensor,
        shard_edges: bool,
        model_comm_group: Optional[ProcessGroup],
    ) -> tuple[Tensor, Adj, Optional[tuple[list, list]]]:
        """Implementation of get_edges, separated for checkpointing."""
        # Build graph from coordinates
        edge_attr, edge_index = self.build_graph(src_coords, dst_coords)

        if shard_edges:
            return shard_edges_1hop(edge_attr, edge_index, src_coords.shape[0], dst_coords.shape[0], model_comm_group)

        return edge_attr, edge_index, None

    def get_edges(
        self,
        batch_size: Optional[int] = None,
        src_coords: Optional[Tensor] = None,
        dst_coords: Optional[Tensor] = None,
        model_comm_group: Optional[ProcessGroup] = None,
        shard_edges: bool = True,
        act_checkpoint: bool = True,
    ) -> tuple[Tensor, Adj, Optional[tuple[list, list]]]:
        """Get dynamic edges constructed from node coordinates.

        Calls build_graph() to construct edges on-the-fly using k-NN, radius graphs, etc.

        Parameters
        ----------
        batch_size : int, optional
            Batch size (currently unused, reserved for future implementation)
        src_coords : Tensor, optional
            Source node coordinates
        dst_coords : Tensor, optional
            Destination node coordinates
        model_comm_group : ProcessGroup, optional
            Model communication group
        shard_edges : bool, optional
            Whether to shard edges, by default True
        act_checkpoint : bool, optional
            Whether to use gradient checkpointing, by default True.

        Returns
        -------
        tuple[Tensor, Adj, Optional[tuple[list, list]]]
            Edge attributes, edge index, and optional edge_shard_shapes

        Raises
        ------
        ValueError
            If coordinates are not provided
        NotImplementedError
            If build_graph() is not yet implemented
        """
        if src_coords is None or dst_coords is None:
            raise ValueError("DynamicGraphProvider requires (src_coords, dst_coords) to construct edges.")

        if act_checkpoint:
            return checkpoint(
                self._get_edges_impl, src_coords, dst_coords, shard_edges, model_comm_group, use_reentrant=False
            )
        return self._get_edges_impl(src_coords, dst_coords, shard_edges, model_comm_group)


class ProjectionGraphProvider(BaseGraphProvider):
    """Provider for sparse projection matrices.

    Builds and stores sparse projection matrix from graph or file.
    """

    def __init__(
        self,
        graph: Optional[HeteroData] = None,
        edges_name: Optional[tuple[str, str, str]] = None,
        edge_weight_attribute: Optional[str] = None,
        src_node_weight_attribute: Optional[str] = None,
        file_path: Optional[str | Path] = None,
        row_normalize: bool = False,
    ) -> None:
        """Initialize ProjectionGraphProvider.

        Parameters
        ----------
        graph : HeteroData, optional
            Graph containing edges for projection
        edges_name : tuple[str, str, str], optional
            Edge type identifier (src, relation, dst)
        edge_weight_attribute : str, optional
            Edge attribute name for weights
        src_node_weight_attribute : str, optional
            Source node attribute name for weights
        file_path : str | Path, optional
            Path to .npz file with projection matrix
        row_normalize : bool
            Whether to normalize weights per row (target node) so each row sums to 1
        """
        super().__init__()

        if file_path is not None:
            self._build_from_file(file_path, row_normalize)
        else:
            assert (
                graph is not None and edges_name is not None
            ), "Must provide graph and edges_name if file_path not given"
            self._build_from_graph(graph, edges_name, edge_weight_attribute, src_node_weight_attribute, row_normalize)

    def _build_from_file(self, file_path: str | Path, row_normalize: bool) -> None:
        """Load projection matrix from file."""
        from scipy.sparse import load_npz

        truncation_data = load_npz(file_path)
        edge_index = torch.tensor(np.vstack(truncation_data.nonzero()), dtype=torch.long)
        weights = torch.tensor(truncation_data.data, dtype=torch.float32)
        src_size, dst_size = truncation_data.shape

        self._create_matrix(edge_index, weights, src_size, dst_size, row_normalize)

    def _build_from_graph(
        self,
        graph: HeteroData,
        edges_name: tuple[str, str, str],
        edge_weight_attribute: Optional[str],
        src_node_weight_attribute: Optional[str],
        row_normalize: bool,
    ) -> None:
        """Build projection matrix from graph."""
        sub_graph = graph[edges_name]

        if edge_weight_attribute:
            weights = sub_graph[edge_weight_attribute].squeeze()
        else:
            weights = torch.ones(sub_graph.edge_index.shape[1], device=sub_graph.edge_index.device)

        if src_node_weight_attribute:
            weights *= graph[edges_name[0]][src_node_weight_attribute][sub_graph.edge_index[0]]

        # PyG convention: edge_index[0]=source, edge_index[1]=target
        # For M @ x, we need matrix shape (targets, sources) with:
        #   - row indices = targets
        #   - col indices = sources
        # -> swap edge_index to [targets, sources] for COO tensor
        edge_index_for_coo = torch.stack([sub_graph.edge_index[1], sub_graph.edge_index[0]])

        self._create_matrix(
            edge_index_for_coo,
            weights,
            graph[edges_name[2]].num_nodes,  # dst_size (targets) = rows
            graph[edges_name[0]].num_nodes,  # src_size (sources) = cols
            row_normalize,
        )

    def _create_matrix(
        self,
        edge_index: Tensor,
        weights: Tensor,
        src_size: int,
        dst_size: int,
        row_normalize: bool,
    ) -> None:
        """Create sparse projection matrix."""

        if row_normalize:
            weights = self._row_normalize_weights(edge_index, weights, src_size)

        self.projection_matrix = torch.sparse_coo_tensor(
            edge_index,
            weights,
            (src_size, dst_size),
            device=edge_index.device,
        ).coalesce()

        self._edge_dim = self.projection_matrix.shape[1]

        row_sums = torch.zeros(src_size, device=weights.device).scatter_add_(0, edge_index[0], weights)
        if not torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5):
            LOGGER.warning(
                "Projection matrix rows do not sum to 1 (min=%.4f, max=%.4f, mean=%.4f). "
                "Consider using row_normalize=True or pre-normalized weights.",
                row_sums.min().item(),
                row_sums.max().item(),
                row_sums.mean().item(),
            )

    @staticmethod
    def _row_normalize_weights(edge_index: Tensor, weights: Tensor, num_rows: int) -> Tensor:
        """Normalize weights per row (target node) so each row sums to 1."""
        total = torch.zeros(num_rows, device=weights.device)
        # edge_index[0] contains row indices (targets) for COO tensor format
        norm = total.scatter_add_(0, edge_index[0].long(), weights)
        norm = norm[edge_index[0]]
        return weights / (norm + 1e-8)

    @property
    def edge_dim(self) -> int:
        """Return projection matrix shape."""
        return self._edge_dim

    @property
    def is_sparse(self) -> bool:
        """This provider returns sparse matrices."""
        return True

    def get_edges(
        self,
        batch_size: Optional[int] = None,
        src_coords: Optional[Tensor] = None,
        dst_coords: Optional[Tensor] = None,
        model_comm_group: Optional[ProcessGroup] = None,
        shard_edges: bool = True,
        device: Optional[torch.device] = None,
    ) -> Tensor:
        """Return the sparse projection matrix.

        Parameters
        ----------
        batch_size : int, optional
            Unused for sparse providers
        src_coords : Tensor, optional
            Unused for sparse providers
        dst_coords : Tensor, optional
            Unused for sparse providers
        model_comm_group : ProcessGroup, optional
            Unused for sparse providers
        shard_edges : bool, optional
            Unused for sparse providers
        device : torch.device, optional
            Target device for matrix

        Returns
        -------
        Tensor
            Sparse projection matrix
        """
        if device is not None:
            # sparse tensors can't be registered as buffers with ddp, so move on demand
            self.projection_matrix = self.projection_matrix.to(device)
        return self.projection_matrix
