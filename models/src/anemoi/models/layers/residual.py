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

import einops
import torch
from torch import nn
from torch_geometric.data import HeteroData

from anemoi.models.distributed.graph import gather_channels
from anemoi.models.distributed.graph import shard_channels
from anemoi.models.distributed.shapes import apply_shard_shapes
from anemoi.models.layers.sparse_projector import build_sparse_projector


class BaseResidualConnection(nn.Module, ABC):
    """Base class for residual connection modules."""

    def __init__(self, graph: HeteroData | None = None) -> None:
        super().__init__()

    @abstractmethod
    def forward(self, x: torch.Tensor, grid_shard_shapes=None, model_comm_group=None) -> torch.Tensor:
        """Define the residual connection operation.

        Should be overridden by subclasses.
        """
        pass


class SkipConnection(BaseResidualConnection):
    """Skip connection module

    This layer returns the most recent timestep from the input sequence.

    This module is used to bypass processing layers and directly pass the latest input forward.
    """

    def __init__(self, step: int = -1, **_) -> None:
        super().__init__()
        self.step = step

    def forward(self, x: torch.Tensor, grid_shard_shapes=None, model_comm_group=None) -> torch.Tensor:
        """Return the last timestep of the input sequence."""
        return x[:, self.step, ...]  # x shape: (batch, time, ens, nodes, features)


class TruncatedConnection(BaseResidualConnection):
    """Truncated skip connection

    This connection applies a coarse-graining and reconstruction of input features using sparse
    projections to truncate high frequency features.

    This module uses two projection operators: one to map features from the full-resolution
    grid to a truncated (coarse) grid, and another to project back to the original resolution.

    Parameters
    ----------
    graph : HeteroData, optional
        The graph containing the subgraphs for down and up projections.
    data_nodes : str, optional
        Name of the nodes representing the data nodes.
    truncation_nodes : str, optional
        Name of the nodes representing the truncated (coarse) nodes.
    edge_weight_attribute : str, optional
        Name of the edge attribute to use as weights for the projections.
    src_node_weight_attribute : str, optional
        Name of the source node attribute to use as weights for the projections.
    autocast : bool, default False
        Whether to use automatic mixed precision for the projections.
    truncation_up_file_path : str, optional
        File path (.npz) to load the up-projection matrix from.
    truncation_down_file_path : str, optional
        File path (.npz) to load the down-projection matrix from.

    Example
    -------
    >>> from torch_geometric.data import HeteroData
    >>> import torch
    >>> # Assume graph is a HeteroData object with the required edges and node types
    >>> graph = HeteroData()
    >>> # ...populate graph with nodes and edges for 'data' and 'int'...
    >>> # Example creating the projection matrices from the graph
    >>> conn = TruncatedConnection(
    ...     graph=graph,
    ...     data_nodes="data",
    ...     truncation_nodes="int",
    ...     edge_weight_attribute="gauss_weight",
    ... )
    >>> x = torch.randn(2, 4, 1, 40192, 44)  # (batch, time, ens, nodes, features)
    >>> out = conn(x)
    >>> print(out.shape)
    torch.Size([2, 4, 1, 40192, 44])

    >>> # Example specifying .npz files for projection matrices
    >>> conn = TruncatedConnection(
    ...     truncation_down_file_path="n320_to_o96.npz",
    ...     truncation_up_file_path="o96_to_n320.npz",
    ... )
    >>> x = torch.randn(2, 4, 1, 40192, 44)
    >>> out = conn(x)
    >>> print(out.shape)
    torch.Size([2, 4, 1, 40192, 44])
    """

    def __init__(
        self,
        graph: Optional[HeteroData] = None,
        data_nodes: Optional[str] = None,
        truncation_nodes: Optional[str] = None,
        edge_weight_attribute: Optional[str] = None,
        src_node_weight_attribute: Optional[str] = None,
        truncation_up_file_path: Optional[str] = None,
        truncation_down_file_path: Optional[str] = None,
        autocast: bool = False,
    ) -> None:
        super().__init__()
        up_edges, down_edges = self._get_edges_name(
            graph,
            data_nodes,
            truncation_nodes,
            truncation_up_file_path,
            truncation_down_file_path,
            edge_weight_attribute,
        )

        self.project_down = build_sparse_projector(
            graph=graph,
            edges_name=down_edges,
            edge_weight_attribute=edge_weight_attribute,
            src_node_weight_attribute=src_node_weight_attribute,
            file_path=truncation_down_file_path,
            autocast=autocast,
        )

        self.project_up = build_sparse_projector(
            graph=graph,
            edges_name=up_edges,
            edge_weight_attribute=edge_weight_attribute,
            src_node_weight_attribute=src_node_weight_attribute,
            file_path=truncation_up_file_path,
            autocast=autocast,
        )

    def _get_edges_name(
        self,
        graph,
        data_nodes,
        truncation_nodes,
        truncation_up_file_path,
        truncation_down_file_path,
        edge_weight_attribute,
    ):
        are_files_specified = truncation_up_file_path is not None and truncation_down_file_path is not None
        if not are_files_specified:
            assert graph is not None, "graph must be provided if file paths are not specified."
            assert data_nodes is not None, "data nodes name must be provided if file paths are not specified."
            assert (
                truncation_nodes is not None
            ), "truncation nodes name must be provided if file paths are not specified."
            up_edges = (truncation_nodes, "to", data_nodes)
            down_edges = (data_nodes, "to", truncation_nodes)
            assert up_edges in graph.edge_types, f"Graph must contain edges {up_edges} for up-projection."
            assert down_edges in graph.edge_types, f"Graph must contain edges {down_edges} for down-projection."
        else:
            assert (
                data_nodes is None or truncation_nodes is None or edge_weight_attribute is None
            ), "If file paths are specified, node and attribute names should not be provided."
            up_edges = down_edges = None  # Not used when loading from files
        return up_edges, down_edges

    def forward(self, x: torch.Tensor, grid_shard_shapes=None, model_comm_group=None) -> torch.Tensor:
        """Apply truncated skip connection."""
        batch_size = x.shape[0]
        x = x[:, -1, ...]  # pick latest step
        shard_shapes = apply_shard_shapes(x, 0, grid_shard_shapes) if grid_shard_shapes is not None else None

        x = einops.rearrange(x, "batch ensemble grid features -> (batch ensemble) grid features")
        x = self._to_channel_shards(x, shard_shapes, model_comm_group)
        x = self.project_down(x)
        x = self.project_up(x)
        x = self._to_grid_shards(x, shard_shapes, model_comm_group)
        x = einops.rearrange(x, "(batch ensemble) grid features -> batch ensemble grid features", batch=batch_size)

        return x

    def _to_channel_shards(self, x, shard_shapes=None, model_comm_group=None):
        return self._reshard(x, shard_channels, shard_shapes, model_comm_group)

    def _to_grid_shards(self, x, shard_shapes=None, model_comm_group=None):
        return self._reshard(x, gather_channels, shard_shapes, model_comm_group)

    def _reshard(self, x, fn, shard_shapes=None, model_comm_group=None):
        if shard_shapes is not None:
            x = fn(x, shard_shapes, model_comm_group)
        return x
