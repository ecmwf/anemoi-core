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
from pathlib import Path
from typing import Any
from typing import Optional

import einops
import torch
from omegaconf import DictConfig
from torch import nn
from torch_geometric.data import HeteroData

from anemoi.graphs.create import GraphCreator
from anemoi.models.distributed.graph import gather_channels
from anemoi.models.distributed.graph import shard_channels
from anemoi.models.distributed.shapes import apply_shard_shapes
from anemoi.models.layers.graph_provider import ProjectionGraphProvider
from anemoi.models.layers.sparse_projector import SparseProjector


class BaseResidualConnection(nn.Module, ABC):
    """Base class for residual connection modules."""

    def __init__(self) -> None:
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

    def __init__(self, step: int = -1) -> None:
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
    truncation_up_file_path : str | Path, optional
        File path (.npz) to load the up-projection matrix from.
    truncation_down_file_path : str | Path, optional
        File path (.npz) to load the down-projection matrix from.
    truncation_matrices_path : str | Path, optional
        Optional base path for resolving truncation matrix file paths.
    truncation_graph : dict, optional
        Graph-based truncation specification (graph_config + edge definitions).
    autocast : bool, default False
        Whether to use automatic mixed precision for the projections.
    row_normalize : bool, optional
        Whether to normalize weights per row (target node) so each row sums to 1

    Example
    -------
    >>> import torch
    >>> # Example using a truncation graph definition
    >>> truncation_graph = {
    ...     "graph_config": {
    ...         "nodes": {"data": {...}, "trunc": {...}},
    ...         "edges": [...],
    ...         "post_processors": [],
    ...     },
    ...     "down_edges_name": ["data", "to", "trunc"],
    ...     "up_edges_name": ["trunc", "to", "data"],
    ...     "edge_weight_attribute": "gauss_weight",
    ... }
    >>> conn = TruncatedConnection(truncation_graph=truncation_graph)
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
        truncation_up_file_path: Optional[str | Path] = None,
        truncation_down_file_path: Optional[str | Path] = None,
        autocast: bool = False,
        truncation_matrices_path: str | Path | None = None,
        truncation_graph: dict[str, Any] | DictConfig | None = None,
        row_normalize: bool = False,
    ) -> None:
        super().__init__()

        graph = None
        if truncation_graph is not None:
            assert (
                truncation_up_file_path is None and truncation_down_file_path is None
            ), "truncation_up_file_path and truncation_down_file_path must be omitted when truncation_graph is set."
            (
                graph,
                up_edges,
                down_edges,
                edge_weight_attribute,
                src_node_weight_attribute,
            ) = self._load_truncation_graph(truncation_graph)
            truncation_up_file_path = None
            truncation_down_file_path = None
        else:
            assert (
                truncation_up_file_path is not None and truncation_down_file_path is not None
            ), "truncation_up_file_path and truncation_down_file_path must be provided when truncation_graph is not set."
            truncation_up_file_path = self._resolve_matrix_path(truncation_up_file_path, truncation_matrices_path)
            truncation_down_file_path = self._resolve_matrix_path(truncation_down_file_path, truncation_matrices_path)
            up_edges = down_edges = None
            edge_weight_attribute = None
            src_node_weight_attribute = None

        self.provider_down = ProjectionGraphProvider(
            graph=graph,
            edges_name=down_edges,
            edge_weight_attribute=edge_weight_attribute,
            src_node_weight_attribute=src_node_weight_attribute,
            file_path=truncation_down_file_path,
            row_normalize=row_normalize,
        )

        self.provider_up = ProjectionGraphProvider(
            graph=graph,
            edges_name=up_edges,
            edge_weight_attribute=edge_weight_attribute,
            src_node_weight_attribute=src_node_weight_attribute,
            file_path=truncation_up_file_path,
            row_normalize=row_normalize,
        )

        self.projector = SparseProjector(autocast=autocast)

    @staticmethod
    def _resolve_matrix_path(file_path: Optional[str | Path], base_path: str | Path | None) -> Optional[Path]:
        if file_path is None:
            return None
        resolved = Path(file_path)
        if base_path and not resolved.is_absolute():
            resolved = Path(base_path) / resolved
        return resolved

    def _build_truncation_graph(self, graph_config: dict[str, Any] | DictConfig) -> HeteroData:
        graph_creator = GraphCreator(config=graph_config)
        graph = HeteroData()
        graph = graph_creator.update_graph(graph)
        graph = graph_creator.clean(graph)
        return graph_creator.post_process(graph)

    def _load_truncation_graph(
        self,
        truncation_graph: dict[str, Any] | DictConfig,
    ) -> tuple[HeteroData, tuple[str, str, str], tuple[str, str, str], Optional[str], Optional[str]]:
        assert isinstance(truncation_graph, (dict, DictConfig)), "truncation_graph must be a mapping"
        graph_config = truncation_graph.get("graph_config")
        assert isinstance(graph_config, (dict, DictConfig)), "truncation_graph.graph_config must be a mapping"
        down_edges_name = truncation_graph.get("down_edges_name")
        up_edges_name = truncation_graph.get("up_edges_name")
        assert (
            down_edges_name is not None and up_edges_name is not None
        ), "truncation_graph must define down_edges_name and up_edges_name"
        assert len(down_edges_name) == 3, "down_edges_name must be [src, relation, dst]"
        assert len(up_edges_name) == 3, "up_edges_name must be [src, relation, dst]"

        graph = self._build_truncation_graph(graph_config)
        down_edges = tuple(down_edges_name)
        up_edges = tuple(up_edges_name)
        assert down_edges in graph.edge_types, f"Graph must contain edges {down_edges} for down-projection."
        assert up_edges in graph.edge_types, f"Graph must contain edges {up_edges} for up-projection."
        edge_weight_attribute = truncation_graph.get("edge_weight_attribute")
        src_node_weight_attribute = truncation_graph.get("src_node_weight_attribute")
        return graph, up_edges, down_edges, edge_weight_attribute, src_node_weight_attribute

    def forward(self, x: torch.Tensor, grid_shard_shapes=None, model_comm_group=None) -> torch.Tensor:
        """Apply truncated skip connection."""
        batch_size = x.shape[0]
        x = x[:, -1, ...]  # pick latest step
        shard_shapes = apply_shard_shapes(x, 0, grid_shard_shapes) if grid_shard_shapes is not None else None

        x = einops.rearrange(x, "batch ensemble grid features -> (batch ensemble) grid features")
        x = self._to_channel_shards(x, shard_shapes, model_comm_group)
        x = self.projector(x, self.provider_down.get_edges(device=x.device))
        x = self.projector(x, self.provider_up.get_edges(device=x.device))
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
