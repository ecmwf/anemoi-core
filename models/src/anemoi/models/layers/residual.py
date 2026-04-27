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

from anemoi.graphs.projection_helpers import DEFAULT_EDGE_WEIGHT_ATTRIBUTE
from anemoi.models.distributed.graph import gather_channels
from anemoi.models.distributed.graph import shard_channels
from anemoi.models.distributed.shapes import apply_shard_shapes
from anemoi.models.layers.graph_provider import ProjectionGraphProvider
from anemoi.models.layers.sparse_projector import SparseProjector


class BaseResidualConnection(nn.Module, ABC):
    """Base class for residual connection modules."""

    def __init__(self, graph: HeteroData | None = None) -> None:
        super().__init__()

    @abstractmethod
    def forward(
        self,
        x: torch.Tensor,
        grid_shard_shapes=None,
        model_comm_group=None,
        n_step_output: int | None = None,
    ) -> torch.Tensor:
        """Define the residual connection operation.

        Should be overridden by subclasses.
        """
        pass

    @staticmethod
    def _expand_time(x: torch.Tensor, n_step_output: int | None) -> torch.Tensor:
        if n_step_output is None:
            return x
        return x.unsqueeze(1).expand(-1, n_step_output, -1, -1, -1)


class SkipConnection(BaseResidualConnection):
    """Skip connection module

    This layer returns the most recent timestep from the input sequence.

    This module is used to bypass processing layers and directly pass the latest input forward.
    """

    def __init__(self, step: int = -1, **_) -> None:
        super().__init__()
        self.step = step

    def forward(
        self,
        x: torch.Tensor,
        grid_shard_shapes=None,
        model_comm_group=None,
        n_step_output: int | None = None,
    ) -> torch.Tensor:
        """Return the last timestep of the input sequence."""
        x_skip = x[:, self.step, ...]  # x shape: (batch, time, ens, nodes, features)
        return self._expand_time(x_skip, n_step_output)


class TruncatedConnection(BaseResidualConnection):
    """Truncated skip connection.

    Applies a coarse-graining and reconstruction of input features using sparse
    projections to truncate high-frequency features.

    Edge names and the edge-weight attribute are expected to be pre-resolved by
    ``ProjectionCreator`` and passed in directly.  File-path loading is still
    supported as an alternative to the graph-based path.

    Parameters
    ----------
    graph : HeteroData, optional
        Graph containing the truncation subgraphs.
    edge_weight_attribute : str, optional
        Edge attribute used as projection weights (default: ``gauss_weight``).
    src_node_weight_attribute : str, optional
        Source-node attribute used as additional projection weights.
    autocast : bool, default False
        Whether to use automatic mixed precision for the projections.
    truncation_up_file_path : str, optional
        Path to an ``.npz`` file for the up-projection matrix.
    truncation_down_file_path : str, optional
        Path to an ``.npz`` file for the down-projection matrix.
    truncation_up_edges_name : tuple[str, str, str], optional
        Pre-resolved ``(src, relation, dst)`` edge type for the up-projection.
    truncation_down_edges_name : tuple[str, str, str], optional
        Pre-resolved ``(src, relation, dst)`` edge type for the down-projection.
    row_normalize : bool, optional
        Normalize projection weights per target node so each row sums to 1.

    Examples
    --------
    >>> # Graph-based path (edge names supplied by ProjectionCreator)
    >>> conn = TruncatedConnection(
    ...     graph=graph,
    ...     data_nodes="data",
    ...     truncation_down_edges_name=("data", "to", "truncation"),
    ...     truncation_up_edges_name=("truncation", "to", "data"),
    ...     edge_weight_attribute="gauss_weight",
    ... )
    >>> x = torch.randn(2, 4, 1, 40192, 44)  # (batch, time, ens, nodes, features)
    >>> out = conn(x)
    >>> print(out.shape)
    torch.Size([2, 4, 1, 40192, 44])

    >>> # File-based path
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
        src_node_weight_attribute: Optional[str] = None,
        edge_weight_attribute: Optional[str] = None,
        truncation_config: Optional[dict] = None,
        truncation_up_edges_name: Optional[tuple[str, str, str]] = None,
        truncation_down_edges_name: Optional[tuple[str, str, str]] = None,
        data_node_name: str = "data",
        autocast: bool = False,
        row_normalize: bool = False,
        # Deprecated: pass inside truncation_config instead.
        truncation_up_file_path: Optional[str] = None,
        truncation_down_file_path: Optional[str] = None,
        **_,
    ) -> None:
        super().__init__()

        truncation_config = self._normalise_truncation_config(
            truncation_config,
            truncation_up_file_path,
            truncation_down_file_path,
        )

        if truncation_config is not None:
            up_path = truncation_config.get("truncation_up_file_path")
            down_path = truncation_config.get("truncation_down_file_path")
            if up_path is not None and down_path is not None:
                truncation_up_file_path = up_path
                truncation_down_file_path = down_path
            else:
                from anemoi.graphs.builders import build_truncation_subgraph

                graph = build_truncation_subgraph(graph, data_node_name, truncation_config)
                truncation_down_edges_name = (data_node_name, "to", "truncation")
                truncation_up_edges_name = ("truncation", "to", data_node_name)

        _edge_weight_attr = (
            edge_weight_attribute if edge_weight_attribute is not None else DEFAULT_EDGE_WEIGHT_ATTRIBUTE
        )

        up_edges, down_edges = self._resolve_edges(
            graph=graph,
            truncation_up_file_path=truncation_up_file_path,
            truncation_down_file_path=truncation_down_file_path,
            truncation_up_edges_name=truncation_up_edges_name,
            truncation_down_edges_name=truncation_down_edges_name,
        )

        self.provider_down = ProjectionGraphProvider(
            graph=graph,
            edges_name=down_edges,
            edge_weight_attribute=_edge_weight_attr,
            src_node_weight_attribute=src_node_weight_attribute,
            file_path=truncation_down_file_path,
            row_normalize=row_normalize,
        )

        self.provider_up = ProjectionGraphProvider(
            graph=graph,
            edges_name=up_edges,
            edge_weight_attribute=_edge_weight_attr,
            src_node_weight_attribute=src_node_weight_attribute,
            file_path=truncation_up_file_path,
            row_normalize=row_normalize,
        )

        self.projector = SparseProjector(autocast=autocast)

    @staticmethod
    def _normalise_truncation_config(
        truncation_config: Optional[dict],
        truncation_up_file_path: Optional[str],
        truncation_down_file_path: Optional[str],
    ) -> Optional[dict]:
        """Forward deprecated top-level file-path kwargs into truncation_config."""
        has_files = truncation_up_file_path is not None or truncation_down_file_path is not None
        if not has_files:
            return truncation_config
        import logging

        logging.getLogger(__name__).warning(
            "Passing 'truncation_up_file_path' / 'truncation_down_file_path' as top-level kwargs "
            "is deprecated. Move them inside 'truncation_config' instead."
        )
        cfg = dict(truncation_config) if truncation_config is not None else {}
        if truncation_up_file_path is not None:
            cfg.setdefault("truncation_up_file_path", truncation_up_file_path)
        if truncation_down_file_path is not None:
            cfg.setdefault("truncation_down_file_path", truncation_down_file_path)
        return cfg

    @staticmethod
    def _resolve_edges(
        *,
        graph: HeteroData | None,
        truncation_up_file_path: str | None,
        truncation_down_file_path: str | None,
        truncation_up_edges_name: tuple[str, str, str] | None,
        truncation_down_edges_name: tuple[str, str, str] | None,
    ) -> tuple[tuple[str, str, str] | None, tuple[str, str, str] | None]:
        """Validate and return the (up, down) edge tuples."""
        files_specified = truncation_up_file_path is not None and truncation_down_file_path is not None
        if files_specified:
            assert (
                truncation_up_edges_name is None and truncation_down_edges_name is None
            ), "Specify either file paths or edge names for truncation, not both."
            return None, None

        assert graph is not None, "graph must be provided when file paths are not specified."
        assert (
            truncation_up_edges_name is not None and truncation_down_edges_name is not None
        ), "Both truncation_up_edges_name and truncation_down_edges_name must be provided."
        up_edges = tuple(truncation_up_edges_name)
        down_edges = tuple(truncation_down_edges_name)
        assert up_edges in graph.edge_types, f"Graph must contain edges {up_edges} for up-projection."
        assert down_edges in graph.edge_types, f"Graph must contain edges {down_edges} for down-projection."
        return up_edges, down_edges

    def forward(
        self,
        x: torch.Tensor,
        grid_shard_shapes=None,
        model_comm_group=None,
        n_step_output: int | None = None,
    ) -> torch.Tensor:
        """Apply truncated skip connection."""
        batch_size = x.shape[0]
        x = x[:, -1, ...]  # pick latest step
        shard_shapes = apply_shard_shapes(x, 0, grid_shard_shapes) if grid_shard_shapes is not None else None

        x = einops.rearrange(x, "batch ensemble grid features -> (batch ensemble) grid features")
        x = self._to_channel_shards(x, shard_shapes, model_comm_group)
        x = self.projector(x, self.provider_down.get_edges(device=x.device))
        x = self.projector(x, self.provider_up.get_edges(device=x.device))
        x = self._to_grid_shards(x, shard_shapes, model_comm_group)
        x = einops.rearrange(
            x,
            "(batch ensemble) grid features -> batch ensemble grid features",
            batch=batch_size,
        )

        return self._expand_time(x, n_step_output)

    def _to_channel_shards(self, x, shard_shapes=None, model_comm_group=None):
        """Move node-major tensors into the channel-sharded layout used by projection kernels."""
        return self._reshard(x, shard_channels, shard_shapes, model_comm_group)

    def _to_grid_shards(self, x, shard_shapes=None, model_comm_group=None):
        """Restore projected tensors back to the original grid-sharded layout."""
        return self._reshard(x, gather_channels, shard_shapes, model_comm_group)

    def _reshard(self, x, fn, shard_shapes=None, model_comm_group=None):
        """Apply a sharding transform only when shard metadata is available."""
        if shard_shapes is not None:
            x = fn(x, shard_shapes, model_comm_group)
        return x
