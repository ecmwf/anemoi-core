# (C) Copyright 2024- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


"""Graph-based spatial smoothness loss for unstructured grids.

Works with any grid topology (TCo, O-grid, HEALPix, icosahedral, etc.)
by using the graph connectivity to define spatial neighbors.

Usage
-----
Add data-to-data edges to your graph configuration:

.. code-block:: yaml

    edges:
      # ... existing encoder/processor/decoder edges ...

      # Add data-to-data edges for spatial smoothness
      - source_name: ${graph.data}
        target_name: ${graph.data}
        edge_builders:
          - _target_: anemoi.graphs.edges.KNNEdges
            num_nearest_neighbours: 6
        attributes: ${graph.attributes.edges}

Then configure the loss, typically inside a CombinedLoss:

.. code-block:: yaml

    training_loss:
      datasets:
        data:
          _target_: anemoi.training.losses.CombinedLoss
          losses:
            - _target_: anemoi.training.losses.MSELoss
              scalers: ['general_variable', 'node_weights']
              ignore_nans: true
            - _target_: anemoi.training.losses.GraphLaplacianSmoothnessLoss
              penalty_weight: 1.0
              scalers: ['node_weights']
              predicted_variables: [z_500, t_850]
          loss_weights: [1.0, 0.05]

The edges are taken from the training graph (``graph_data`` is injected by the
loss factory because this class declares ``needs_graph_data``); ``graph_path``
or ``edge_index``/``edge_index_path`` can be used as standalone alternatives.
Restrict the penalty to a subset of variables with ``predicted_variables``
(handled by the loss variable mapper).
"""

import logging
from pathlib import Path
from typing import Union

import torch
from torch.distributed.distributed_c10d import ProcessGroup
from torch_geometric.data import HeteroData

from anemoi.training.losses.base import BaseLoss
from anemoi.training.utils.enums import TensorDim

LOGGER = logging.getLogger(__name__)


class GraphLaplacianSmoothnessLoss(BaseLoss):
    """Graph Laplacian smoothness loss for unstructured grids.

    Penalizes the graph Laplacian of the predictions:
    ``L[i] = sum_j A[i,j] * (f[i] - f[j])``

    where A is the adjacency matrix. This is equivalent to penalizing
    the difference between each node and the average of its neighbors.

    Tensor convention: (batch, time, ensemble, grid, variable) — 5D.
    """

    needs_graph_data: bool = True

    def __init__(
        self,
        graph_data: HeteroData | None = None,
        data_node_name: str | None = None,
        graph_path: Union[str, Path] | None = None,
        src_nodes_name: str | None = None,
        dst_nodes_name: str | None = None,
        edge_index: torch.Tensor | None = None,
        edge_index_path: Union[str, Path] | None = None,
        penalty_weight: float = 1.0,
        ignore_nans: bool = True,
        **kwargs,  # noqa: ARG002
    ) -> None:
        """Initialize the graph Laplacian smoothness loss.

        Parameters
        ----------
        graph_data : HeteroData, optional
            Graph data object containing edges. Injected by the loss factory
            (``needs_graph_data = True``).
        data_node_name : str, optional
            Dataset node name, injected by the loss factory. Used as the
            default for ``src_nodes_name``/``dst_nodes_name``.
        graph_path : str or Path, optional
            Path to a saved graph .pt file, as an alternative to ``graph_data``.
        src_nodes_name : str, optional
            Source node name in graph, defaults to ``data_node_name`` or "data".
        dst_nodes_name : str, optional
            Destination node name in graph, defaults to ``data_node_name`` or "data".
        edge_index : torch.Tensor, optional
            Edge connectivity tensor of shape (2, num_edges)
        edge_index_path : str or Path, optional
            Path to file containing edge_index
        penalty_weight : float, optional
            Weight for the penalty, by default 1.0
        ignore_nans : bool, optional
            Whether to handle NaN locations, by default True
        **kwargs : dict
            Additional keyword arguments (ignored)
        """
        super().__init__(ignore_nans=ignore_nans)
        self.penalty_weight = penalty_weight
        # The Laplacian gathers/scatters with full-grid edge indices, which is
        # invalid on a grid shard; force the gather-before-loss path.
        self.supports_sharding = False

        src_nodes_name = src_nodes_name or data_node_name or "data"
        dst_nodes_name = dst_nodes_name or data_node_name or "data"

        resolved_edge_index = self._resolve_edge_index(
            graph_data=graph_data,
            graph_path=graph_path,
            src_nodes_name=src_nodes_name,
            dst_nodes_name=dst_nodes_name,
            edge_index=edge_index,
            edge_index_path=edge_index_path,
        )

        if resolved_edge_index is not None:
            self.register_buffer("edge_index", resolved_edge_index.long())
        else:
            LOGGER.warning(
                "No edge_index provided. GraphLaplacianSmoothnessLoss will return zero loss. "
                "To fix this, add %s-to-%s edges to your graph config or provide "
                "graph_path, edge_index, or edge_index_path.",
                src_nodes_name,
                dst_nodes_name,
            )
            self.register_buffer("edge_index", torch.empty(2, 0, dtype=torch.long))

        # Degree computed lazily; non-persistent to avoid checkpoint size mismatch
        self.register_buffer("degree", torch.empty(0), persistent=False)

    @staticmethod
    def _resolve_edge_index(
        *,
        graph_data: HeteroData | None,
        graph_path: Union[str, Path] | None,
        src_nodes_name: str,
        dst_nodes_name: str,
        edge_index: torch.Tensor | None,
        edge_index_path: Union[str, Path] | None,
    ) -> torch.Tensor | None:
        """Resolve edge_index from the various sources (in priority order)."""
        # Priority 1: Direct edge_index tensor
        if edge_index is not None:
            LOGGER.info("Using directly provided edge_index")
            return edge_index

        # Priority 2: From graph_data object
        if graph_data is not None:
            return _extract_edge_index_from_graph(graph_data, src_nodes_name, dst_nodes_name)

        # Priority 3: From graph file path
        if graph_path is not None:
            graph_data = _load_graph_from_file(graph_path)
            return _extract_edge_index_from_graph(graph_data, src_nodes_name, dst_nodes_name)

        # Priority 4: From edge_index file
        if edge_index_path is not None:
            return _load_edge_index_from_file(edge_index_path)

        return None

    def _compute_degree(self, num_nodes: int) -> torch.Tensor:
        """Compute node degrees lazily and cache them."""
        if self.degree.numel() > 0:
            return self.degree

        degree = torch.zeros(num_nodes, device=self.edge_index.device)
        src_nodes = self.edge_index[0]
        degree.scatter_add_(0, src_nodes, torch.ones_like(src_nodes, dtype=torch.float))
        self.degree = degree.clamp(min=1.0)
        return self.degree

    def _load_from_state_dict(
        self,
        state_dict: dict,
        prefix: str,
        local_metadata: dict,
        strict: bool,
        missing_keys: list[str],
        unexpected_keys: list[str],
        error_msgs: list[str],
    ) -> None:
        """Override to ignore the lazily-computed 'degree' buffer from old checkpoints."""
        degree_key = prefix + "degree"
        state_dict.pop(degree_key, None)
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    @property
    def name(self) -> str:
        return "graph_laplacian_smoothness"

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,  # noqa: ARG002
        squash: bool = True,
        *,
        scaler_indices: tuple[int, ...] | None = None,
        without_scalers: list[str] | list[int] | None = None,
        grid_shard_slice: slice | None = None,
        group: ProcessGroup | None = None,
        **kwargs,  # noqa: ARG002
    ) -> torch.Tensor:
        """Compute the graph Laplacian smoothness penalty.

        Parameters
        ----------
        pred : torch.Tensor
            Prediction tensor, shape (bs, output_times, ensemble, num_nodes, n_outputs)
        target : torch.Tensor
            Target tensor (unused: this penalty operates on predictions only)
        squash : bool, optional
            Average over variable dimension, by default True
        scaler_indices : tuple[int, ...] | None
            Scaler indices for loss scaling.
        without_scalers : list[str] | list[int] | None
            Scalers to exclude.
        grid_shard_slice : slice | None
            Grid shard slice for distributed operation.
        group : ProcessGroup | None
            Process group for distributed reduction.
        **kwargs
            Additional keyword arguments (unused).

        Returns
        -------
        torch.Tensor
            Spatial smoothness penalty loss
        """
        if self.edge_index.numel() == 0:
            return torch.tensor(0.0, device=pred.device, requires_grad=True)

        is_sharded = grid_shard_slice is not None
        bs, time, ens, num_nodes, nvars = pred.shape

        src_idx = self.edge_index[0]
        tgt_idx = self.edge_index[1]

        # Compute neighbor averages via message passing
        neighbor_sum = torch.zeros(bs, time, ens, num_nodes, nvars, device=pred.device, dtype=pred.dtype)
        pred_tgt = pred[:, :, :, tgt_idx, :]
        src_idx_expanded = src_idx.view(1, 1, 1, -1, 1).expand(bs, time, ens, -1, nvars)
        neighbor_sum.scatter_add_(dim=TensorDim.GRID, index=src_idx_expanded, src=pred_tgt)

        degree = self._compute_degree(num_nodes)
        neighbor_avg = neighbor_sum / degree.view(1, 1, 1, -1, 1)

        # Laplacian: difference from neighbor average
        laplacian_sq = (pred - neighbor_avg) ** 2

        # Note: we intentionally do NOT mask based on target NaN locations.
        # This loss operates purely on predictions and should encourage
        # spatial coherence everywhere, especially at unobserved grid points.

        out = self.penalty_weight * laplacian_sq

        out = self.scale(
            out,
            scaler_indices,
            without_scalers=without_scalers,
            grid_shard_slice=grid_shard_slice,
        )

        return self.reduce(out, squash, group=group if is_sharded else None)


def _load_graph_from_file(graph_path: Union[str, Path]) -> HeteroData:
    """Load a graph HeteroData object from a .pt file."""
    graph_path = Path(graph_path)
    if not graph_path.exists():
        msg = f"Graph file not found: {graph_path}"
        raise FileNotFoundError(msg)
    LOGGER.info("Loading graph data from %s", graph_path)
    return torch.load(graph_path, map_location="cpu", weights_only=False)


def _extract_edge_index_from_graph(
    graph_data: HeteroData,
    src_nodes_name: str,
    dst_nodes_name: str,
) -> torch.Tensor | None:
    """Extract edge_index from a HeteroData graph object."""
    edge_key = (src_nodes_name, "to", dst_nodes_name)
    if edge_key in graph_data.edge_types:
        ei = graph_data[edge_key].edge_index
        LOGGER.info("Extracted edge_index from graph_data[%s] with %d edges", edge_key, ei.shape[1])
        return ei

    available_edges = list(graph_data.edge_types)
    LOGGER.warning(
        "Edge type %s not found in graph_data. Available: %s. "
        "Consider adding data-to-data edges in your graph config.",
        edge_key,
        available_edges,
    )
    return None


def _load_edge_index_from_file(path: Union[str, Path]) -> torch.Tensor:
    """Load edge_index from a .pt, .npy or .npz file."""
    path = Path(path)
    if not path.exists():
        msg = f"Edge index file not found: {path}"
        raise FileNotFoundError(msg)

    if path.suffix == ".pt":
        return torch.load(path, map_location="cpu", weights_only=True)
    if path.suffix == ".npy":
        import numpy as np

        return torch.from_numpy(np.load(path))
    if path.suffix == ".npz":
        import numpy as np

        data = np.load(path)
        if "edge_index" in data:
            return torch.from_numpy(data["edge_index"])
        if "row" in data and "col" in data:
            return torch.stack([torch.from_numpy(data["row"]), torch.from_numpy(data["col"])])
        msg = f"Cannot find edge_index in {path}"
        raise ValueError(msg)
    msg = f"Unsupported file format: {path.suffix}"
    raise ValueError(msg)
