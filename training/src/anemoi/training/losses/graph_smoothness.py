# (C) Copyright 2024- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Graph Laplacian smoothness penalty on gridded predictions (ported from feat/testeos).

The neighbours of each grid point are the ``<data> -> <data>`` edges of the training graph, e.g.

.. code-block:: yaml

    graph:
      edges:
        - source_name: data
          target_name: data
          edge_builders:
            - _target_: anemoi.graphs.edges.KNNEdges
              num_nearest_neighbours: 6
          attributes: ${graph.attributes.edges}

    training:
      training_loss:
        datasets:
          data:
            _target_: anemoi.training.losses.CombinedLoss
            losses:
              - _target_: anemoi.training.losses.MSELoss
                scalers: [general_variable, node_weights]
                ignore_nans: true
              - _target_: anemoi.training.losses.GraphLaplacianSmoothnessLoss
                scalers: [node_weights]
                predicted_variables: [z_500, msl]
            loss_weights: [1.0, 0.1]
"""

import logging

import torch
from torch.distributed.distributed_c10d import ProcessGroup
from torch_geometric.data import HeteroData

from anemoi.models.data import TensorLayout
from anemoi.training.losses.base import BaseLoss
from anemoi.training.losses.base import Squash_mode

LOGGER = logging.getLogger(__name__)


class GraphLaplacianSmoothnessLoss(BaseLoss):
    """Penalise the graph Laplacian of the prediction, ``f[i] - mean_{j in N(i)} f[j]``.

    The neighbours ``N(i)`` of grid point ``i`` are the targets of its edges in the
    ``<dataset> -> <dataset>`` edge set of the graph. The squared Laplacian is scaled by the
    configured scalers and reduced like any gridded loss. The target is not used: the penalty
    acts on the prediction everywhere, also where there is no observation. Gridded datasets only.

    Parameters
    ----------
    graph_data : HeteroData, optional
        Training graph, injected by the loss factory.
    data_node_name : str, optional
        Node group of the dataset, injected by the loss factory.
    src_nodes_name, dst_nodes_name : str, optional
        Node groups of the neighbour edges; default ``data_node_name``.
    penalty_weight : float, optional
        Factor on the squared Laplacian, by default 1.0.
    ignore_nans : bool, optional
        Accepted for a common loss interface; target NaNs do not matter here.
    **kwargs
        Ignored.
    """

    needs_graph_data: bool = True
    needs_data_node_name: bool = True

    def __init__(
        self,
        graph_data: HeteroData | None = None,
        data_node_name: str | None = None,
        src_nodes_name: str | None = None,
        dst_nodes_name: str | None = None,
        penalty_weight: float = 1.0,
        ignore_nans: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(ignore_nans=ignore_nans)
        del kwargs
        self.penalty_weight = penalty_weight
        # The Laplacian gathers neighbours with full-grid node indices, so it needs the whole grid.
        self.supports_sharding = False

        src_nodes_name = src_nodes_name or data_node_name or "data"
        dst_nodes_name = dst_nodes_name or data_node_name or "data"
        edge_type = (src_nodes_name, "to", dst_nodes_name)
        if graph_data is None or edge_type not in graph_data.edge_types:
            available = None if graph_data is None else list(graph_data.edge_types)
            msg = (
                f"GraphLaplacianSmoothnessLoss needs the {edge_type} edges of the training graph "
                f"(available: {available}). Add {src_nodes_name} -> {dst_nodes_name} edges to the graph config."
            )
            raise ValueError(msg)

        # The graph may be built on the GPU; the buffers are moved with the module later.
        edge_index = graph_data[edge_type].edge_index.long().cpu()
        num_nodes = graph_data[src_nodes_name].num_nodes
        degree = torch.zeros(num_nodes, dtype=torch.float32)
        degree.scatter_add_(0, edge_index[0], torch.ones(edge_index.shape[1], dtype=torch.float32))
        LOGGER.info("GraphLaplacianSmoothnessLoss uses %d %s edges.", edge_index.shape[1], edge_type)
        # Non-persistent: rebuilt from the graph, so checkpoints do not depend on them.
        self.register_buffer("edge_index", edge_index, persistent=False)
        self.register_buffer("degree", degree.clamp(min=1.0), persistent=False)

    @property
    def name(self) -> str:
        """Used for logging identification purposes."""
        return "graph_laplacian_smoothness"

    def _forward_impl(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        layout: TensorLayout,
        squash: bool = True,
        scaler_indices: tuple[int, ...] | None = None,
        without_scalers: list[str] | list[int] | None = None,
        grid_shard_slice: slice | None = None,
        group: ProcessGroup | None = None,
        squash_mode: Squash_mode = "avg",
        **_kwargs,
    ) -> torch.Tensor:
        del target
        if not layout.has_axis("time"):
            msg = "GraphLaplacianSmoothnessLoss supports gridded datasets only."
            raise ValueError(msg)
        is_sharded = grid_shard_slice is not None
        grid = layout.grid
        if pred.shape[grid] != self.degree.shape[0]:
            msg = (
                f"GraphLaplacianSmoothnessLoss: the prediction has {pred.shape[grid]} grid points, the graph "
                f"{self.degree.shape[0]} nodes."
            )
            raise ValueError(msg)

        src, dst = self.edge_index
        neighbour_sum = torch.zeros_like(pred).index_add_(grid, src, pred.index_select(grid, dst))
        degree_shape = [1] * pred.ndim
        degree_shape[grid] = -1
        neighbour_mean = neighbour_sum / self.degree.view(degree_shape).to(pred.dtype)

        out = self.penalty_weight * torch.square(pred - neighbour_mean)
        out = self.scale(
            out,
            scaler_indices,
            layout=layout,
            without_scalers=without_scalers,
            grid_shard_slice=grid_shard_slice,
        )
        return self.reduce(
            out,
            layout=layout,
            squash=squash,
            group=group if is_sharded else None,
            squash_mode=squash_mode,
        )

    def forward(
        self,
        pred: "Source",  # noqa: F821
        target: "Source",  # noqa: F821
        squash: bool = True,
        *,
        scaler_indices: tuple[int, ...] | None = None,
        without_scalers: list[str] | list[int] | None = None,
        grid_shard_slice: slice | None = None,
        group: ProcessGroup | None = None,
        squash_mode: Squash_mode = "avg",
        **kwargs,
    ) -> torch.Tensor:
        """Compute the smoothness penalty of the prediction.

        Parameters
        ----------
        pred : Source
            Gridded prediction.
        target : Source
            Gridded target (unused).
        squash : bool, optional
            Reduce the variable dimension, by default True.
        scaler_indices : tuple[int, ...], optional
            Indices to subset the calculated scaler with, by default None.
        without_scalers : list[str] | list[int] | None, optional
            Scalers to exclude, by name or dimension. By default None.
        grid_shard_slice : slice, optional
            Slice of the grid if the tensors come sharded, by default None.
        group : ProcessGroup, optional
            Distributed group, by default None.
        squash_mode : {"avg", "sum"}, optional
            Reduction mode for the variable dimension, by default ``"avg"``.
        **kwargs
            Additional keyword arguments.

        Returns
        -------
        torch.Tensor
            Smoothness penalty.
        """
        return self._apply_pairwise(
            pred,
            target,
            self._evaluate_loss_tensor,
            squash=squash,
            scaler_indices=scaler_indices,
            without_scalers=without_scalers,
            grid_shard_slice=grid_shard_slice,
            group=group,
            squash_mode=squash_mode,
            **kwargs,
        )
