# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Shared CSR structure and reduction for nonlinear edge scores."""

import torch

from anemoi.training.losses.graph_score_base import BaseGraphScoreLoss
from anemoi.training.losses.graph_score_graph import GraphScoreGraph


class BaseGraphEdgeScoreLoss(BaseGraphScoreLoss):
    """Evaluate nonlinear edge statistics and reduce them by CSR rows."""

    uses_edge_tensors: bool = True
    graph: GraphScoreGraph

    def __init__(
        self,
        *,
        graph: GraphScoreGraph,
        no_autocast: bool = True,
        ignore_nans: bool = False,
    ) -> None:
        super().__init__(graph=graph, no_autocast=no_autocast, ignore_nans=ignore_nans)

    @staticmethod
    def _edge_difference(
        node_values: torch.Tensor,
        source_index: torch.Tensor,
        destination_index: torch.Tensor,
    ) -> torch.Tensor:
        """Return ``source - destination`` for each nonzero CSR entry."""
        return node_values[..., source_index, :] - node_values[..., destination_index, :]

    def _compute_edge_validity(
        self,
        y_pred_ens: torch.Tensor,
        y_target: torch.Tensor,
        source_index: torch.Tensor,
        destination_index: torch.Tensor,
        edge_weights: torch.Tensor,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
        """Compute node and edge validity plus valid weight sums per destination."""
        if not self.ignore_nans:
            return None, None, None
        node_valid = torch.isfinite(y_target) & torch.isfinite(y_pred_ens).all(dim=2)
        edge_valid = node_valid[..., source_index, :] & node_valid[..., destination_index, :]
        weight_shape = (1,) * (edge_valid.ndim - 2) + (-1, 1)
        valid_edge_weights = edge_valid.to(dtype=y_pred_ens.dtype) * edge_weights.view(weight_shape)
        valid_weight_sum = torch.zeros_like(node_valid, dtype=y_pred_ens.dtype)
        valid_weight_sum.index_add_(-2, destination_index, valid_edge_weights)
        return node_valid, edge_valid, valid_weight_sum

    def _aggregate_edge_values(
        self,
        edge_values: torch.Tensor,
        destination_index: torch.Tensor,
        edge_weights: torch.Tensor,
        num_nodes: int,
        *,
        node_valid: torch.Tensor | None,
        edge_valid: torch.Tensor | None,
        valid_weight_sum: torch.Tensor | None,
    ) -> torch.Tensor:
        """Return a weighted sum for each CSR row with NaN normalization."""
        weight_shape = (1,) * (edge_values.ndim - 2) + (-1, 1)
        weights = edge_weights.to(dtype=edge_values.dtype).view(weight_shape)
        if edge_valid is not None:
            edge_values = torch.where(edge_valid, edge_values, torch.zeros_like(edge_values))

        node_values = torch.zeros(
            (*edge_values.shape[:-2], num_nodes, edge_values.shape[-1]),
            dtype=edge_values.dtype,
            device=edge_values.device,
        )
        node_values.index_add_(-2, destination_index, edge_values * weights)

        if node_valid is None:
            return node_values
        assert valid_weight_sum is not None
        if self.row_normalize:
            safe_weight_sum = torch.where(
                valid_weight_sum > 0,
                valid_weight_sum,
                torch.ones_like(valid_weight_sum),
            )
            node_values = node_values / safe_weight_sum
        return node_values.masked_fill((valid_weight_sum <= 0) | ~node_valid, torch.nan)
