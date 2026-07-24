# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Shared edge transformations for graph score losses.

Tensor shapes use ``B`` for batch, ``T`` for time, ``M`` for ensemble
members, ``N`` for nodes, ``E`` for edges, and ``V`` for variables.
"""

import torch

from anemoi.training.losses.graph_score_base import BaseGraphScoreLoss
from anemoi.training.losses.graph_score_graph import GraphScoreGraph


class BaseGraphEdgeScoreLoss(BaseGraphScoreLoss):
    """Shared graph access, validity, and aggregation for edge-based scores."""

    graph: GraphScoreGraph

    def __init__(
        self,
        *,
        graph: GraphScoreGraph,
        no_autocast: bool = True,
        ignore_nans: bool = False,
    ) -> None:
        super().__init__(graph=graph, no_autocast=no_autocast, ignore_nans=ignore_nans)

    @property
    def edge_index(self) -> torch.Tensor:
        return self.graph.edge_index

    @property
    def edge_src_index(self) -> torch.Tensor:
        return self.graph.edge_src_index

    @property
    def edge_dst_index(self) -> torch.Tensor:
        return self.graph.edge_dst_index

    @property
    def edge_weights(self) -> torch.Tensor:
        return self.graph.edge_weights

    @property
    def num_nodes(self) -> int:
        return self.graph.num_nodes

    def _edge_difference(self, node_values: torch.Tensor) -> torch.Tensor:
        """Return signed ``source - destination`` values with shape ``(..., E, V)``."""
        return node_values[..., self.edge_src_index, :] - node_values[..., self.edge_dst_index, :]

    def _valid_edges(
        self,
        y_pred_ens: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor | None:
        """Return the complete-case edge mask with shape ``(B, T, E, V)``.

        A node is valid only when its target and every ensemble member are
        finite. Requiring complete cases keeps the effective ensemble size
        fixed within each score calculation.
        """
        if not self.ignore_nans:
            return None
        node_valid = torch.isfinite(y) & torch.isfinite(y_pred_ens).all(dim=2)
        return node_valid[..., self.edge_src_index, :] & node_valid[..., self.edge_dst_index, :]

    def _aggregate_edges(
        self,
        edge_values: torch.Tensor,
        valid_edges: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Aggregate ``(B, T, E, V)`` edge scores to ``(B, T, N, V)`` nodes."""
        weights = self.edge_weights.to(dtype=edge_values.dtype).view(1, 1, -1, 1)
        if valid_edges is not None:
            safe_edge_values = torch.where(
                valid_edges,
                edge_values,
                torch.zeros_like(edge_values),
            )
            valid_weights = weights * valid_edges.to(dtype=edge_values.dtype)

            valid_row_weight_sum = torch.zeros(
                (*edge_values.shape[:2], self.num_nodes, edge_values.shape[-1]),
                dtype=edge_values.dtype,
                device=edge_values.device,
            )
            valid_row_weight_sum.index_add_(2, self.edge_dst_index, valid_weights)

            # Re-normalize only graphs configured for row normalization. Raw
            # weighted graphs retain their original scale after masking.
            if self.row_normalize:
                edge_weight_sums = valid_row_weight_sum[:, :, self.edge_dst_index, :]
                safe_weight_sums = torch.where(
                    edge_weight_sums > 0,
                    edge_weight_sums,
                    torch.ones_like(edge_weight_sums),
                )
                effective_weights = torch.where(
                    valid_edges & (edge_weight_sums > 0),
                    weights / safe_weight_sums,
                    torch.zeros_like(weights),
                )
            else:
                effective_weights = torch.where(
                    valid_edges,
                    weights,
                    torch.zeros_like(weights),
                )

            node_scores = torch.zeros_like(valid_row_weight_sum)
            node_scores.index_add_(
                2,
                self.edge_dst_index,
                safe_edge_values * effective_weights,
            )
            return node_scores.masked_fill(valid_row_weight_sum <= 0, torch.nan)

        weighted_values = edge_values * weights
        node_scores = torch.zeros(
            (*weighted_values.shape[:2], self.num_nodes, weighted_values.shape[-1]),
            dtype=weighted_values.dtype,
            device=weighted_values.device,
        )
        node_scores.index_add_(2, self.edge_dst_index, weighted_values)
        return node_scores
