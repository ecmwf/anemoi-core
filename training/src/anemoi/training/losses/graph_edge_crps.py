# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
import torch
from torch_geometric.data import HeteroData

from anemoi.training.losses.graph_edge_score_base import BaseGraphEdgeScoreLoss
from anemoi.training.losses.graph_score_graph import GraphScoreGraph


class GraphEdgeCRPSLoss(BaseGraphEdgeScoreLoss):
    """Almost-fair CRPS over signed graph edge differences.

    For each graph edge ``src -> dst`` this loss scores the ensemble of signed
    differences ``x[src] - x[dst]`` against the observed signed difference. Edge
    scores are then aggregated back to destination nodes so the result remains a
    node-shaped score tensor compatible with the usual scalers and multiscale
    wrapper.

    ``alpha`` controls the finite-ensemble correction: ``1.0`` is the fair CRPS,
    ``0.0`` is the standard empirical CRPS, and intermediate values reproduce
    the almost-fair interpolation used by :class:`anemoi.training.losses.CRPS`.

    ``loss_graph.row_normalize`` controls whether each destination node uses a
    weighted average or a raw weighted sum over incoming edges.

    When ``ignore_nans=True``, invalid nodes remove every edge that touches
    them. The remaining valid edges are only re-normalized when
    ``row_normalize=True``; otherwise they keep their original raw edge
    weights.
    """

    def __init__(
        self,
        loss_graph: dict,
        graph_data: HeteroData,
        alpha: float = 1.0,
        no_autocast: bool = True,
        ignore_nans: bool = False,
    ) -> None:
        """Graph edge-difference CRPS.

        Parameters
        ----------
        loss_graph : dict
            Graph-based edge definition.
        graph_data : HeteroData
            Graph data used to build the edge graph.
        alpha : float
            Factor for the linear combination of fair and empirical CRPS. A
            value of 1.0 is fully fair; 0.0 is empirical CRPS.
        no_autocast : bool
            Whether to disable autocast for the full edge CRPS calculation.
        ignore_nans : bool
            Whether to drop invalid nodes before graph aggregation. Remaining
            edge weights are only re-normalized when
            ``loss_graph.row_normalize=True``.
        """
        assert 0.0 <= alpha <= 1.0, "alpha must be in the interval [0, 1]."

        graph = GraphScoreGraph.from_definition(
            loss_graph,
            graph_data,
            graph_name="Graph edge CRPS neighbourhood",
            allow_none=False,
            require_square=True,
        )
        super().__init__(graph=graph, no_autocast=no_autocast, ignore_nans=ignore_nans)
        self.alpha = alpha

    @property
    def name(self) -> str:
        if self.alpha == 1.0:
            return "fgraph_edge_crps"
        if self.alpha == 0.0:
            return "graph_edge_crps"
        return f"afgraph_edge_crps{self.alpha:.2f}"

    def _pair_coefficient(self, ensemble_size: int) -> float:
        """Return the almost-fair coefficient for unordered member pairs."""
        return (1.0 - (1.0 - self.alpha) / ensemble_size) / (ensemble_size * (ensemble_size - 1))

    def _compute_local_score_tensor(
        self,
        y_pred_ens: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        """Compute edge CRPS and aggregate it to shape ``(B, T, N, V)``."""
        ensemble_size = y_pred_ens.shape[2]
        edge_valid = self._valid_edges(y_pred_ens, y)

        obs_edge_difference = self._edge_difference(y)
        obs_term = torch.zeros_like(obs_edge_difference)
        pair_term = torch.zeros_like(obs_edge_difference)

        # Accumulate member-to-observation errors and unordered member-pair
        # distances in edge space before aggregating once to destination nodes.
        for i in range(ensemble_size):
            member_edge_difference = self._edge_difference(y_pred_ens[:, :, i])

            member_obs_error = torch.abs(member_edge_difference - obs_edge_difference)
            if edge_valid is not None:
                member_obs_error = torch.where(
                    edge_valid,
                    member_obs_error,
                    torch.zeros_like(member_obs_error),
                )
            obs_term = obs_term + member_obs_error

            for j in range(i + 1, ensemble_size):
                pair_edge_difference = self._edge_difference(y_pred_ens[:, :, j])
                pair_distance = torch.abs(member_edge_difference - pair_edge_difference)
                if edge_valid is not None:
                    pair_distance = torch.where(
                        edge_valid,
                        pair_distance,
                        torch.zeros_like(pair_distance),
                    )
                pair_term = pair_term + pair_distance

        score_edges = obs_term / ensemble_size - self._pair_coefficient(ensemble_size) * pair_term

        if edge_valid is not None:
            score_edges = score_edges.masked_fill(~edge_valid, torch.nan)

        return self._aggregate_edges(score_edges, valid_edges=edge_valid)
