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


class GraphEdgeEnergyScoreLoss(BaseGraphEdgeScoreLoss):
    """Fair energy score over signed graph edge-difference neighbourhoods.

    For each graph edge ``src -> dst`` this loss forms signed differences
    ``x[src] - x[dst]``. For each destination node, the incoming edge
    differences are scored jointly with an energy score. The resulting score is
    node-shaped, so it remains compatible with the usual scalers, reductions,
    sharding path, and multiscale wrapper.

    The edge-difference vector norm mirrors ``GraphEnergyScoreLoss``: when
    ``loss_graph.row_normalize=True`` it is a weighted RMS-style norm over
    incoming edges, so neighbourhood size does not change the scale by itself.

    When ``ignore_nans=True``, invalid nodes remove every edge that touches
    them. The remaining valid edges are only re-normalized when
    ``row_normalize=True``; otherwise they keep their original raw edge
    weights.
    """

    def __init__(
        self,
        fair: bool = True,
        loss_graph: dict | None = None,
        graph_data: HeteroData | None = None,
        no_autocast: bool = True,
        ignore_nans: bool = False,
    ) -> None:
        """Graph edge-difference energy score.

        Parameters
        ----------
        fair : bool
            Whether to use the fair ensemble correction.
        loss_graph : dict | None
            Graph-based edge definition.
        graph_data : HeteroData | None
            Graph data used to build the edge graph.
        no_autocast : bool
            Whether to disable autocast for the full edge energy score
            calculation.
        ignore_nans : bool
            Whether to drop invalid nodes before graph aggregation. Remaining
            edge weights are only re-normalized when
            ``loss_graph.row_normalize=True``.
        """
        graph = GraphScoreGraph.from_definition(
            loss_graph,
            graph_data,
            graph_name="Graph edge energy score neighbourhood",
            allow_none=False,
            require_square=True,
        )
        super().__init__(graph=graph, no_autocast=no_autocast, ignore_nans=ignore_nans)
        self.fair = fair

    @property
    def name(self) -> str:
        prefix = "f" if self.fair else ""
        return f"{prefix}graph_edge_energy_score"

    def _compute_local_score_tensor(
        self,
        y_pred_ens: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the edge-difference energy score with shape ``(B, T, N, V)``."""
        ensemble_size = y_pred_ens.shape[2]
        edge_valid = self._valid_edges(y_pred_ens, y)

        obs_edge_difference = self._edge_difference(y)

        # First term: mean member-to-observation distance between the vectors
        # of incoming signed edge differences at each destination node.
        obs_term_sum = torch.zeros(
            (*y.shape[:2], self.num_nodes, y.shape[-1]),
            dtype=y_pred_ens.dtype,
            device=y_pred_ens.device,
        )
        for i in range(ensemble_size):
            member_edge_difference = self._edge_difference(y_pred_ens[:, :, i])
            obs_term_sum = obs_term_sum + self.graph.weighted_row_l2_norm(
                member_edge_difference - obs_edge_difference,
                valid_edges=edge_valid,
            )
        obs_term = obs_term_sum / ensemble_size

        # Second term: distance between every unordered pair of members.
        pair_distance_sum = torch.zeros_like(obs_term)
        for i in range(ensemble_size):
            member_edge_difference = self._edge_difference(y_pred_ens[:, :, i])
            for j in range(i + 1, ensemble_size):
                pair_edge_difference = self._edge_difference(y_pred_ens[:, :, j])
                pair_distance_sum = pair_distance_sum + self.graph.weighted_row_l2_norm(
                    member_edge_difference - pair_edge_difference,
                    valid_edges=edge_valid,
                )

        pair_coefficient = 1.0 / (ensemble_size * (ensemble_size - 1)) if self.fair else 1.0 / (ensemble_size**2)
        return obs_term - pair_coefficient * pair_distance_sum
