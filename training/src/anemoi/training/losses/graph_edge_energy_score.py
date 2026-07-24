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
    """Fair energy score over graph edge differences.

    For each graph edge ``src -> dst``, the difference is
    ``x[src] - x[dst]``. The incoming edge differences at each node are scored
    together, giving one score for each node and variable.

    With ``loss_graph.row_normalize=True``, the distance is
    ``sqrt(sum_e w_e d_e**2)`` with edge weights that sum to one.

    When ``ignore_nans=True``, edges containing missing values are left out. If
    ``row_normalize=True``, the weights of the remaining edges are scaled to
    sum to one; otherwise their original weights are kept.
    """

    def __init__(
        self,
        loss_graph: dict,
        graph_data: HeteroData,
        fair: bool = True,
        no_autocast: bool = True,
        ignore_nans: bool = False,
    ) -> None:
        """Graph edge-difference energy score.

        Parameters
        ----------
        loss_graph : dict
            Graph-based edge definition.
        graph_data : HeteroData
            Graph data used to build the edge graph.
        fair : bool
            Whether to use the fair ensemble correction.
        no_autocast : bool
            Whether to keep the original numerical precision throughout.
        ignore_nans : bool
            Whether to leave out edges containing missing values. With
            ``loss_graph.row_normalize=True``, the remaining weights are
            scaled to sum to one.
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
        """Return one edge energy score per batch, output step, node, and variable."""
        ensemble_size = y_pred_ens.shape[2]
        edge_valid = self._valid_edges(y_pred_ens, y)

        obs_edge_difference = self._edge_difference(y)

        # Mean distance between each member and the observation, measured over
        # the incoming edge differences of each node.
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

        # Sum of distances over unordered pairs of ensemble members.
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
