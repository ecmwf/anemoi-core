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

from anemoi.training.losses.graph_score_base import BaseGraphScoreLoss
from anemoi.training.losses.graph_score_graph import GraphScoreGraph


class GraphEnergyScoreLoss(BaseGraphScoreLoss):
    """Fair energy score over graph neighbourhoods.

    Without ``loss_graph``, this reduces to the pointwise CRPS-equivalent
    energy score. With a graph, the absolute pointwise difference is replaced
    by ``sqrt(sum_j w_j (x_j - y_j)**2)`` over the incoming neighbours of each
    node.

    ``loss_graph.row_normalize`` controls whether each target node uses
    a weighted neighbourhood average or a raw weighted sum.

    When ``ignore_nans=True``, edges containing missing values are left out. If
    ``row_normalize=True``, the weights of the remaining edges are scaled to
    sum to one; otherwise their original weights are kept.
    """

    def __init__(
        self,
        fair: bool = True,
        loss_graph: dict | None = None,
        graph_data: HeteroData | None = None,
        no_autocast: bool = True,
        ignore_nans: bool = False,
    ) -> None:
        """Graph neighbourhood energy score.

        Parameters
        ----------
        fair : bool
            Whether to use the fair ensemble correction.
        loss_graph : dict | None
            Graph-based neighbourhood definition. If ``None``, use the
            pointwise CRPS-equivalent energy score.
        graph_data : HeteroData | None
            Graph data used to build the neighbourhood graph.
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
            graph_name="Graph energy score neighbourhood",
            allow_none=True,
            require_square=True,
        )
        super().__init__(graph=graph, no_autocast=no_autocast, ignore_nans=ignore_nans)
        self.fair = fair

    @property
    def name(self) -> str:
        prefix = "f" if self.fair else ""
        return f"{prefix}graph_energy_score"

    def _stable_neighbourhood_norm(
        self,
        node_differences: torch.Tensor,
        node_valid: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return the weighted distance over each incoming neighbourhood."""
        if self.graph is None:
            neighbourhood_norms = torch.abs(node_differences)
            if node_valid is not None:
                neighbourhood_norms = neighbourhood_norms.masked_fill(
                    ~node_valid.unsqueeze(2),
                    torch.nan,
                )
            return neighbourhood_norms

        valid_edges = None
        expanded_node_valid = None
        if node_valid is not None:
            expanded_node_valid = node_valid.unsqueeze(2).expand(
                *node_differences.shape[:-2],
                node_valid.shape[-2],
                node_valid.shape[-1],
            )
            valid_edges = expanded_node_valid[..., self.graph.edge_src_index, :]

        edge_values = node_differences[..., self.graph.edge_src_index, :]
        neighbourhood_norms = self.graph.weighted_row_l2_norm(
            edge_values,
            valid_edges=valid_edges,
        )
        if expanded_node_valid is not None:
            neighbourhood_norms = neighbourhood_norms.masked_fill(
                ~expanded_node_valid,
                torch.nan,
            )
        return neighbourhood_norms

    def _compute_local_score_tensor(
        self,
        y_pred_ens: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        """Return one graph energy score per batch, output step, node, and variable."""
        ensemble_size = y_pred_ens.shape[2]

        # Leave out a node if its observation or any ensemble value is missing.
        # Both parts of the score then use the same ensemble members.
        node_valid = None
        if self.ignore_nans:
            node_valid = torch.isfinite(y) & torch.isfinite(y_pred_ens).all(dim=2)

        # Mean distance from each ensemble member to the observation.
        obs_distance = y_pred_ens - y.unsqueeze(2)
        obs_term = self._stable_neighbourhood_norm(
            obs_distance,
            node_valid=node_valid,
        ).mean(dim=2)

        # Sum of distances over unordered pairs of ensemble members.
        pair_distance_sum = torch.zeros_like(obs_term)
        for i in range(ensemble_size):
            pair_distance = y_pred_ens[:, :, i].unsqueeze(2) - y_pred_ens[:, :, i + 1 :]
            if pair_distance.shape[2] == 0:
                continue
            pair_distance_sum = pair_distance_sum + self._stable_neighbourhood_norm(
                pair_distance,
                node_valid=node_valid,
            ).sum(dim=2)

        pair_coefficient = 1.0 / (ensemble_size * (ensemble_size - 1)) if self.fair else 1.0 / (ensemble_size**2)
        return obs_term - pair_coefficient * pair_distance_sum
