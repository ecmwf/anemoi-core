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
    """Almost-fair CRPS over edge differences stored by a CSR graph."""

    def __init__(
        self,
        loss_graph: dict,
        graph_data: HeteroData,
        alpha: float = 1.0,
        no_autocast: bool = True,
        ignore_nans: bool = False,
    ) -> None:
        assert 0.0 <= alpha <= 1.0, "alpha must be in the interval [0, 1]."
        graph = GraphScoreGraph.from_definition(
            loss_graph,
            graph_data,
            graph_name="Graph edge CRPS neighbourhood",
            allow_none=False,
        )
        assert graph is not None
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
        return (1.0 - (1.0 - self.alpha) / ensemble_size) / (ensemble_size * (ensemble_size - 1))

    def _compute_local_score_tensor(
        self,
        y_pred_ens: torch.Tensor,
        y_target: torch.Tensor,
        matrix: torch.Tensor | None,
        source_index: torch.Tensor | None,
        destination_index: torch.Tensor | None,
        edge_weights: torch.Tensor | None,
    ) -> torch.Tensor:
        assert matrix is None
        assert source_index is not None
        assert destination_index is not None
        assert edge_weights is not None
        ensemble_size = y_pred_ens.shape[2]
        node_valid, edge_valid, valid_weight_sum = self._compute_edge_validity(
            y_pred_ens,
            y_target,
            source_index,
            destination_index,
            edge_weights,
        )

        observed_edge = self._edge_difference(y_target, source_index, destination_index)
        observation_sum = torch.zeros_like(observed_edge)
        pair_sum = torch.zeros_like(observed_edge)
        for first in range(ensemble_size):
            first_edge = self._edge_difference(
                y_pred_ens[:, :, first],
                source_index,
                destination_index,
            )
            observation_distance = torch.abs(first_edge - observed_edge)
            if edge_valid is not None:
                observation_distance = torch.where(
                    edge_valid,
                    observation_distance,
                    torch.zeros_like(observation_distance),
                )
            observation_sum = observation_sum + observation_distance

            for second in range(first + 1, ensemble_size):
                second_edge = self._edge_difference(
                    y_pred_ens[:, :, second],
                    source_index,
                    destination_index,
                )
                pair_distance = torch.abs(first_edge - second_edge)
                if edge_valid is not None:
                    pair_distance = torch.where(
                        edge_valid,
                        pair_distance,
                        torch.zeros_like(pair_distance),
                    )
                pair_sum = pair_sum + pair_distance

        edge_score = observation_sum / ensemble_size - self._pair_coefficient(ensemble_size) * pair_sum
        return self._aggregate_edge_values(
            edge_score,
            destination_index,
            edge_weights,
            y_target.shape[-2],
            node_valid=node_valid,
            edge_valid=edge_valid,
            valid_weight_sum=valid_weight_sum,
        )
