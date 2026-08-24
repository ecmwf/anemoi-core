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


class GraphVariogramScoreLoss(BaseGraphEdgeScoreLoss):
    """Variogram score over node pairs stored by a CSR graph."""

    def __init__(
        self,
        loss_graph: dict,
        graph_data: HeteroData,
        p: float = 1.0,
        fair: bool = True,
        no_autocast: bool = True,
        ignore_nans: bool = False,
    ) -> None:
        assert p > 0.0, "p must be strictly positive."
        graph = GraphScoreGraph.from_definition(
            loss_graph,
            graph_data,
            graph_name="Graph variogram neighbourhood",
            allow_none=False,
        )
        assert graph is not None
        super().__init__(graph=graph, no_autocast=no_autocast, ignore_nans=ignore_nans)
        self.p = p
        self.fair = fair

    @property
    def name(self) -> str:
        prefix = "f" if self.fair else ""
        return f"{prefix}graph_variogram_score_p{self.p:g}"

    def _edge_variogram(
        self,
        node_values: torch.Tensor,
        source_index: torch.Tensor,
        destination_index: torch.Tensor,
        edge_valid: torch.Tensor | None,
    ) -> torch.Tensor:
        edge_difference = self._edge_difference(node_values, source_index, destination_index)
        if edge_valid is not None:
            edge_difference = torch.where(
                edge_valid,
                edge_difference,
                torch.zeros_like(edge_difference),
            )
        return torch.abs(edge_difference).pow(self.p)

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

        observed_variogram = self._edge_variogram(
            y_target,
            source_index,
            destination_index,
            edge_valid,
        )
        member_sum = torch.zeros_like(observed_variogram)
        if self.fair:
            member_cross_sum = torch.zeros_like(observed_variogram)
            running_sum = torch.zeros_like(observed_variogram)

        for member in range(ensemble_size):
            member_variogram = self._edge_variogram(
                y_pred_ens[:, :, member],
                source_index,
                destination_index,
                edge_valid,
            )
            member_sum = member_sum + member_variogram
            if self.fair:
                member_cross_sum = member_cross_sum + member_variogram * running_sum
                running_sum = running_sum + member_variogram

        member_mean = member_sum / ensemble_size
        if self.fair:
            edge_score = (
                observed_variogram.square()
                - 2.0 * observed_variogram * member_mean
                + 2.0 * member_cross_sum / (ensemble_size * (ensemble_size - 1))
            )
        else:
            edge_score = (member_mean - observed_variogram).square()

        return self._aggregate_edge_values(
            edge_score,
            destination_index,
            edge_weights,
            y_target.shape[-2],
            node_valid=node_valid,
            edge_valid=edge_valid,
            valid_weight_sum=valid_weight_sum,
        )
