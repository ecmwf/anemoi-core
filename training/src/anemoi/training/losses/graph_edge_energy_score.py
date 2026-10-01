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

from anemoi.training.losses.graph_edge_operations import weighted_edge_row_l2_norm
from anemoi.training.losses.graph_energy_score_base import BaseGraphEnergyScoreLoss
from anemoi.training.losses.graph_score_graph import GraphScoreGraph


class GraphEdgeEnergyScoreLoss(BaseGraphEnergyScoreLoss):
    """Energy score over graph edge differences."""

    uses_edge_tensors: bool = True

    def __init__(
        self,
        loss_graph: dict,
        graph_data: HeteroData,
        fair: bool = True,
        no_autocast: bool = True,
        ignore_nans: bool = False,
    ) -> None:
        graph = GraphScoreGraph.from_definition(
            loss_graph,
            graph_data,
            graph_name="Graph edge energy score neighbourhood",
            allow_none=False,
            remove_self_edges=True,
        )
        assert graph is not None
        super().__init__(
            graph=graph,
            fair=fair,
            no_autocast=no_autocast,
            ignore_nans=ignore_nans,
        )

    @property
    def name(self) -> str:
        prefix = "f" if self.fair else ""
        return f"{prefix}graph_edge_energy_score"

    def _neighbourhood_norm(
        self,
        differences: torch.Tensor,
        matrix: torch.Tensor | None,
        source_index: torch.Tensor | None,
        destination_index: torch.Tensor | None,
        edge_weights: torch.Tensor | None,
        node_valid: torch.Tensor | None,
        edge_valid: torch.Tensor | None,
        valid_weight_sum: torch.Tensor | None,
    ) -> torch.Tensor:
        assert matrix is None
        assert source_index is not None
        assert destination_index is not None
        assert edge_weights is not None

        return weighted_edge_row_l2_norm(
            differences,
            source_index,
            destination_index,
            edge_weights,
            node_valid=node_valid,
            edge_valid=edge_valid,
            valid_weight_sum=valid_weight_sum,
            row_normalize=self.row_normalize,
        )
