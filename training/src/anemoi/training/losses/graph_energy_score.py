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

from anemoi.training.losses.graph_energy_score_base import BaseGraphEnergyScoreLoss
from anemoi.training.losses.graph_score_base import csr_matmul
from anemoi.training.losses.graph_score_base import safe_sqrt
from anemoi.training.losses.graph_score_base import scale_node_differences
from anemoi.training.losses.graph_score_graph import GraphScoreGraph


class GraphEnergyScoreLoss(BaseGraphEnergyScoreLoss):
    """Energy score based on CSR graph neighbourhood norms.

    Without ``loss_graph``, the norm is pointwise and the score is equivalent
    to CRPS. With a graph, each norm is ``sqrt(A @ q**2)``. Rows of ``A``
    represent destination nodes and columns represent source nodes.
    """

    def __init__(
        self,
        fair: bool = True,
        loss_graph: dict | None = None,
        graph_data: HeteroData | None = None,
        no_autocast: bool = True,
        ignore_nans: bool = False,
    ) -> None:
        graph = GraphScoreGraph.from_definition(
            loss_graph,
            graph_data,
            graph_name="Graph energy score neighbourhood",
            allow_none=True,
        )
        super().__init__(
            graph=graph,
            fair=fair,
            no_autocast=no_autocast,
            ignore_nans=ignore_nans,
        )

    @property
    def name(self) -> str:
        prefix = "f" if self.fair else ""
        return f"{prefix}graph_energy_score"

    def _neighbourhood_norm(
        self,
        differences: torch.Tensor,
        matrix: torch.Tensor | None,
        row_weight_sum: torch.Tensor | None,
        node_valid: torch.Tensor | None,
        valid_weight_sum: torch.Tensor | None,
    ) -> torch.Tensor:
        assert row_weight_sum is None
        if matrix is None:
            norm = torch.abs(differences)
            return norm if node_valid is None else norm.masked_fill(~node_valid, torch.nan)

        if node_valid is not None:
            differences = torch.where(node_valid, differences, torch.zeros_like(differences))
        scaled, scale = scale_node_differences(differences)
        squared_norm = csr_matmul(matrix, scaled.square())
        if node_valid is None:
            return scale * safe_sqrt(squared_norm)

        assert valid_weight_sum is not None
        if self.row_normalize:
            squared_norm = squared_norm / torch.where(
                valid_weight_sum > 0,
                valid_weight_sum,
                torch.ones_like(valid_weight_sum),
            )
        norm = scale * safe_sqrt(squared_norm)
        return norm.masked_fill((valid_weight_sum <= 0) | ~node_valid, torch.nan)
