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


class GraphEdgeEnergyScoreLoss(BaseGraphEnergyScoreLoss):
    """Energy score for graph edge differences.

    This class inherits from ``BaseGraphEnergyScoreLoss`` rather than
    ``BaseGraphEdgeScoreLoss`` because it calculates edge norms without
    materializing an edge tensor.

    For node differences ``q``, the weighted squared edge norm is
    ``A @ q**2 - 2*q*(A @ q) + q**2*(A @ 1)``. This identity avoids an
    edge tensor.
    """

    uses_row_weight_sums: bool = True

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
        row_weight_sum: torch.Tensor | None,
        node_valid: torch.Tensor | None,
        valid_weight_sum: torch.Tensor | None,
    ) -> torch.Tensor:
        assert matrix is not None
        assert row_weight_sum is not None

        safe_differences = differences
        if node_valid is not None:
            safe_differences = torch.where(node_valid, differences, torch.zeros_like(differences))

        # Edge differences are invariant to a constant spatial offset.
        # Improve stability of computation
        safe_differences = safe_differences - safe_differences[..., :1, :]
        if node_valid is not None:
            safe_differences = torch.where(
                node_valid,
                safe_differences,
                torch.zeros_like(safe_differences),
            )

        scaled, scale = scale_node_differences(safe_differences)
        moments = csr_matmul(matrix, torch.cat((scaled, scaled.square()), dim=-1))
        projected, projected_square = moments.chunk(2, dim=-1)

        if node_valid is None:
            row_shape = (1,) * (scaled.ndim - 2) + (row_weight_sum.shape[0], 1)
            effective_weight_sum = row_weight_sum.view(row_shape)
        else:
            assert valid_weight_sum is not None
            effective_weight_sum = valid_weight_sum

        squared_norm = projected_square - 2.0 * scaled * projected + scaled.square() * effective_weight_sum
        if node_valid is not None and self.row_normalize:
            squared_norm = squared_norm / torch.where(
                effective_weight_sum > 0,
                effective_weight_sum,
                torch.ones_like(effective_weight_sum),
            )
        norm = scale * safe_sqrt(squared_norm)
        if node_valid is not None:
            norm = norm.masked_fill((effective_weight_sum <= 0) | ~node_valid, torch.nan)
        return norm
