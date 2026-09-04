# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Shared ensemble aggregation for CSR energy scores."""

from abc import abstractmethod

import torch

from anemoi.training.losses.graph_score_base import BaseGraphScoreLoss
from anemoi.training.losses.graph_score_base import csr_matmul
from anemoi.training.losses.graph_score_graph import GraphScoreGraph


class BaseGraphEnergyScoreLoss(BaseGraphScoreLoss):
    """Combine a neighbourhood norm into empirical or fair energy scores."""

    uses_row_weight_sums: bool = False

    def __init__(
        self,
        *,
        graph: GraphScoreGraph | None,
        fair: bool,
        no_autocast: bool = True,
        ignore_nans: bool = False,
    ) -> None:
        super().__init__(graph=graph, no_autocast=no_autocast, ignore_nans=ignore_nans)
        self.fair = fair

    @abstractmethod
    def _neighbourhood_norm(
        self,
        differences: torch.Tensor,
        matrix: torch.Tensor | None,
        row_weight_sum: torch.Tensor | None,
        node_valid: torch.Tensor | None,
        valid_weight_sum: torch.Tensor | None,
    ) -> torch.Tensor:
        """Return one norm per batch, time, node, and variable."""

    def _compute_local_score_tensor(
        self,
        y_pred_ens: torch.Tensor,
        y_target: torch.Tensor,
        matrix: torch.Tensor | None,
        source_index: torch.Tensor | None,
        destination_index: torch.Tensor | None,
        edge_weights: torch.Tensor | None,
    ) -> torch.Tensor:
        """Evaluate distances from each member to the observation and between unordered member pairs."""
        assert source_index is None
        assert destination_index is None
        assert edge_weights is None
        ensemble_size = y_pred_ens.shape[2]

        node_valid = None
        valid_weight_sum = None
        if self.ignore_nans:
            node_valid = torch.isfinite(y_target) & torch.isfinite(y_pred_ens).all(dim=2)
            if matrix is not None:
                valid_weight_sum = csr_matmul(matrix, node_valid.to(dtype=y_pred_ens.dtype))

        row_weight_sum = None
        if self.uses_row_weight_sums:
            assert matrix is not None
            ones = torch.ones(
                matrix.shape[1],
                1,
                dtype=y_pred_ens.dtype,
                device=y_pred_ens.device,
            )
            row_weight_sum = torch.sparse.mm(matrix, ones).squeeze(-1)

        observation_sum = torch.zeros_like(y_target)
        for member in range(ensemble_size):
            observation_sum = observation_sum + self._neighbourhood_norm(
                y_pred_ens[:, :, member] - y_target,
                matrix,
                row_weight_sum,
                node_valid,
                valid_weight_sum,
            )

        pair_sum = torch.zeros_like(y_target)
        for first in range(ensemble_size):
            for second in range(first + 1, ensemble_size):
                pair_sum = pair_sum + self._neighbourhood_norm(
                    y_pred_ens[:, :, first] - y_pred_ens[:, :, second],
                    matrix,
                    row_weight_sum,
                    node_valid,
                    valid_weight_sum,
                )

        pair_coefficient = 1.0 / (ensemble_size * (ensemble_size - 1)) if self.fair else 1.0 / (ensemble_size**2)
        return observation_sum / ensemble_size - pair_coefficient * pair_sum
