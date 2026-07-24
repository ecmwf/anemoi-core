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
    """Variogram score over graph neighbourhood pairs.

    Notes
    -----
    The fair ensemble estimator used here is derived by combining three facts:

    - Scheuerer and Hamill (2015) define the variogram score.
    - Ferro (2014) defines a fair ensemble score as an unbiased estimator of
      the score of the underlying forecast distribution when the ensemble is a
      random sample.
    - Allen, Ginsbourger, and Ziegel (2024) show that the variogram score is a
      kernel score.

    Applying Ferro's unbiased ensemble correction to the variogram kernel gives
    the off-diagonal ``M (M - 1)`` normalization used below for ``fair=True``.
    The expanded finite-ensemble expression matches the formulation documented
    in the ``scoringrules`` reference implementation.

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
        p: float = 1.0,
        fair: bool = True,
        no_autocast: bool = True,
        ignore_nans: bool = False,
    ) -> None:
        """Graph neighbourhood variogram score.

        Parameters
        ----------
        loss_graph : dict
            Graph-based neighbourhood pair definition.
        graph_data : HeteroData
            Graph data used to build the neighbourhood pair graph.
        p : float
            Variogram exponent. Typical values are in (0, 2].
        fair : bool
            Whether to use the fair ensemble correction.
        no_autocast : bool
            Whether to disable autocast for the full variogram score
            calculation.
        ignore_nans : bool
            Whether to drop invalid nodes before graph aggregation. Remaining
            edge weights are only re-normalized when
            ``loss_graph.row_normalize=True``.
        """
        assert p > 0.0, "p must be strictly positive."

        graph = GraphScoreGraph.from_definition(
            loss_graph,
            graph_data,
            graph_name="Graph variogram neighbourhood",
            allow_none=False,
            require_square=True,
        )
        super().__init__(graph=graph, no_autocast=no_autocast, ignore_nans=ignore_nans)
        self.p = p
        self.fair = fair

    @property
    def name(self) -> str:
        prefix = "f" if self.fair else ""
        return f"{prefix}graph_variogram_score_p{self.p:g}"

    def _edge_variogram(self, x: torch.Tensor) -> torch.Tensor:
        """Return ``|x[src] - x[dst]|**p`` with shape ``(..., E, V)``."""
        return torch.abs(self._edge_difference(x)).pow(self.p)

    def _compute_local_score_tensor(
        self,
        y_pred_ens: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        """Compute edge variograms and aggregate them to ``(B, T, N, V)``."""
        ensemble_size = y_pred_ens.shape[2]
        edge_valid = self._valid_edges(y_pred_ens, y)

        obs_variogram = self._edge_variogram(y)
        if edge_valid is not None:
            obs_variogram = torch.where(
                edge_valid,
                obs_variogram,
                torch.zeros_like(obs_variogram),
            )

        member_sum = torch.zeros_like(obs_variogram)
        if self.fair:
            member_cross_sum = torch.zeros_like(obs_variogram)
            running_sum = torch.zeros_like(obs_variogram)

        # The running sum forms every unordered cross-product without storing
        # all member-pair tensors at once.
        for i in range(ensemble_size):
            member_variogram = self._edge_variogram(y_pred_ens[:, :, i])
            if edge_valid is not None:
                member_variogram = torch.where(
                    edge_valid,
                    member_variogram,
                    torch.zeros_like(member_variogram),
                )
            member_sum = member_sum + member_variogram
            if self.fair:
                member_cross_sum = member_cross_sum + member_variogram * running_sum
                running_sum = running_sum + member_variogram

        member_mean = member_sum / ensemble_size
        if self.fair:
            score_edges = (
                obs_variogram.square()
                - 2.0 * obs_variogram * member_mean
                + 2.0 * member_cross_sum / (ensemble_size * (ensemble_size - 1))
            )
        else:
            score_edges = (member_mean - obs_variogram).square()

        if edge_valid is not None:
            score_edges = score_edges.masked_fill(~edge_valid, torch.nan)

        return self._aggregate_edges(score_edges, valid_edges=edge_valid)
