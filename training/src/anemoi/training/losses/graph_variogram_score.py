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

    For each edge ``src -> dst``, the variogram is
    ``|x[src] - x[dst]|**p``. The score compares the ensemble variograms with
    the observed variogram. With ``fair=True``, products use two different
    ensemble members and are divided by ``M (M - 1)``.

    ``loss_graph.row_normalize`` controls whether each destination node uses a
    weighted average or a raw weighted sum over incoming edges.

    When ``ignore_nans=True``, edges containing missing values are left out. If
    ``row_normalize=True``, the weights of the remaining edges are scaled to
    sum to one; otherwise their original weights are kept.
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
            Whether to keep the original numerical precision throughout.
        ignore_nans : bool
            Whether to leave out edges containing missing values. With
            ``loss_graph.row_normalize=True``, the remaining weights are
            scaled to sum to one.
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

    def _edge_variogram(
        self,
        x: torch.Tensor,
        valid_edges: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return ``|x[src] - x[dst]|**p`` with shape ``(..., E, V)``."""
        edge_difference = self._edge_difference(x)
        if valid_edges is not None:
            # Set missing edge differences to zero before raising them to the
            # variogram power.
            edge_difference = torch.where(
                valid_edges,
                edge_difference,
                torch.zeros_like(edge_difference),
            )
        return torch.abs(edge_difference).pow(self.p)

    def _compute_local_score_tensor(
        self,
        y_pred_ens: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        """Return one variogram score per batch, output step, node, and variable."""
        ensemble_size = y_pred_ens.shape[2]
        edge_valid = self._valid_edges(y_pred_ens, y)

        obs_variogram = self._edge_variogram(y, edge_valid)

        member_sum = torch.zeros_like(obs_variogram)
        if self.fair:
            member_cross_sum = torch.zeros_like(obs_variogram)
            running_sum = torch.zeros_like(obs_variogram)

        # Before member i, running_sum equals sum(v_j, j < i). Their product
        # therefore adds every unordered term v_i * v_j exactly once.
        for i in range(ensemble_size):
            member_variogram = self._edge_variogram(y_pred_ens[:, :, i], edge_valid)
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
