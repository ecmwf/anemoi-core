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
from anemoi.training.losses.graph_score_base import GraphScoreGraph


class GraphEnergyScoreLoss(BaseGraphScoreLoss):
    """Fair energy score over graph neighbourhoods.

    ``loss_graph.row_normalize`` controls whether each target node uses
    a weighted neighbourhood average or a raw weighted sum.

    When ``ignore_nans=True``, invalid source and destination nodes are dropped
    before graph aggregation. The remaining support is only re-normalized when
    ``row_normalize=True``; otherwise the valid nodes keep their original raw
    edge weights.
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
            Graph-based neighbourhood definition.
        graph_data : HeteroData | None
            Graph data used to build the neighbourhood graph.
        no_autocast : bool
            Whether to disable autocast for the full graph energy score
            calculation.
        ignore_nans : bool
            Whether to drop invalid nodes before graph aggregation. Remaining
            neighbourhood weights are only re-normalized when
            ``loss_graph.row_normalize=True``.
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

    def _aggregation_metadata(
        self,
        dtype: torch.dtype,
    ) -> tuple[int, torch.Tensor, torch.Tensor, torch.Tensor] | None:
        if self.graph is None:
            return None
        return self.graph.aggregation_metadata(dtype)

    def _stable_neighbourhood_norm(
        self,
        batch: torch.Tensor,
        aggregation_metadata: tuple[int, torch.Tensor, torch.Tensor, torch.Tensor] | None,
        node_valid: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if aggregation_metadata is None:
            out = torch.abs(batch)
            if node_valid is not None:
                out = out.masked_fill(~node_valid.unsqueeze(2), torch.nan)
            return out

        input_shape = batch.shape
        flat_batch = batch.reshape(-1, *input_shape[-2:])
        num_rows, row_index, col_index, weights = aggregation_metadata
        row_indices = row_index.view(1, -1, 1).expand(flat_batch.shape[0], -1, flat_batch.shape[-1])
        weight_view = weights.view(1, -1, 1)

        gathered_abs = torch.abs(flat_batch[:, col_index, :])
        support = torch.zeros(
            flat_batch.shape[0],
            num_rows,
            flat_batch.shape[-1],
            dtype=flat_batch.dtype,
            device=flat_batch.device,
        )
        valid_dest = None

        if node_valid is not None:
            expanded_valid = node_valid.unsqueeze(2).expand(
                *batch.shape[:-2],
                node_valid.shape[-2],
                node_valid.shape[-1],
            )
            flat_valid = expanded_valid.reshape(-1, node_valid.shape[-2], node_valid.shape[-1])
            valid_dest = flat_valid
            gathered_valid = flat_valid[:, col_index, :] & torch.isfinite(gathered_abs)
            safe_gathered_abs = torch.where(gathered_valid, gathered_abs, torch.zeros_like(gathered_abs))

            if self.row_normalize:
                weight_sum = torch.zeros_like(support)
                weight_sum.index_add_(1, row_index, weight_view * gathered_valid.to(dtype=flat_batch.dtype))
                gathered_weight_sum = weight_sum[:, row_index, :]
                safe_weight_sum = torch.where(
                    gathered_weight_sum > 0,
                    gathered_weight_sum,
                    torch.ones_like(gathered_weight_sum),
                )
                edge_weights = torch.where(
                    gathered_valid & (gathered_weight_sum > 0),
                    weight_view / safe_weight_sum,
                    torch.zeros_like(weight_view),
                )
            else:
                edge_weights = torch.where(gathered_valid, weight_view, torch.zeros_like(weight_view))
        else:
            safe_gathered_abs = gathered_abs
            edge_weights = weight_view

        active_edges = (edge_weights > 0).expand_as(safe_gathered_abs)
        support.zero_()
        support.index_add_(1, row_index, active_edges.to(dtype=flat_batch.dtype))
        active_gathered_abs = torch.where(active_edges, safe_gathered_abs, torch.zeros_like(safe_gathered_abs))

        row_max = torch.zeros_like(support)
        row_max.scatter_reduce_(1, row_indices, active_gathered_abs, reduce="amax", include_self=False)
        gathered_row_max = row_max[:, row_index, :]
        safe_row_max = torch.where(gathered_row_max > 0, gathered_row_max, torch.ones_like(gathered_row_max))
        scaled_abs = active_gathered_abs / safe_row_max
        scaled_abs = torch.where(gathered_row_max > 0, scaled_abs, torch.zeros_like(scaled_abs))

        norm_sq = torch.zeros_like(support)
        norm_sq.index_add_(1, row_index, edge_weights * scaled_abs.square())
        positive_norm = norm_sq > 0
        safe_norm_sq = torch.where(positive_norm, norm_sq, torch.ones_like(norm_sq))
        sqrt_norm_sq = torch.where(positive_norm, torch.sqrt(safe_norm_sq), torch.zeros_like(norm_sq))
        out = row_max * sqrt_norm_sq

        if valid_dest is not None:
            out = out.masked_fill((support <= 0) | ~valid_dest, torch.nan)

        return out.reshape(*input_shape[:-2] + out.shape[-2:])

    def _compute_local_score_tensor(
        self,
        y_pred_ens: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        ens_size = y_pred_ens.shape[2]
        aggregation_metadata = self._aggregation_metadata(y_pred_ens.dtype)

        node_valid = None
        if self.ignore_nans:
            node_valid = torch.isfinite(y) & torch.isfinite(y_pred_ens).all(dim=2)

        obs_distance = y_pred_ens - y.unsqueeze(2)
        obs_term = self._stable_neighbourhood_norm(obs_distance, aggregation_metadata, node_valid=node_valid).mean(
            dim=2,
        )

        pair_distance_sum = torch.zeros_like(obs_term)
        for i in range(ens_size):
            pair_distance = y_pred_ens[:, :, i].unsqueeze(2) - y_pred_ens[:, :, i + 1 :]
            if pair_distance.shape[2] == 0:
                continue
            pair_distance_sum = pair_distance_sum + self._stable_neighbourhood_norm(
                pair_distance,
                aggregation_metadata,
                node_valid=node_valid,
            ).sum(dim=2)

        coef = 1.0 / (ens_size * (ens_size - 1)) if self.fair else 1.0 / (ens_size**2)
        return obs_term - coef * pair_distance_sum
