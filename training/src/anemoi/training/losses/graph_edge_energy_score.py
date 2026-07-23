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


class GraphEdgeEnergyScoreLoss(BaseGraphScoreLoss):
    """Fair energy score over signed graph edge-difference neighbourhoods.

    For each graph edge ``src -> dst`` this loss forms signed differences
    ``x[src] - x[dst]``. For each destination node, the incoming edge
    differences are scored jointly with an energy score. The resulting score is
    node-shaped, so it remains compatible with the usual scalers, reductions,
    sharding path, and multiscale wrapper.

    The edge-difference vector norm mirrors ``GraphEnergyScoreLoss``: when
    ``loss_graph.row_normalize=True`` it is a weighted RMS-style norm over
    incoming edges, so neighbourhood size does not change the scale by itself.

    When ``ignore_nans=True``, invalid nodes remove every edge that touches
    them. The remaining valid edges are only re-normalized when
    ``row_normalize=True``; otherwise they keep their original raw edge
    weights.
    """

    def __init__(
        self,
        fair: bool = True,
        loss_graph: dict | None = None,
        graph_data: HeteroData | None = None,
        no_autocast: bool = True,
        ignore_nans: bool = False,
    ) -> None:
        """Graph edge-difference energy score.

        Parameters
        ----------
        fair : bool
            Whether to use the fair ensemble correction.
        loss_graph : dict | None
            Graph-based edge definition.
        graph_data : HeteroData | None
            Graph data used to build the edge graph.
        no_autocast : bool
            Whether to disable autocast for the full edge energy score
            calculation.
        ignore_nans : bool
            Whether to drop invalid nodes before graph aggregation. Remaining
            edge weights are only re-normalized when
            ``loss_graph.row_normalize=True``.
        """
        graph = GraphScoreGraph.from_definition(
            loss_graph,
            graph_data,
            graph_name="Graph edge energy score neighbourhood",
            allow_none=False,
            require_square=True,
        )
        super().__init__(graph=graph, no_autocast=no_autocast, ignore_nans=ignore_nans)
        self.fair = fair

    @property
    def edge_src_index(self) -> torch.Tensor:
        return self.graph.edge_src_index

    @property
    def edge_dst_index(self) -> torch.Tensor:
        return self.graph.edge_dst_index

    @property
    def edge_weights(self) -> torch.Tensor:
        return self.graph.edge_weights

    @property
    def num_nodes(self) -> int:
        return self.graph.num_nodes

    @property
    def name(self) -> str:
        prefix = "f" if self.fair else ""
        return f"{prefix}graph_edge_energy_score"

    def _edge_difference(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, :, self.edge_src_index] - x[:, :, self.edge_dst_index]

    def _stable_edge_neighbourhood_norm(
        self,
        edge_values: torch.Tensor,
        valid_edges: torch.Tensor | None = None,
    ) -> torch.Tensor:
        input_shape = edge_values.shape
        flat_edge_values = edge_values.reshape(-1, *input_shape[-2:])
        row_index = self.edge_dst_index
        weights = self.edge_weights.to(dtype=edge_values.dtype)
        row_indices = row_index.view(1, -1, 1).expand(flat_edge_values.shape[0], -1, flat_edge_values.shape[-1])
        weight_view = weights.view(1, -1, 1)

        gathered_abs = torch.abs(flat_edge_values)
        support = torch.zeros(
            flat_edge_values.shape[0],
            self.num_nodes,
            flat_edge_values.shape[-1],
            dtype=flat_edge_values.dtype,
            device=flat_edge_values.device,
        )

        if valid_edges is not None:
            extra_prefix_dims = edge_values.ndim - valid_edges.ndim
            expanded_valid = valid_edges.reshape(
                *valid_edges.shape[:-2],
                *((1,) * extra_prefix_dims),
                *valid_edges.shape[-2:],
            ).expand(input_shape)
            flat_valid = expanded_valid.reshape(-1, *valid_edges.shape[-2:])
            safe_gathered_abs = torch.where(flat_valid, gathered_abs, torch.zeros_like(gathered_abs))

            if self.row_normalize:
                weight_sum = torch.zeros_like(support)
                weight_sum.index_add_(1, row_index, weight_view * flat_valid.to(dtype=flat_edge_values.dtype))
                gathered_weight_sum = weight_sum[:, row_index, :]
                safe_weight_sum = torch.where(
                    gathered_weight_sum > 0,
                    gathered_weight_sum,
                    torch.ones_like(gathered_weight_sum),
                )
                edge_weights = torch.where(
                    flat_valid & (gathered_weight_sum > 0),
                    weight_view / safe_weight_sum,
                    torch.zeros_like(weight_view),
                )
            else:
                edge_weights = torch.where(flat_valid, weight_view, torch.zeros_like(weight_view))
        else:
            safe_gathered_abs = gathered_abs
            edge_weights = weight_view

        active_edges = (edge_weights > 0).expand_as(safe_gathered_abs)
        support.zero_()
        support.index_add_(1, row_index, active_edges.to(dtype=flat_edge_values.dtype))
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

        if valid_edges is not None:
            out = out.masked_fill(support <= 0, torch.nan)

        return out.reshape(*input_shape[:-2] + out.shape[-2:])

    def _compute_local_score_tensor(
        self,
        y_pred_ens: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        ens_size = y_pred_ens.shape[2]
        edge_valid = None
        if self.ignore_nans:
            node_valid = torch.isfinite(y) & torch.isfinite(y_pred_ens).all(dim=2)
            edge_valid = node_valid[:, :, self.edge_src_index, :] & node_valid[:, :, self.edge_dst_index, :]

        obs_edge_difference = self._edge_difference(y)

        obs_term_sum = torch.zeros(
            (*y.shape[:2], self.num_nodes, y.shape[-1]),
            dtype=y_pred_ens.dtype,
            device=y_pred_ens.device,
        )
        for i in range(ens_size):
            member_edge_difference = self._edge_difference(y_pred_ens[:, :, i])
            obs_term_sum = obs_term_sum + self._stable_edge_neighbourhood_norm(
                member_edge_difference - obs_edge_difference,
                valid_edges=edge_valid,
            )
        obs_term = obs_term_sum / ens_size

        pair_distance_sum = torch.zeros_like(obs_term)
        for i in range(ens_size):
            member_edge_difference = self._edge_difference(y_pred_ens[:, :, i])
            for j in range(i + 1, ens_size):
                pair_edge_difference = self._edge_difference(y_pred_ens[:, :, j])
                pair_distance_sum = pair_distance_sum + self._stable_edge_neighbourhood_norm(
                    member_edge_difference - pair_edge_difference,
                    valid_edges=edge_valid,
                )

        coef = 1.0 / (ens_size * (ens_size - 1)) if self.fair else 1.0 / (ens_size**2)
        return obs_term - coef * pair_distance_sum
