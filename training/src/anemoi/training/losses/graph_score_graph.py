# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Graph edges, weights, and neighbourhood distances.

Shapes use ``N`` for nodes, ``E`` for edges, and ``V`` for variables. Any
leading dimensions are left unchanged.
"""

import logging
from collections.abc import Mapping

import torch
from torch import nn
from torch_geometric.data import HeteroData

LOGGER = logging.getLogger(__name__)


class GraphScoreGraph(nn.Module):
    """Hold graph edges and weights and calculate neighbourhood distances."""

    def __init__(
        self,
        edge_index: torch.Tensor,
        edge_weights: torch.Tensor,
        *,
        num_src_nodes: int,
        num_dst_nodes: int,
        row_normalize: bool,
    ) -> None:
        super().__init__()
        self.register_buffer("edge_index", edge_index.long(), persistent=False)
        self.register_buffer("edge_src_index", edge_index[0].long(), persistent=False)
        self.register_buffer("edge_dst_index", edge_index[1].long(), persistent=False)
        self.register_buffer("edge_weights", edge_weights, persistent=False)
        self.num_src_nodes = num_src_nodes
        self.num_dst_nodes = num_dst_nodes
        self.row_normalize = row_normalize

    @property
    def shape(self) -> tuple[int, int]:
        """Return the graph shape as ``(destination nodes, source nodes)``."""
        return (self.num_dst_nodes, self.num_src_nodes)

    @property
    def num_nodes(self) -> int:
        """Return the number of destination nodes."""
        return self.num_dst_nodes

    def weighted_row_l2_norm(
        self,
        edge_values: torch.Tensor,
        valid_edges: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute ``sqrt(sum_e w_e x_e**2)`` for edges entering each node.

        Parameters
        ----------
        edge_values : torch.Tensor
            Edge values with shape ``(..., E, V)``.
        valid_edges : torch.Tensor | None
            ``True`` where an edge value is available. It has the same shape
            as ``edge_values``. Missing edge values are left out.

        Returns
        -------
        torch.Tensor
            One value per destination node with shape ``(..., N, V)``. A node
            with no available incoming edges has value ``NaN``.

        Notes
        -----
        Values are divided by the largest incoming magnitude before squaring
        and multiplied by it again afterwards. This leaves the distance
        unchanged while keeping the intermediate squares representable.
        """
        input_shape = edge_values.shape
        flat_edge_values = edge_values.reshape(-1, *input_shape[-2:])
        row_index = self.edge_dst_index
        weights = self.edge_weights.to(dtype=edge_values.dtype)
        row_indices = row_index.view(1, -1, 1).expand(
            flat_edge_values.shape[0],
            -1,
            flat_edge_values.shape[-1],
        )
        weight_view = weights.view(1, -1, 1)

        gathered_abs = torch.abs(flat_edge_values)
        active_edge_count = torch.zeros(
            flat_edge_values.shape[0],
            self.num_nodes,
            flat_edge_values.shape[-1],
            dtype=flat_edge_values.dtype,
            device=flat_edge_values.device,
        )

        # Edges with missing values are left out. If requested, the remaining
        # weights are scaled to sum to one.
        if valid_edges is not None:
            if valid_edges.shape != edge_values.shape:
                msg = (
                    f"valid_edges shape {tuple(valid_edges.shape)} must match "
                    f"edge_values shape {tuple(edge_values.shape)}."
                )
                raise ValueError(msg)
            flat_valid = valid_edges.reshape(-1, *input_shape[-2:])
            safe_gathered_abs = torch.where(
                flat_valid,
                gathered_abs,
                torch.zeros_like(gathered_abs),
            )

            if self.row_normalize:
                valid_row_weight_sum = torch.zeros_like(active_edge_count)
                valid_row_weight_sum.index_add_(
                    1,
                    row_index,
                    weight_view * flat_valid.to(dtype=flat_edge_values.dtype),
                )
                gathered_weight_sum = valid_row_weight_sum[:, row_index, :]
                safe_weight_sum = torch.where(
                    gathered_weight_sum > 0,
                    gathered_weight_sum,
                    torch.ones_like(gathered_weight_sum),
                )
                effective_weights = torch.where(
                    flat_valid & (gathered_weight_sum > 0),
                    weight_view / safe_weight_sum,
                    torch.zeros_like(weight_view),
                )
            else:
                effective_weights = torch.where(
                    flat_valid,
                    weight_view,
                    torch.zeros_like(weight_view),
                )
        else:
            safe_gathered_abs = gathered_abs
            effective_weights = weight_view

        # Edges with zero weight are left out of both the maximum and the sum.
        active_edges = (effective_weights > 0).expand_as(safe_gathered_abs)
        active_edge_count.index_add_(
            1,
            row_index,
            active_edges.to(dtype=flat_edge_values.dtype),
        )
        active_gathered_abs = torch.where(
            active_edges,
            safe_gathered_abs,
            torch.zeros_like(safe_gathered_abs),
        )

        row_max = torch.zeros_like(active_edge_count)
        row_max.scatter_reduce_(
            1,
            row_indices,
            active_gathered_abs,
            reduce="amax",
            include_self=False,
        )
        gathered_row_max = row_max[:, row_index, :]
        safe_row_max = torch.where(
            gathered_row_max > 0,
            gathered_row_max,
            torch.ones_like(gathered_row_max),
        )
        scaled_abs = active_gathered_abs / safe_row_max
        scaled_abs = torch.where(
            gathered_row_max > 0,
            scaled_abs,
            torch.zeros_like(scaled_abs),
        )

        norm_sq = torch.zeros_like(active_edge_count)
        norm_sq.index_add_(1, row_index, effective_weights * scaled_abs.square())
        positive_norm = norm_sq > 0
        # For q = 0, use sqrt(q) = 0 directly because the derivative of the
        # square root is singular at the origin.
        safe_norm_sq = torch.where(positive_norm, norm_sq, torch.ones_like(norm_sq))
        sqrt_norm_sq = torch.where(
            positive_norm,
            torch.sqrt(safe_norm_sq),
            torch.zeros_like(norm_sq),
        )
        row_norms = row_max * sqrt_norm_sq

        if valid_edges is not None:
            row_norms = row_norms.masked_fill(active_edge_count <= 0, torch.nan)

        return row_norms.reshape(*input_shape[:-2] + row_norms.shape[-2:])

    @classmethod
    def from_definition(
        cls,
        graph_definition: Mapping[str, object] | None,
        graph_data: HeteroData | None,
        *,
        graph_name: str,
        allow_none: bool = False,
        require_square: bool = True,
    ) -> "GraphScoreGraph | None":
        """Read the graph edges and their weights."""
        if graph_definition is None:
            if allow_none:
                LOGGER.info("%s: %s", graph_name, None)
                return None
            error_msg = f"{graph_name} must be provided."
            raise AssertionError(error_msg)

        if not isinstance(graph_definition, Mapping):
            msg = f"{graph_name} must be a mapping or None, got {type(graph_definition).__name__}."
            raise TypeError(msg)

        assert graph_data is not None, "graph_data must be provided when using a graph score loss graph."

        # Each edge weight is multiplied by its source-node weight when both
        # are provided.
        edges_name = graph_definition.get("edges_name")
        assert edges_name is not None, "Graph score definition must include 'edges_name'."
        edges_name = tuple(edges_name)

        sub_graph = graph_data[edges_name]
        edge_index = sub_graph.edge_index.long()

        edge_weight_attribute = graph_definition.get("edge_weight_attribute")
        if edge_weight_attribute is not None:
            edge_weights = sub_graph[edge_weight_attribute].reshape(-1)
        else:
            edge_weights = torch.ones(
                edge_index.shape[1],
                dtype=torch.float32,
                device=edge_index.device,
            )

        src_node_weight_attribute = graph_definition.get("src_node_weight_attribute")
        if src_node_weight_attribute is not None:
            src_weights = graph_data[edges_name[0]][src_node_weight_attribute].reshape(
                -1,
            )
            edge_weights = edge_weights * src_weights[edge_index[0]]

        # Source and destination nodes must describe the same forecast grid.
        num_src_nodes = graph_data[edges_name[0]].num_nodes
        num_dst_nodes = graph_data[edges_name[2]].num_nodes
        cls._validate_node_space(
            edges_name,
            num_src_nodes,
            num_dst_nodes,
            require_square=require_square,
        )

        cls._validate_weights(
            edge_index[1].long(),
            edge_weights,
            num_dst_nodes,
            graph_name=graph_name,
        )

        row_normalize = bool(graph_definition.get("row_normalize", False))
        if row_normalize:
            edge_weights = cls._row_normalize_weights(
                edge_index[1].long(),
                edge_weights,
                num_dst_nodes,
            )

        cls._validate_row_sums(
            edge_index[1].long(),
            edge_weights,
            num_dst_nodes,
            graph_definition.get("validate_row_sums", True),
            graph_name=graph_name,
        )

        if require_square and num_src_nodes == num_dst_nodes:
            LOGGER.info(
                "%s: edges=%s nodes=%s",
                graph_name,
                edge_index.shape[1],
                num_dst_nodes,
            )
        else:
            LOGGER.info(
                "%s: edges=%s src_nodes=%s dst_nodes=%s",
                graph_name,
                edge_index.shape[1],
                num_src_nodes,
                num_dst_nodes,
            )

        return cls(
            edge_index=edge_index,
            edge_weights=edge_weights,
            num_src_nodes=num_src_nodes,
            num_dst_nodes=num_dst_nodes,
            row_normalize=row_normalize,
        )

    @staticmethod
    def _validate_node_space(
        edges_name: tuple[str, ...],
        num_src_nodes: int,
        num_dst_nodes: int,
        *,
        require_square: bool,
    ) -> None:
        if not require_square:
            return
        if edges_name[0] != edges_name[2]:
            msg = (
                "Graph score losses require source and destination nodes to use the same node type, "
                f"got {edges_name[0]!r} and {edges_name[2]!r}."
            )
            raise ValueError(msg)
        if num_src_nodes != num_dst_nodes:
            msg = (
                "Graph score losses require a grid-preserving loss graph with the same number "
                "of source and target nodes."
            )
            raise ValueError(msg)

    @staticmethod
    def _row_normalize_weights(
        row_index: torch.Tensor,
        weights: torch.Tensor,
        num_rows: int,
    ) -> torch.Tensor:
        totals = torch.zeros(num_rows, dtype=weights.dtype, device=weights.device)
        totals = totals.scatter_add_(0, row_index, weights)
        return weights / totals[row_index]

    @staticmethod
    def _validate_weights(
        row_index: torch.Tensor,
        weights: torch.Tensor,
        num_rows: int,
        *,
        graph_name: str,
    ) -> None:
        if weights.numel() != row_index.numel():
            msg = (
                f"{graph_name} must provide exactly one scalar weight per edge, "
                f"got {weights.numel()} weights for {row_index.numel()} edges."
            )
            raise ValueError(msg)
        if torch.is_complex(weights) or not torch.isfinite(weights).all():
            msg = f"{graph_name} weights must be finite real values."
            raise ValueError(msg)
        if torch.any(weights < 0):
            msg = f"{graph_name} weights must be non-negative."
            raise ValueError(msg)

        row_totals = torch.zeros(num_rows, dtype=weights.dtype, device=weights.device)
        row_totals.scatter_add_(0, row_index, weights)
        zero_weight_rows = torch.count_nonzero(row_totals <= 0).item()
        if zero_weight_rows:
            msg = f"{graph_name} must have positive total weight for every node; found {zero_weight_rows} empty rows."
            raise ValueError(msg)

    @staticmethod
    def _validate_row_sums(
        row_index: torch.Tensor,
        weights: torch.Tensor,
        num_rows: int,
        validate_row_sums: bool,
        *,
        graph_name: str,
    ) -> None:
        if not validate_row_sums:
            return

        row_sums = torch.zeros(
            num_rows,
            dtype=weights.dtype,
            device=weights.device,
        ).scatter_add_(0, row_index, weights)
        if not torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5):
            LOGGER.warning(
                "%s row weights do not sum to 1 (min=%.4f, max=%.4f, mean=%.4f). "
                "Consider using row_normalize=True or pre-normalized weights.",
                graph_name,
                row_sums.min().item(),
                row_sums.max().item(),
                row_sums.mean().item(),
            )
