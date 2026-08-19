# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Validated CSR graphs used by graph scores."""

import logging
from collections.abc import Mapping

import torch
from torch import nn
from torch_geometric.data import HeteroData

from anemoi.models.layers.graph_provider import ProjectionGraphProvider

LOGGER = logging.getLogger(__name__)


class GraphScoreGraph(nn.Module):
    """Own a validated sparse projection provider and graph metadata."""

    def __init__(
        self,
        graph_provider: ProjectionGraphProvider,
        *,
        num_src_nodes: int,
        num_dst_nodes: int,
        row_normalize: bool,
    ) -> None:
        super().__init__()
        self.graph_provider = graph_provider
        self.num_src_nodes = num_src_nodes
        self.num_dst_nodes = num_dst_nodes
        self.row_normalize = row_normalize

    @property
    def shape(self) -> tuple[int, int]:
        """Return the matrix shape as ``(destination nodes, source nodes)``."""
        return (self.num_dst_nodes, self.num_src_nodes)

    def get_matrix(self, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """Return a CSR matrix whose rows represent destinations and columns represent sources."""
        return self.graph_provider.get_edges(device=device, dtype=dtype)

    def get_edge_tensors(
        self,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return source indices, destination indices, and weights for the requested device and dtype."""
        matrix = self.get_matrix(device=device, dtype=dtype)
        return self._edge_tensors_from_matrix(matrix)

    @staticmethod
    def _edge_tensors_from_matrix(matrix: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Convert a CSR matrix into source indices, destination indices, and weights."""
        if matrix.layout != torch.sparse_csr:
            msg = f"Graph scores require a CSR matrix, got {matrix.layout}."
            raise TypeError(msg)
        row_counts = matrix.crow_indices()[1:] - matrix.crow_indices()[:-1]
        destinations = torch.arange(matrix.shape[0], device=matrix.device).repeat_interleave(row_counts)
        # CSR metadata tensors are views of the sparse matrix. Dense Dynamo kernels
        # require independent tensors; otherwise FakeTensor conversion attempts an
        # unsupported CSR view.
        return matrix.col_indices().clone(), destinations, matrix.values().clone()

    @classmethod
    def from_definition(
        cls,
        graph_definition: Mapping[str, object] | None,
        graph_data: HeteroData | None,
        *,
        graph_name: str,
        allow_none: bool = False,
    ) -> "GraphScoreGraph | None":
        """Validate a graph-score definition and build its CSR provider."""
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
        edges_name_value = graph_definition.get("edges_name")
        assert edges_name_value is not None, "Graph score definition must include 'edges_name'."
        edges_name = tuple(edges_name_value)
        if len(edges_name) != 3:
            msg = f"Graph score 'edges_name' must contain three entries, got {edges_name}."
            raise ValueError(msg)

        sub_graph = graph_data[edges_name]
        edge_index = sub_graph.edge_index.long()
        edge_weight_attribute = graph_definition.get("edge_weight_attribute")
        if edge_weight_attribute is None:
            edge_weights = torch.ones(edge_index.shape[1], dtype=torch.float32, device=edge_index.device)
        else:
            edge_weights = sub_graph[edge_weight_attribute].reshape(-1)

        src_node_weight_attribute = graph_definition.get("src_node_weight_attribute")
        if src_node_weight_attribute is not None:
            src_weights = graph_data[edges_name[0]][src_node_weight_attribute].reshape(-1)
            edge_weights = edge_weights * src_weights[edge_index[0]]

        num_src_nodes = graph_data[edges_name[0]].num_nodes
        num_dst_nodes = graph_data[edges_name[2]].num_nodes
        cls._validate_node_pairing(
            edges_name,
            num_src_nodes,
            num_dst_nodes,
        )
        cls._validate_weights(
            edge_index[1],
            edge_weights,
            num_dst_nodes,
            graph_name=graph_name,
        )

        row_normalize = bool(graph_definition.get("row_normalize", False))
        weights_for_validation = edge_weights
        if row_normalize:
            weights_for_validation = cls._row_normalize_weights(
                edge_index[1],
                edge_weights,
                num_dst_nodes,
            )
        cls._validate_row_sums(
            edge_index[1],
            weights_for_validation,
            num_dst_nodes,
            bool(graph_definition.get("validate_row_sums", False)),
            graph_name=graph_name,
        )

        graph_provider = ProjectionGraphProvider(
            graph=graph_data,
            edges_name=edges_name,
            edge_weight_attribute=edge_weight_attribute,
            src_node_weight_attribute=src_node_weight_attribute,
            row_normalize=row_normalize,
        )
        LOGGER.info(
            "%s: edges=%s nodes=%s",
            graph_name,
            graph_provider.projection_matrix.values().numel(),
            num_dst_nodes,
        )
        return cls(
            graph_provider=graph_provider,
            num_src_nodes=num_src_nodes,
            num_dst_nodes=num_dst_nodes,
            row_normalize=row_normalize,
        )

    @staticmethod
    def _validate_node_pairing(
        edges_name: tuple[str, ...],
        num_src_nodes: int,
        num_dst_nodes: int,
    ) -> None:
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
        totals.scatter_add_(0, row_index, weights)
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
        row_sums = torch.zeros(num_rows, dtype=weights.dtype, device=weights.device)
        row_sums.scatter_add_(0, row_index, weights)
        if not torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5):
            LOGGER.warning(
                "%s row weights do not sum to 1 (min=%.4f, max=%.4f, mean=%.4f). "
                "Consider using row_normalize=True or pre-normalized weights.",
                graph_name,
                row_sums.min().item(),
                row_sums.max().item(),
                row_sums.mean().item(),
            )
