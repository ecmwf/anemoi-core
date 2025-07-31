# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import os
from collections.abc import Iterable
from pathlib import Path

import scipy.sparse as sp
import torch

from anemoi.graphs.create import GraphCreator


class GraphExporter:
    """Class for exporting the graph to a sparse format."""

    def __init__(
        self,
        graph: str | Path,
        output_path: str | Path,
        edges_name: Iterable[tuple[str, str, str]] = None,
        edge_attribute_name: str = None,
        **kwargs,
    ):
        if isinstance(graph, Path) or isinstance(graph, str):
            if graph.endswith(".pt"):
                self.graph = torch.load(graph, weights_only=False, map_location="cpu")
            elif not graph.endswith(".yaml"):
                raise ValueError("The argument graph must be an actual graph (.pt) or a recipe to build one (.yaml).")

        self.graph = GraphCreator(graph).create(save_path=None).to("cpu")

        self.edges_name = self.graph.edge_types if edges_name is None else edges_name
        self.edge_attribute_name = edge_attribute_name
        self.output_path = output_path

    def get_edges_info(self, source_name: str, target_name: str) -> tuple[torch.Tensor, torch.Tensor]:
        """Get the edges info from the graph."""
        edge_index = self.graph[(source_name, "to", target_name)]["edge_index"]

        if self.edge_attribute_name is None:
            edge_attribute = torch.ones(edge_index.shape[1])
        else:
            assert (
                self.edge_attribute_name in self.graph[(source_name, "to", target_name)]
            ), f"Edge attribute {self.edge_attribute_name} not found in graph edges from {source_name} to {target_name}"
            edge_attribute = self.graph[(source_name, "to", target_name)][self.edge_attribute_name].squeeze()

        assert edge_attribute.ndim == 1, f"Edge attribute {self.edge_attribute_name} should be 1D."
        return edge_index, edge_attribute

    def get_nodes_info(self, source_name: str, target_name: str) -> tuple[int, int]:
        """Get the nodes info from the graph."""
        num_source_nodes = self.graph[source_name].num_nodes
        num_target_nodes = self.graph[target_name].num_nodes
        return num_source_nodes, num_target_nodes

    @staticmethod
    def get_sparse_matrix(edge_index, edge_attribute, num_source_nodes, num_target_nodes):
        # Create sparse matrix
        A = torch.sparse_coo_tensor(
            edge_index, edge_attribute, (num_source_nodes, num_target_nodes), device=edge_index.device
        )
        return A.coalesce()

    @staticmethod
    def convert_to_scipy_sparse(A):
        """Convert PyTorch sparse tensor to SciPy sparse matrix and save.

        Args:
            A: PyTorch sparse COO tensor
            filename: Output filename (.npz extension)
        """
        # Get indices and values from PyTorch sparse tensor
        indices = A.indices().cpu().numpy()
        values = A.values().cpu().numpy()
        shape = A.shape

        # Create SciPy sparse matrix (COO format)
        scipy_sparse = sp.coo_matrix((values, (indices[0], indices[1])), shape=shape)
        return scipy_sparse

    def export(self):
        """Export the graph to a sparse format."""
        os.makedirs(self.output_path, exist_ok=True)
        for source_nodes, _, target_nodes in self.edges_name:
            edge_index, edge_attribute = self.get_edges_info(source_nodes, target_nodes)
            num_source_nodes, num_target_nodes = self.get_nodes_info(source_nodes, target_nodes)
            A = GraphExporter.get_sparse_matrix(edge_index, edge_attribute, num_source_nodes, num_target_nodes)
            A = GraphExporter.convert_to_scipy_sparse(A)

            output_path = Path(self.output_path) / f"{self.edge_attribute_name}-{source_nodes}_to_{target_nodes}.npz"
            sp.save_npz(output_path, A)
