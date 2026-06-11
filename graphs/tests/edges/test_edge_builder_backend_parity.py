# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Parity tests: torch-cluster (pyg) vs scikit-learn (sklearn) edge builders.

These tests verify that the two code paths produce identical edge sets.
They are skipped automatically when torch-cluster / pyg-lib is not installed.
"""

import numpy as np
import pytest
import torch
from torch_geometric.data import HeteroData

from anemoi.graphs.edges.builders.cutoff import CutOffEdges
from anemoi.graphs.edges.builders.cutoff import ReversedCutOffEdges
from anemoi.graphs.edges.builders.knn import KNNEdges
from anemoi.graphs.edges.builders.knn import ReversedKNNEdges
from anemoi.graphs.utils import TORCH_CLUSTER_AVAILABLE

pytestmark = pytest.mark.skipif(
    not TORCH_CLUSTER_AVAILABLE,
    reason="torch-cluster / pyg-lib not installed; skipping parity tests",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _edge_set_from_index(edge_index: torch.Tensor) -> frozenset:
    """Convert a (2, E) edge-index tensor to a frozenset of (src, tgt) tuples."""
    arr = edge_index.cpu().numpy()
    return frozenset(zip(arr[0].tolist(), arr[1].tolist()))


def _edge_set_from_adj(adj_matrix) -> frozenset:
    """Convert a scipy coo_matrix (row=target, col=source) to a frozenset of (src, tgt) tuples."""
    return frozenset(zip(adj_matrix.col.tolist(), adj_matrix.row.tolist()))


# ---------------------------------------------------------------------------
# KNN parity
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("edge_builder_cls", [KNNEdges, ReversedKNNEdges])
@pytest.mark.parametrize("k", [1, 2, 3])
def test_knn_pyg_vs_sklearn(edge_builder_cls, k, graph_with_nodes: HeteroData):
    """KNN edges from pyg and sklearn must be identical sets."""
    builder = edge_builder_cls("test_nodes", "test_nodes", num_nearest_neighbours=k)
    source_nodes, target_nodes = builder.prepare_node_data(graph_with_nodes)
    src_coords, tgt_coords = builder.get_cartesian_node_coordinates(source_nodes, target_nodes)

    pyg_index = builder._compute_edge_index_pyg(src_coords, tgt_coords)
    sklearn_adj = builder._compute_adj_matrix_sklearn(src_coords, tgt_coords)

    pyg_set = _edge_set_from_index(pyg_index)
    sklearn_set = _edge_set_from_adj(sklearn_adj)

    assert pyg_set == sklearn_set, (
        f"{edge_builder_cls.__name__}(k={k}): pyg produced {len(pyg_set)} edges, "
        f"sklearn produced {len(sklearn_set)} edges; "
        f"symmetric diff = {pyg_set.symmetric_difference(sklearn_set)}"
    )


@pytest.mark.parametrize("edge_builder_cls", [KNNEdges, ReversedKNNEdges])
def test_knn_edge_count(edge_builder_cls, graph_with_nodes: HeteroData):
    """Each target node must have exactly k edges in both backends."""
    k = 3
    builder = edge_builder_cls("test_nodes", "test_nodes", num_nearest_neighbours=k)
    source_nodes, target_nodes = builder.prepare_node_data(graph_with_nodes)
    src_coords, tgt_coords = builder.get_cartesian_node_coordinates(source_nodes, target_nodes)

    pyg_index = builder._compute_edge_index_pyg(src_coords, tgt_coords)
    sklearn_adj = builder._compute_adj_matrix_sklearn(src_coords, tgt_coords)

    n_target = tgt_coords.shape[0]

    # pyg: edge_index[1] contains target node indices
    pyg_tgt_counts = np.bincount(pyg_index[1].cpu().numpy(), minlength=n_target)
    # sklearn: adj_matrix.row contains target node indices
    sklearn_tgt_counts = np.bincount(sklearn_adj.row, minlength=n_target)

    np.testing.assert_array_equal(
        pyg_tgt_counts,
        sklearn_tgt_counts,
        err_msg=f"{edge_builder_cls.__name__}: per-target-node edge counts differ",
    )


# ---------------------------------------------------------------------------
# CutOff parity
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("edge_builder_cls", [CutOffEdges, ReversedCutOffEdges])
@pytest.mark.parametrize("cutoff_factor", [0.5, 1.0])
def test_cutoff_pyg_vs_sklearn(edge_builder_cls, cutoff_factor, graph_with_nodes: HeteroData):
    """CutOff edges from pyg and sklearn must be identical sets."""
    # max_num_neighbours set high to avoid backend-specific truncation differences
    builder = edge_builder_cls(
        "test_nodes", "test_nodes", cutoff_factor=cutoff_factor, max_num_neighbours=512
    )
    source_nodes, target_nodes = builder.prepare_node_data(graph_with_nodes)
    src_coords, tgt_coords = builder.get_cartesian_node_coordinates(source_nodes, target_nodes)

    pyg_index = builder._compute_edge_index_pyg(src_coords, tgt_coords)
    sklearn_adj = builder._compute_adj_matrix_sklearn(src_coords, tgt_coords)

    pyg_set = _edge_set_from_index(pyg_index)
    sklearn_set = _edge_set_from_adj(sklearn_adj)

    assert pyg_set == sklearn_set, (
        f"{edge_builder_cls.__name__}(factor={cutoff_factor}): "
        f"pyg produced {len(pyg_set)} edges, sklearn produced {len(sklearn_set)} edges; "
        f"symmetric diff = {pyg_set.symmetric_difference(sklearn_set)}"
    )


@pytest.mark.parametrize("edge_builder_cls", [CutOffEdges, ReversedCutOffEdges])
def test_cutoff_full_pipeline_agrees(edge_builder_cls, graph_with_nodes: HeteroData):
    """Full compute_edge_index with pyg backend and via sklearn conversion produce same edges.

    Patches the TORCH_CLUSTER_AVAILABLE flag to force both paths on the same graph.
    """
    import anemoi.graphs.edges.builders.base as _base_module
    import anemoi.graphs.utils as _utils_module

    builder = edge_builder_cls("test_nodes", "test_nodes", cutoff_factor=0.5, max_num_neighbours=512)
    source_nodes, target_nodes = builder.prepare_node_data(graph_with_nodes)

    # --- pyg path ---
    _base_module.TORCH_CLUSTER_AVAILABLE = True
    _utils_module.TORCH_CLUSTER_AVAILABLE = True
    pyg_edge_index = builder.compute_edge_index(
         source_nodes, target_nodes
    )

    # --- sklearn path ---
    _base_module.TORCH_CLUSTER_AVAILABLE = False
    _utils_module.TORCH_CLUSTER_AVAILABLE = False
    sklearn_edge_index = builder.compute_edge_index(
         source_nodes, target_nodes
    )

    pyg_set = _edge_set_from_index(pyg_edge_index)
    sklearn_set = _edge_set_from_index(sklearn_edge_index)

    assert pyg_set == sklearn_set, (
        f"{edge_builder_cls.__name__} full pipeline: "
        f"symmetric diff = {pyg_set.symmetric_difference(sklearn_set)}"
    )
