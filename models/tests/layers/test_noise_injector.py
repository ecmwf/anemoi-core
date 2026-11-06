# (C) Copyright 2025-2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from pathlib import Path

import pytest
import torch
from scipy.sparse import csr_matrix
from scipy.sparse import save_npz
from torch_geometric.data import HeteroData

from anemoi.models.layers.ensemble import NoiseConditioning
from anemoi.models.layers.ensemble import NoiseInjector


def _build_noise_graph(
    source_node_name: str = "src_noise",
    target_node_name: str = "dst_hidden",
) -> tuple[HeteroData, tuple[str, str, str]]:
    graph = HeteroData()
    graph[source_node_name].num_nodes = 2
    graph[target_node_name].num_nodes = 4

    edge_index = torch.tensor([[0, 1, 0, 1, 0, 1, 0, 1], [0, 0, 1, 1, 2, 2, 3, 3]])
    edge_weight = torch.tensor([0.3, 0.7, 0.3, 0.7, 0.3, 0.7, 0.3, 0.7])

    noise_edges_name = (source_node_name, "to", target_node_name)
    graph[noise_edges_name].edge_index = edge_index
    graph[noise_edges_name].gauss_weight = edge_weight
    return graph, noise_edges_name


def test_noise_conditioning_graph_projection_shape() -> None:
    graph, noise_edges_name = _build_noise_graph()
    injector = NoiseConditioning(
        noise_std=1,
        noise_channels_dim=2,
        noise_mlp_hidden_dim=4,
        layer_kernels={},
        noise_edges_name=noise_edges_name,
        edge_weight_attribute="gauss_weight",
        row_normalize_noise_matrix=False,
        graph_data=graph,
        sparse_projector_backend="torch",
    )

    batch_size = 2
    ensemble_size = 3
    hidden_nodes = graph[noise_edges_name[2]].num_nodes
    x = torch.zeros((batch_size * ensemble_size * hidden_nodes, 8))

    _, noise = injector(
        x=x,
        batch_size=batch_size,
        ensemble_size=ensemble_size,
        grid_size=hidden_nodes,
        grid_shard_sizes=[],
    )

    assert noise is not None
    assert noise.shape == (batch_size * ensemble_size * hidden_nodes, injector.noise_channels)


def test_noise_conditioning_rejects_mixed_sources() -> None:
    graph, noise_edges_name = _build_noise_graph()
    with pytest.raises(AssertionError, match="noise_matrix or noise_edges_name"):
        NoiseConditioning(
            noise_std=1,
            noise_channels_dim=2,
            noise_mlp_hidden_dim=4,
            layer_kernels={},
            noise_matrix="dummy.npz",
            noise_edges_name=noise_edges_name,
            edge_weight_attribute="gauss_weight",
            graph_data=graph,
        )


def test_noise_conditioning_requires_graph_data() -> None:
    _, noise_edges_name = _build_noise_graph()
    with pytest.raises(AssertionError, match="graph_data must be provided"):
        NoiseConditioning(
            noise_std=1,
            noise_channels_dim=2,
            noise_mlp_hidden_dim=4,
            layer_kernels={},
            noise_edges_name=noise_edges_name,
            edge_weight_attribute="gauss_weight",
            graph_data=None,
        )


@pytest.mark.parametrize("backend", ["triton", "torch"])
def test_noise_conditioning_passes_its_backend_to_the_projection(backend: str) -> None:
    graph, noise_edges_name = _build_noise_graph()
    injector = NoiseConditioning(
        noise_std=1,
        noise_channels_dim=2,
        noise_mlp_hidden_dim=4,
        layer_kernels={},
        noise_edges_name=noise_edges_name,
        edge_weight_attribute="gauss_weight",
        row_normalize_noise_matrix=False,
        graph_data=graph,
        sparse_projector_backend=backend,
    )

    assert injector.noise_graph_provider.backend == backend


@pytest.mark.parametrize("backend", ["triton", "torch"])
def test_noise_injector_passes_the_sparse_projector_settings_on(tmp_path: Path, backend: str) -> None:
    noise_matrix = tmp_path / "noise.npz"
    save_npz(noise_matrix, csr_matrix([[0.3, 0.7], [0.3, 0.7], [0.3, 0.7], [0.3, 0.7]]))
    # The ensemble model passes these settings to every noise injector it builds.
    injector = NoiseInjector(
        noise_std=1,
        noise_channels_dim=2,
        noise_mlp_hidden_dim=4,
        num_channels=8,
        layer_kernels={},
        noise_matrix=str(noise_matrix),
        sparse_projector_num_chunks=2,
        sparse_projector_backend=backend,
    )

    assert injector._noise_conditioning.noise_graph_provider.backend == backend
    assert injector._noise_conditioning._sparse_projector.num_chunks == 2
