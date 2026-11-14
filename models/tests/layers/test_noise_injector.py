# (C) Copyright 2025-2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import pytest
import torch
from torch_geometric.data import HeteroData

from anemoi.models.layers.ensemble import NoiseConditioning


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


def _capture_mlp_input(injector: NoiseConditioning) -> list[torch.Tensor]:
    """Record the noise handed to the noise MLP, which is the drawn noise after any projection."""
    seen = []
    forward = injector.noise_mlp.forward

    def recording_forward(noise: torch.Tensor) -> torch.Tensor:
        seen.append(noise.detach().clone())
        return forward(noise)

    injector.noise_mlp.forward = recording_forward
    return seen


def test_noise_conditioning_draws_one_randn_field_per_call() -> None:
    """Without a noise matrix the noise is one draw of shape (batch, ensemble, grid, channels), times noise_std."""
    injector = NoiseConditioning(noise_std=2, noise_channels_dim=3, noise_mlp_hidden_dim=4, layer_kernels={})
    seen = _capture_mlp_input(injector)
    batch_size, ensemble_size, grid_size = 2, 3, 5

    torch.manual_seed(0)
    injector(
        x=torch.zeros(batch_size * ensemble_size * grid_size, 8),
        batch_size=batch_size,
        ensemble_size=ensemble_size,
        grid_size=grid_size,
        grid_shard_sizes=None,
    )
    torch.manual_seed(0)
    expected = torch.randn(batch_size, ensemble_size, grid_size, 3) * 2

    torch.testing.assert_close(seen[0], expected.reshape(-1, 3), rtol=0.0, atol=0.0)


def test_noise_conditioning_with_a_noise_graph_projects_one_randn_field() -> None:
    """With a noise graph the noise is one draw on the noise grid, projected to the hidden grid."""
    graph, noise_edges_name = _build_noise_graph()
    injector = NoiseConditioning(
        noise_std=1,
        noise_channels_dim=2,
        noise_mlp_hidden_dim=4,
        layer_kernels={},
        noise_edges_name=noise_edges_name,
        edge_weight_attribute="gauss_weight",
        graph_data=graph,
    )
    seen = _capture_mlp_input(injector)
    batch_size, ensemble_size, hidden_nodes = 2, 3, graph[noise_edges_name[2]].num_nodes
    noise_nodes = graph[noise_edges_name[0]].num_nodes

    torch.manual_seed(0)
    injector(
        x=torch.zeros(batch_size * ensemble_size * hidden_nodes, 8),
        batch_size=batch_size,
        ensemble_size=ensemble_size,
        grid_size=hidden_nodes,
        grid_shard_sizes=None,
    )
    torch.manual_seed(0)
    drawn = torch.randn(batch_size, ensemble_size, noise_nodes, 2).reshape(batch_size * ensemble_size, noise_nodes, 2)
    projected = injector._sparse_projector(drawn, injector.noise_graph_provider.get_edges(device=drawn.device))

    torch.testing.assert_close(seen[0], projected.reshape(-1, 2), rtol=0.0, atol=0.0)
