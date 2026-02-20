# (C) Copyright 2025 Anemoi contributors.
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
from anemoi.models.layers.ensemble import NoiseInjector


def _build_noise_graph() -> HeteroData:
    graph = HeteroData()
    graph["noise"].num_nodes = 2
    graph["hidden"].num_nodes = 4

    edge_index = torch.tensor([[0, 1, 0, 1, 0, 1, 0, 1], [0, 0, 1, 1, 2, 2, 3, 3]])
    edge_weight = torch.tensor([0.3, 0.7, 0.3, 0.7, 0.3, 0.7, 0.3, 0.7])

    graph[("noise", "to", "hidden")].edge_index = edge_index
    graph[("noise", "to", "hidden")].gauss_weight = edge_weight
    return graph


def test_noise_conditioning_graph_projection_shape() -> None:
    graph = _build_noise_graph()
    injector = NoiseConditioning(
        noise_std=1,
        noise_channels_dim=2,
        noise_mlp_hidden_dim=4,
        layer_kernels={},
        noise_edges_name=("noise", "to", "hidden"),
        edge_weight_attribute="gauss_weight",
        row_normalize_noise_matrix=False,
        graph_data=graph,
    )

    batch_size = 2
    ensemble_size = 3
    hidden_nodes = graph["hidden"].num_nodes
    x = torch.zeros((batch_size * ensemble_size * hidden_nodes, 8))

    _, noise = injector(
        x=x,
        batch_size=batch_size,
        ensemble_size=ensemble_size,
        grid_size=hidden_nodes,
        shard_shapes_ref=[],
    )

    assert noise is not None
    assert noise.shape == (batch_size * ensemble_size * hidden_nodes, injector.noise_channels)


def test_noise_conditioning_rejects_mixed_sources() -> None:
    graph = _build_noise_graph()
    with pytest.raises(AssertionError, match="noise_matrix or noise_edges_name"):
        NoiseConditioning(
            noise_std=1,
            noise_channels_dim=2,
            noise_mlp_hidden_dim=4,
            layer_kernels={},
            noise_matrix="dummy.npz",
            noise_edges_name=("noise", "to", "hidden"),
            edge_weight_attribute="gauss_weight",
            graph_data=graph,
        )


def test_noise_conditioning_requires_graph_data() -> None:
    with pytest.raises(AssertionError, match="graph_data must be provided"):
        NoiseConditioning(
            noise_std=1,
            noise_channels_dim=2,
            noise_mlp_hidden_dim=4,
            layer_kernels={},
            noise_edges_name=("noise", "to", "hidden"),
            edge_weight_attribute="gauss_weight",
            graph_data=None,
        )


def test_noise_injector_accepts_graph_data() -> None:
    injector = NoiseInjector(
        noise_std=1,
        noise_channels_dim=2,
        noise_mlp_hidden_dim=4,
        num_channels=8,
        layer_kernels={},
        graph_data=HeteroData(),
    )

    batch_size = 1
    ensemble_size = 2
    grid_size = 3
    x = torch.zeros((batch_size * ensemble_size * grid_size, 8))

    out, noise = injector(
        x=x,
        batch_size=batch_size,
        ensemble_size=ensemble_size,
        grid_size=grid_size,
        shard_shapes_ref=[],
    )

    assert noise is None
    assert out.shape == x.shape
