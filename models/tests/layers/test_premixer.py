# (C) Copyright 2026 Anemoi contributors.
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

from anemoi.models.distributed.shapes import GraphShardInfo
from anemoi.models.layers.graph_provider import create_graph_provider
from anemoi.models.layers.premixer import GraphTransformerPreMixer
from anemoi.models.layers.utils import load_layer_kernels


class TestGraphTransformerPreMixer:
    """Test the GraphTransformerPreMixer class."""

    NUM_NODES: int = 64
    IN_CHANNELS: int = 12
    NUM_CHANNELS: int = 32
    K: int = 4

    @pytest.fixture
    def fake_graph(self) -> HeteroData:
        """A data -> data graph, as built by KNNEdges with source == target."""
        torch.manual_seed(0)
        graph = HeteroData()
        graph["data"].x = torch.rand((self.NUM_NODES, 2))
        # k edges per data node, dst-sorted, so every node is a target -- no orphans
        target = torch.arange(self.NUM_NODES).repeat_interleave(self.K)
        source = torch.randint(0, self.NUM_NODES, (self.NUM_NODES * self.K,))
        graph[("data", "to", "data")].edge_index = torch.stack([source, target], dim=0)
        graph[("data", "to", "data")].edge_length = torch.rand((self.NUM_NODES * self.K, 1))
        graph[("data", "to", "data")].edge_dirs = torch.rand((self.NUM_NODES * self.K, 2))
        return graph

    @pytest.fixture
    def graph_provider(self, fake_graph):
        return create_graph_provider(
            graph=fake_graph[("data", "to", "data")],
            edge_attributes=["edge_length", "edge_dirs"],
            src_size=self.NUM_NODES,
            dst_size=self.NUM_NODES,
            trainable_size=0,
        )

    def build(self, graph_provider, **overrides):
        kwargs = {
            "in_channels": self.IN_CHANNELS,
            "num_channels": self.NUM_CHANNELS,
            "num_layers": 2,
            "num_chunks": 2,
            "num_heads": 4,
            "mlp_hidden_ratio": 2,
            "edge_dim": graph_provider.edge_dim,
            "layer_kernels": load_layer_kernels(instance=False),
            "graph_attention_backend": "pyg",
        }
        kwargs.update(overrides)
        return GraphTransformerPreMixer(**kwargs)

    def run(self, premixer, graph_provider, x):
        edge_attr, edge_index, edge_shard_sizes = graph_provider.get_edges(batch_size=1)
        return premixer(
            x,
            batch_size=1,
            shard_info=GraphShardInfo(nodes=None, edges=edge_shard_sizes),
            edge_attr=edge_attr,
            edge_index=edge_index,
        )

    def test_preserves_shape(self, graph_provider):
        """Output width must equal input width so encoder/decoder shapes are untouched."""
        premixer = self.build(graph_provider)
        x = torch.rand((self.NUM_NODES, self.IN_CHANNELS))
        assert self.run(premixer, graph_provider, x).shape == x.shape

    def test_is_identity_at_init(self, graph_provider):
        """With initialise_out_zero the module must be an exact identity.

        This is what lets a checkpoint trained without a pre-mixer be forked
        and reproduce its parent bit-for-bit on step 0.
        """
        premixer = self.build(graph_provider, initialise_out_zero=True)
        x = torch.rand((self.NUM_NODES, self.IN_CHANNELS))
        torch.testing.assert_close(self.run(premixer, graph_provider, x), x, rtol=0, atol=0)

    def test_not_identity_when_out_is_trained(self, graph_provider):
        """Once the output projection moves off zero the module must actually mix."""
        premixer = self.build(graph_provider, initialise_out_zero=False)
        x = torch.rand((self.NUM_NODES, self.IN_CHANNELS))
        assert not torch.allclose(self.run(premixer, graph_provider, x), x)

    def test_mixes_across_neighbours(self, graph_provider, fake_graph):
        """The output at a node must depend on its neighbours' features.

        This is the whole point: the encoder can only pool first moments, so
        the pre-mixer has to make each token a function of its neighbourhood.
        """
        premixer = self.build(graph_provider, initialise_out_zero=False)
        x = torch.rand((self.NUM_NODES, self.IN_CHANNELS), requires_grad=True)

        out = self.run(premixer, graph_provider, x)

        # Pick a target node that has at least one neighbour other than itself
        edge_index = fake_graph[("data", "to", "data")].edge_index
        target = 0
        neighbours = edge_index[0][edge_index[1] == target]
        neighbours = neighbours[neighbours != target]
        assert len(neighbours) > 0, "test graph gave node 0 no distinct neighbour"

        out[target].sum().backward()
        assert x.grad[neighbours].abs().sum() > 0, "output does not depend on neighbour features"

    def test_gradients_flow(self, graph_provider):
        """Every pre-mixer parameter must receive a gradient."""
        premixer = self.build(graph_provider, initialise_out_zero=False)
        x = torch.rand((self.NUM_NODES, self.IN_CHANNELS))

        self.run(premixer, graph_provider, x).sum().backward()

        for name, param in premixer.named_parameters():
            assert param.grad is not None, f"param.grad is None for {name}"
