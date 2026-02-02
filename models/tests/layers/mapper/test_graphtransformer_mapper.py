# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from dataclasses import asdict
from dataclasses import dataclass
from dataclasses import field

import pytest
import torch
from torch import nn
from torch_geometric.data import HeteroData

from anemoi.models.layers.graph_provider import create_graph_provider
from anemoi.models.layers.mapper import GraphTransformerBackwardMapper
from anemoi.models.layers.mapper import GraphTransformerBaseMapper
from anemoi.models.layers.mapper import GraphTransformerForwardMapper
from anemoi.models.layers.utils import load_layer_kernels
from anemoi.utils.config import DotDict


@pytest.fixture(scope="module")
def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ConcreteGraphTransformerBaseMapper(GraphTransformerBaseMapper):
    """Concrete implementation of GraphTransformerBaseMapper for testing."""

    def pre_process(self, x, shard_shapes, model_comm_group=None, x_src_is_sharded=False, x_dst_is_sharded=False):
        shapes_src, shapes_dst = shard_shapes
        x_src, x_dst = x
        return x_src, x_dst, shapes_src, shapes_dst

    def post_process(self, x_dst, **kwargs):
        return x_dst


@dataclass
class MapperConfig:
    in_channels_src: int = 3
    in_channels_dst: int = 3
    hidden_dim: int = 256
    num_chunks: int = 2
    num_heads: int = 16
    mlp_hidden_ratio: int = 7
    qk_norm: bool = True
    cpu_offload: bool = False
    layer_kernels: field(default_factory=DotDict) = None
    shard_strategy: str = "edges"
    graph_attention_backend: str = "pyg"
    edge_dim: int = None  # Will be set from graph_provider
    edge_pre_mlp: bool = False

    def __post_init__(self):
        self.layer_kernels = load_layer_kernels(instance=False)


class TestGraphTransformerBaseMapper:
    """Test the GraphTransformerBaseMapper class."""

    NUM_EDGES: int = 150
    NUM_SRC_NODES: int = 100
    NUM_DST_NODES: int = 200
    OUT_CHANNELS_DST: int = 5

    @pytest.fixture
    def mapper_init(self):
        return MapperConfig()

    @pytest.fixture
    def graph_provider(self, fake_graph, device):
        provider = create_graph_provider(
            graph=fake_graph[("nodes", "to", "nodes")],
            edge_attributes=["edge_attr1", "edge_attr2"],
            src_size=self.NUM_SRC_NODES,
            dst_size=self.NUM_DST_NODES,
            trainable_size=6,
        )
        return provider.to(device)

    @pytest.fixture
    def mapper(self, mapper_init, graph_provider, device):
        config = asdict(mapper_init)
        config["edge_dim"] = graph_provider.edge_dim
        return ConcreteGraphTransformerBaseMapper(
            **config,
            out_channels_dst=self.OUT_CHANNELS_DST,
        ).to(device)

    @pytest.fixture
    def pair_tensor(self, mapper_init, device):
        return (
            torch.rand(self.NUM_SRC_NODES, mapper_init.in_channels_src, device=device),
            torch.rand(self.NUM_DST_NODES, mapper_init.in_channels_dst, device=device),
        )

    @pytest.fixture
    def fake_graph(self, device) -> HeteroData:
        """Fake graph."""
        graph = HeteroData()
        graph[("nodes", "to", "nodes")].edge_index = torch.concat(
            [
                torch.randint(0, self.NUM_SRC_NODES, (1, self.NUM_EDGES), device=device),
                torch.randint(0, self.NUM_DST_NODES, (1, self.NUM_EDGES), device=device),
            ],
            axis=0,
        )
        graph[("nodes", "to", "nodes")].edge_attr1 = torch.rand((self.NUM_EDGES, 1), device=device)
        graph[("nodes", "to", "nodes")].edge_attr2 = torch.rand((self.NUM_EDGES, 32), device=device)
        return graph

    def test_initialization(self, mapper, mapper_init):
        assert isinstance(mapper, GraphTransformerBaseMapper)
        assert mapper.in_channels_src == mapper_init.in_channels_src
        assert mapper.in_channels_dst == mapper_init.in_channels_dst
        assert mapper.hidden_dim == mapper_init.hidden_dim
        assert mapper.out_channels_dst == self.OUT_CHANNELS_DST
        assert isinstance(mapper.activation, nn.Module)

    def test_pre_process(self, mapper, pair_tensor):
        # Should be a no-op in the base class
        x = pair_tensor
        shard_shapes = [list(x[0].shape)], [list(x[1].shape)]

        x_src, x_dst, shapes_src, shapes_dst = mapper.pre_process(x, shard_shapes)
        assert x_src.shape == torch.Size(
            x[0].shape
        ), f"x_src.shape ({x_src.shape}) != torch.Size(x[0].shape) ({torch.Size(x[0].shape)})"
        assert x_dst.shape == torch.Size(
            x[1].shape
        ), f"x_dst.shape ({x_dst.shape}) != torch.Size(x[1].shape) ({x[1].shape})"
        assert shapes_src == [
            list(x[0].shape)
        ], f"shapes_src ({shapes_src}) != [list(x[0].shape)] ({[list(x[0].shape)]})"
        assert shapes_dst == [
            list(x[1].shape)
        ], f"shapes_dst ({shapes_dst}) != [list(x[1].shape)] ({[list(x[1].shape)]})"

    def test_post_process(self, mapper, pair_tensor):
        # Should be a no-op in the base class
        x_dst = pair_tensor[1]
        shapes_dst = [list(x_dst.shape)]

        result = mapper.post_process(x_dst, shapes_dst=shapes_dst)
        assert torch.equal(result, x_dst)


class TestGraphTransformerForwardMapper(TestGraphTransformerBaseMapper):
    """Test the GraphTransformerForwardMapper class."""

    OUT_CHANNELS_DST = None

    @pytest.fixture
    def mapper(self, mapper_init, graph_provider, device):
        config = asdict(mapper_init)
        config["edge_dim"] = graph_provider.edge_dim
        return GraphTransformerForwardMapper(**config).to(device)

    def test_pre_process(self, mapper, mapper_init, pair_tensor):
        x = pair_tensor
        shard_shapes = [list(x[0].shape)], [list(x[1].shape)]

        x_src, x_dst, shapes_src, shapes_dst = mapper.pre_process(x, shard_shapes)
        assert x_src.shape == torch.Size([self.NUM_SRC_NODES, mapper_init.hidden_dim]), (
            f"x_src.shape ({x_src.shape}) != torch.Size"
            f"([self.NUM_SRC_NODES, hidden_dim]) ({torch.Size([self.NUM_SRC_NODES, mapper_init.hidden_dim])})"
        )
        assert x_dst.shape == torch.Size([self.NUM_DST_NODES, mapper_init.hidden_dim]), (
            f"x_dst.shape ({x_dst.shape}) != torch.Size"
            "([self.NUM_DST_NODES, hidden_dim]) ({torch.Size([self.NUM_DST_NODES, hidden_dim])})"
        )
        assert shapes_src == [[self.NUM_SRC_NODES, mapper_init.hidden_dim]]
        assert shapes_dst == [[self.NUM_DST_NODES, mapper_init.hidden_dim]]

    def test_forward_backward(self, mapper_init, mapper, pair_tensor, graph_provider):
        x = pair_tensor
        batch_size = 1
        shard_shapes = [list(x[0].shape)], [list(x[1].shape)]

        edge_attr, edge_index, _ = graph_provider.get_edges(batch_size=batch_size)
        x_src, x_dst = mapper.forward(x, batch_size, shard_shapes, edge_attr, edge_index)
        assert x_src.shape == torch.Size([self.NUM_SRC_NODES, mapper_init.in_channels_src])
        assert x_dst.shape == torch.Size([self.NUM_DST_NODES, mapper_init.hidden_dim])

        # Dummy loss
        target = torch.rand(self.NUM_DST_NODES, mapper_init.hidden_dim, device=x_dst.device)
        loss_fn = nn.MSELoss()

        loss = loss_fn(x_dst, target)

        # Check loss
        assert loss.item() >= 0

        loss.backward()

        # Check gradients
        assert graph_provider.trainable.trainable.grad is not None
        assert graph_provider.trainable.trainable.grad.shape == graph_provider.trainable.trainable.shape

        for param in mapper.parameters():
            assert param.grad is not None, f"param.grad is None for {param}"
            assert (
                param.grad.shape == param.shape
            ), f"param.grad.shape ({param.grad.shape}) != param.shape ({param.shape}) for {param}"

    def test_chunking(self, mapper, pair_tensor, graph_provider):
        x = pair_tensor
        batch_size = 1
        shard_shapes = [list(x[0].shape)], [list(x[1].shape)]

        edge_attr, edge_index, _ = graph_provider.get_edges(batch_size=batch_size)

        mapper.num_chunks = 4
        x_src_c, x_dst_c = mapper.forward(x, batch_size, shard_shapes, edge_attr, edge_index)

        mapper.num_chunks = 1
        x_src, x_dst = mapper.forward(x, batch_size, shard_shapes, edge_attr, edge_index)

        assert torch.allclose(
            x_src, x_src_c, atol=1e-4
        ), f"x_src ({x_src}) != x_src_c ({x_src_c}) when num_chunks is changed"
        assert torch.allclose(
            x_dst, x_dst_c, atol=1e-4
        ), f"x_dst ({x_dst}) != x_dst_c ({x_dst_c}) when num_chunks is changed"

    def test_strategy(self, mapper, pair_tensor, graph_provider):
        x = pair_tensor
        batch_size = 1
        shard_shapes = [list(x[0].shape)], [list(x[1].shape)]

        edge_attr, edge_index, _ = graph_provider.get_edges(batch_size=batch_size)

        out_heads = mapper.mapper_forward_with_heads_sharding(x, batch_size, shard_shapes, edge_attr, edge_index)

        out_edges = mapper.mapper_forward_with_edge_sharding(x, batch_size, shard_shapes, edge_attr, edge_index)

        assert torch.allclose(
            out_heads, out_edges, atol=1e-4
        ), f"out_heads ({out_heads}) != out_edges ({out_edges}) when using different strategies"


class TestGraphTransformerBackwardMapper(TestGraphTransformerBaseMapper):
    """Test the GraphTransformerBackwardMapper class."""

    @pytest.fixture
    def mapper(self, mapper_init, graph_provider, device):
        config = asdict(mapper_init)
        config["edge_dim"] = graph_provider.edge_dim
        return GraphTransformerBackwardMapper(
            **config,
            out_channels_dst=self.OUT_CHANNELS_DST,
        ).to(device)

    def test_pre_process(self, mapper, mapper_init, pair_tensor):
        x = pair_tensor
        shard_shapes = [list(x[0].shape)], [list(x[1].shape)]

        x_src, x_dst, shapes_src, shapes_dst = mapper.pre_process(x, shard_shapes)
        assert x_src.shape == torch.Size([self.NUM_SRC_NODES, mapper_init.in_channels_src]), (
            f"x_src.shape ({x_src.shape}) != torch.Size"
            f"([self.NUM_SRC_NODES, in_channels_src]) ({torch.Size([self.NUM_SRC_NODES, mapper_init.in_channels_src])})"
        )
        assert x_dst.shape == torch.Size([self.NUM_DST_NODES, mapper_init.hidden_dim]), (
            f"x_dst.shape ({x_dst.shape}) != torch.Size"
            f"([self.NUM_DST_NODES, hidden_dim]) ({torch.Size([self.NUM_DST_NODES, mapper_init.hidden_dim])})"
        )
        assert shapes_src == [[self.NUM_SRC_NODES, mapper_init.hidden_dim]]
        assert shapes_dst == [[self.NUM_DST_NODES, mapper_init.hidden_dim]]

    def test_post_process(self, mapper, mapper_init):
        x_dst = torch.rand(
            self.NUM_DST_NODES,
            mapper_init.hidden_dim,
            device=next(mapper.parameters()).device,
        )
        shapes_dst = [list(x_dst.shape)]

        result = mapper.post_process(x_dst, shapes_dst=shapes_dst)
        assert (
            torch.Size([self.NUM_DST_NODES, self.OUT_CHANNELS_DST]) == result.shape
        ), f"[self.NUM_DST_NODES, out_channels_dst] ({[self.NUM_DST_NODES, self.OUT_CHANNELS_DST]}) != result.shape ({result.shape})"

    def test_forward_backward(self, mapper_init, mapper, pair_tensor, graph_provider):
        shard_shapes = [list(pair_tensor[0].shape)], [list(pair_tensor[1].shape)]
        batch_size = 1

        # Different size for x_dst, as the Backward mapper changes the channels in shape in pre-processor
        device = next(mapper.parameters()).device
        x = (
            torch.rand(self.NUM_SRC_NODES, mapper_init.hidden_dim, device=device),
            torch.rand(self.NUM_DST_NODES, mapper_init.in_channels_src, device=device),
        )

        edge_attr, edge_index, _ = graph_provider.get_edges(batch_size=batch_size)
        result = mapper.forward(x, batch_size, shard_shapes, edge_attr, edge_index)
        assert result.shape == torch.Size([self.NUM_DST_NODES, self.OUT_CHANNELS_DST])

        # Dummy loss
        target = torch.rand(self.NUM_DST_NODES, self.OUT_CHANNELS_DST, device=result.device)
        loss_fn = nn.MSELoss()

        loss = loss_fn(result, target)

        # Check loss
        assert loss.item() >= 0

        loss.backward()

        # Check gradients
        assert graph_provider.trainable.trainable.grad is not None
        assert graph_provider.trainable.trainable.grad.shape == graph_provider.trainable.trainable.shape

        for param in mapper.parameters():
            assert param.grad is not None, f"param.grad is None for {param}"
            assert (
                param.grad.shape == param.shape
            ), f"param.grad.shape ({param.grad.shape}) != param.shape ({param.shape}) for {param}"

    def test_chunking(self, mapper_init, mapper, pair_tensor, graph_provider):
        shard_shapes = [list(pair_tensor[0].shape)], [list(pair_tensor[1].shape)]
        batch_size = 1

        device = next(mapper.parameters()).device
        x = (
            torch.rand(self.NUM_SRC_NODES, mapper_init.hidden_dim, device=device),
            torch.rand(self.NUM_DST_NODES, mapper_init.in_channels_src, device=device),
        )

        edge_attr, edge_index, _ = graph_provider.get_edges(batch_size=batch_size)

        mapper.num_chunks = 4
        out_c = mapper.forward(x, batch_size, shard_shapes, edge_attr, edge_index)

        mapper.num_chunks = 1
        out = mapper.forward(x, batch_size, shard_shapes, edge_attr, edge_index)

        assert torch.allclose(out, out_c, atol=1e-4), f"out ({out}) != out_c ({out_c}) when num_chunks is changed"

    def test_strategy(self, mapper_init, mapper, pair_tensor, graph_provider):
        shard_shapes = [list(pair_tensor[0].shape)], [list(pair_tensor[1].shape)]
        batch_size = 1

        device = next(mapper.parameters()).device
        x = (
            torch.rand(self.NUM_SRC_NODES, mapper_init.hidden_dim, device=device),
            torch.rand(self.NUM_DST_NODES, mapper_init.in_channels_src, device=device),
        )

        edge_attr, edge_index, _ = graph_provider.get_edges(batch_size=batch_size)

        out_heads = mapper.mapper_forward_with_heads_sharding(x, batch_size, shard_shapes, edge_attr, edge_index)

        out_edges = mapper.mapper_forward_with_edge_sharding(x, batch_size, shard_shapes, edge_attr, edge_index)

        assert torch.allclose(
            out_heads, out_edges, atol=1e-4
        ), f"out_heads ({out_heads}) != out_edges ({out_edges}) when using different strategies"
