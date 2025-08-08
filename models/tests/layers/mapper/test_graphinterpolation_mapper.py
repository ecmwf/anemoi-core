# (C) Copyright 2025 Anemoi contributors.
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

from anemoi.models.layers.mapper import GraphInterpolationBackwardMapper
from anemoi.models.layers.mapper import GraphInterpolationBaseMapper
from anemoi.models.layers.mapper import GraphInterpolationForwardMapper
from anemoi.models.layers.utils import load_layer_kernels
from anemoi.utils.config import DotDict


@dataclass
class MapperConfig:
    in_channels_src: int = 3
    out_channels_dst: int = 5
    # trainable_size: int = 6
    src_grid_size: int = 0
    dst_grid_size: int = 0
    cpu_offload: bool = False
    layer_kernels: field(default_factory=DotDict) = None

    def __post_init__(self):
        self.layer_kernels = load_layer_kernels(instance=False)


class TestGraphInterpolationBaseMapper:
    """Test the GraphInterpolationBaseMapper class."""

    NUM_EDGES_PER_DST_NODE: int = 3
    NUM_SRC_NODES: int = 100
    NUM_DST_NODES: int = 200

    @pytest.fixture
    def mapper_init(self):
        return MapperConfig(src_grid_size=self.NUM_SRC_NODES, dst_grid_size=self.NUM_DST_NODES)

    @pytest.fixture
    def mapper(self, mapper_init, fake_graph):
        return GraphInterpolationBaseMapper(
            **asdict(mapper_init),
            hidden_dim=4,
            sub_graph=fake_graph[("src_nodes", "to", "dst_nodes")],
            sub_graph_edge_attributes=["edge_attr1", "edge_attr2"],
        )

    @pytest.fixture
    def pair_tensor(self, mapper_init):
        return (
            torch.rand(self.NUM_SRC_NODES, mapper_init.in_channels_src),
            torch.rand(self.NUM_DST_NODES, 1),  # features of dst_nodes are not used
        )

    @pytest.fixture
    def fake_graph(self) -> HeteroData:
        """Fake graph."""
        graph = HeteroData()
        num_edges = self.NUM_EDGES_PER_DST_NODE * self.NUM_DST_NODES
        graph[("src_nodes", "to", "dst_nodes")].edge_index = torch.concat(
            [
                torch.randint(0, self.NUM_SRC_NODES, (1, num_edges)),
                torch.arange(self.NUM_DST_NODES).repeat_interleave(self.NUM_EDGES_PER_DST_NODE)[None, :],
            ],
            axis=0,
        )
        graph[("src_nodes", "to", "dst_nodes")].edge_attr1 = torch.rand((num_edges, 1))
        graph[("src_nodes", "to", "dst_nodes")].edge_attr2 = torch.rand((num_edges, 32))
        graph[("src_nodes", "to", "dst_nodes")].dist = torch.rand((num_edges, 1))
        return graph

    def test_initialization(self, mapper, mapper_init):
        assert isinstance(mapper, GraphInterpolationBaseMapper)
        assert mapper.in_channels_src == mapper_init.in_channels_src
        assert mapper.out_channels_dst == mapper_init.out_channels_dst
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

        result = mapper.post_process(x_dst, shapes_dst)
        assert torch.equal(result, x_dst)


class TestGraphInterpolationForwardMapper(TestGraphInterpolationBaseMapper):
    """Test the GraphInterpolationForwardMapper class."""

    @pytest.fixture
    def mapper(self, mapper_init, fake_graph):
        return GraphInterpolationForwardMapper(
            **asdict(mapper_init),
            sub_graph=fake_graph[("src_nodes", "to", "dst_nodes")],
            sub_graph_edge_attributes=["edge_attr1", "edge_attr2"],
        )

    def test_pre_process(self, mapper, mapper_init, pair_tensor):
        x = pair_tensor
        shard_shapes = [list(x[0].shape)], [list(x[1].shape)]

        x_src, _, shapes_src, _ = mapper.pre_process(x, shard_shapes)
        assert x_src.shape == torch.Size([self.NUM_SRC_NODES, mapper_init.out_channels_dst]), (
            f"x_src.shape ({x_src.shape}) != torch.Size"
            f"([self.NUM_SRC_NODES, out_channels_dst]) ({torch.Size([self.NUM_SRC_NODES, mapper_init.out_channels_dst])})"
        )
        assert shapes_src == [[self.NUM_SRC_NODES, mapper_init.out_channels_dst]]
        # in_channels_dst is not used

    def test_forward_backward(self, mapper_init, mapper, pair_tensor):
        x = pair_tensor
        batch_size = 1
        shard_shapes = [list(x[0].shape)], [list(x[1].shape)]

        x_src, x_dst = mapper.forward(x, batch_size, shard_shapes)
        assert x_src.shape == torch.Size([self.NUM_SRC_NODES, mapper_init.in_channels_src])

        # Dummy loss
        target = torch.rand(x_dst.shape)
        loss_fn = nn.MSELoss()

        loss = loss_fn(x_dst, target)

        # Check loss
        assert loss.item() >= 0

        loss.backward()

        # Check gradients
        assert mapper.trainable.trainable is None

        for param in mapper.parameters():
            assert param.grad is not None, f"param.grad is None for {param}"
            assert (
                param.grad.shape == param.shape
            ), f"param.grad.shape ({param.grad.shape}) != param.shape ({param.shape}) for {param}"


class TestGraphInterpolationBackwardMapper(TestGraphInterpolationBaseMapper):
    """Test the GraphInterpolationBackwardMapper class."""

    @pytest.fixture
    def mapper(self, mapper_init, fake_graph):
        return GraphInterpolationBackwardMapper(
            **asdict(mapper_init),
            sub_graph=fake_graph[("src_nodes", "to", "dst_nodes")],
            sub_graph_edge_attributes=["edge_attr1", "edge_attr2"],
        )

    def test_pre_process(self, mapper, mapper_init, pair_tensor):
        x = pair_tensor
        shard_shapes = [list(x[0].shape)], [list(x[1].shape)]

        x_src, _, shapes_src, _ = mapper.pre_process(x, shard_shapes)
        assert x_src.shape == torch.Size([self.NUM_SRC_NODES, mapper_init.in_channels_src]), (
            f"x_src.shape ({x_src.shape}) != torch.Size"
            f"([self.NUM_SRC_NODES, in_channels_src]) ({torch.Size([self.NUM_SRC_NODES, mapper_init.in_channels_src])})"
        )
        assert shapes_src == [[self.NUM_SRC_NODES, mapper_init.in_channels_src]]
        # in_channels_dst is not used

    def test_post_process(self, mapper, mapper_init):
        x_dst = torch.rand(self.NUM_DST_NODES, mapper.hidden_dim)
        shapes_dst = [list(x_dst.shape)]

        result = mapper.post_process(x_dst, shapes_dst)
        assert (
            torch.Size([self.NUM_DST_NODES, mapper_init.out_channels_dst]) == result.shape
        ), f"[self.NUM_DST_NODES, out_channels_dst] ({[self.NUM_DST_NODES, mapper_init.out_channels_dst]}) != result.shape ({result.shape})"

    def test_forward_backward(self, mapper_init, mapper, pair_tensor):
        shard_shapes = [list(pair_tensor[0].shape)], [list(pair_tensor[1].shape)]
        batch_size = 1

        # Different size for x_dst, as the Backward mapper changes the channels in shape in pre-processor
        x = (
            torch.rand(self.NUM_SRC_NODES, mapper.hidden_dim),
            torch.rand(self.NUM_DST_NODES, mapper_init.in_channels_src),
        )

        result = mapper.forward(x, batch_size, shard_shapes)
        assert result.shape == torch.Size([self.NUM_DST_NODES, mapper_init.out_channels_dst])

        # Dummy loss
        target = torch.rand(self.NUM_DST_NODES, mapper_init.out_channels_dst)
        loss_fn = nn.MSELoss()

        loss = loss_fn(result, target)

        # Check loss
        assert loss.item() >= 0

        loss.backward()

        # Check gradients
        assert mapper.trainable.trainable is None

        for param in mapper.parameters():
            assert param.grad is not None, f"param.grad is None for {param}"
            assert (
                param.grad.shape == param.shape
            ), f"param.grad.shape ({param.grad.shape}) != param.shape ({param.shape}) for {param}"
