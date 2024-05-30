# (C) Copyright 2024 European Centre for Medium-Range Weather Forecasts.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import pytest
import torch
from torch import nn
from torch_geometric.data import HeteroData

from anemoi.models.layers.mapper import GraphTransformerBackwardMapper
from anemoi.models.layers.mapper import GraphTransformerBaseMapper
from anemoi.models.layers.mapper import GraphTransformerForwardMapper


class TestGraphTransformerBaseMapper:
    BIG_GRID_SIZE = 1000
    GRID_SIZE = 100

    @pytest.fixture
    def mapper_init(self):
        in_channels_src: int = 3
        in_channels_dst: int = 3
        hidden_dim: int = 256
        out_channels_dst: int = 5
        cpu_offload: bool = False
        activation: str = "SiLU"
        trainable_size: int = 6
        num_heads: int = 16
        mlp_hidden_ratio: int = 7
        return (
            in_channels_src,
            in_channels_dst,
            hidden_dim,
            out_channels_dst,
            cpu_offload,
            activation,
            trainable_size,
            num_heads,
            mlp_hidden_ratio,
        )

    @pytest.fixture
    def mapper(self, mapper_init, fake_graph):
        (
            in_channels_src,
            in_channels_dst,
            hidden_dim,
            out_channels_dst,
            cpu_offload,
            activation,
            trainable_size,
            num_heads,
            mlp_hidden_ratio,
        ) = mapper_init
        return GraphTransformerBaseMapper(
            in_channels_src=in_channels_src,
            in_channels_dst=in_channels_dst,
            hidden_dim=hidden_dim,
            out_channels_dst=out_channels_dst,
            cpu_offload=cpu_offload,
            activation=activation,
            sub_graph=fake_graph,
            trainable_size=trainable_size,
            num_heads=num_heads,
            mlp_hidden_ratio=mlp_hidden_ratio,
        )

    @pytest.fixture
    def pair_tensor(self, mapper_init):
        (
            in_channels_src,
            in_channels_dst,
            _hidden_dim,
            _out_channels_dst,
            _cpu_offload,
            _activation,
            _trainable_size,
            _num_heads,
            _mlp_hidden_ratio,
        ) = mapper_init
        return (
            torch.rand(self.BIG_GRID_SIZE, in_channels_src),
            torch.rand(self.GRID_SIZE, in_channels_dst),
        )

    @pytest.fixture
    def fake_graph(self):
        graph = HeteroData()
        graph.edge_attr = torch.rand((self.GRID_SIZE, 128))
        graph.edge_index = torch.randint(0, self.GRID_SIZE, (2, self.GRID_SIZE))
        return graph

    def test_initialization(self, mapper, mapper_init):
        (
            in_channels_src,
            in_channels_dst,
            hidden_dim,
            out_channels_dst,
            _cpu_offload,
            activation,
            _trainable_size,
            _num_heads,
            _mlp_hidden_ratio,
        ) = mapper_init
        assert isinstance(mapper, GraphTransformerBaseMapper)
        assert mapper.in_channels_src == in_channels_src
        assert mapper.in_channels_dst == in_channels_dst
        assert mapper.hidden_dim == hidden_dim
        assert mapper.out_channels_dst == out_channels_dst
        assert mapper.activation == activation

    def test_pre_process(self, mapper, mapper_init, pair_tensor):
        # Should be a no-op in the base class
        x = pair_tensor
        (
            _in_channels_src,
            _in_channels_dst,
            _hidden_dim,
            _out_channels_dst,
            _cpu_offload,
            _activation,
            _trainable_size,
            _num_heads,
            _mlp_hidden_ratio,
        ) = mapper_init
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


class TestGraphTransformerForwardMapper(TestGraphTransformerBaseMapper):
    @pytest.fixture
    def mapper(self, mapper_init, fake_graph):
        (
            in_channels_src,
            in_channels_dst,
            hidden_dim,
            out_channels_dst,
            cpu_offload,
            activation,
            trainable_size,
            num_heads,
            mlp_hidden_ratio,
        ) = mapper_init
        return GraphTransformerForwardMapper(
            in_channels_src=in_channels_src,
            in_channels_dst=in_channels_dst,
            hidden_dim=hidden_dim,
            out_channels_dst=out_channels_dst,
            cpu_offload=cpu_offload,
            activation=activation,
            sub_graph=fake_graph,
            trainable_size=trainable_size,
            num_heads=num_heads,
            mlp_hidden_ratio=mlp_hidden_ratio,
        )

    def test_pre_process(self, mapper, mapper_init, pair_tensor):
        x = pair_tensor
        (
            _in_channels_src,
            _in_channels_dst,
            hidden_dim,
            _out_channels_dst,
            _cpu_offload,
            _activation,
            _trainable_size,
            _num_heads,
            _mlp_hidden_ratio,
        ) = mapper_init
        shard_shapes = [list(x[0].shape)], [list(x[1].shape)]

        x_src, x_dst, shapes_src, shapes_dst = mapper.pre_process(x, shard_shapes)
        assert x_src.shape == torch.Size([self.BIG_GRID_SIZE, hidden_dim]), (
            f"x_src.shape ({x_src.shape}) != torch.Size"
            f"([self.BIG_GRID_SIZE, hidden_dim]) ({torch.Size([self.BIG_GRID_SIZE, hidden_dim])})"
        )
        assert x_dst.shape == torch.Size([self.GRID_SIZE, hidden_dim]), (
            f"x_dst.shape ({x_dst.shape}) != torch.Size"
            "([self.GRID_SIZE, hidden_dim]) ({torch.Size([self.GRID_SIZE, hidden_dim])})"
        )
        assert shapes_src == [[self.BIG_GRID_SIZE, hidden_dim]]
        assert shapes_dst == [[self.GRID_SIZE, hidden_dim]]

    def test_forward_backward(self, mapper_init, mapper, pair_tensor):
        (
            in_channels_src,
            _in_channels_dst,
            hidden_dim,
            _out_channels_dst,
            _cpu_offload,
            _activation,
            _trainable_size,
            _num_heads,
            _mlp_hidden_ratio,
        ) = mapper_init
        x = pair_tensor
        batch_size = 1
        shard_shapes = [list(x[0].shape)], [list(x[1].shape)]

        x_src, x_dst = mapper.forward(x, batch_size, shard_shapes)
        assert x_src.shape == torch.Size([self.BIG_GRID_SIZE, in_channels_src])
        assert x_dst.shape == torch.Size([self.GRID_SIZE, hidden_dim])

        # Dummy loss
        target = torch.rand(self.GRID_SIZE, hidden_dim)
        loss_fn = nn.MSELoss()

        loss = loss_fn(x_dst, target)

        # Check loss
        assert loss.item() >= 0

        loss.backward()

        # Check gradients
        assert mapper.trainable.trainable.grad is not None
        assert mapper.trainable.trainable.grad.shape == mapper.trainable.trainable.shape

        for param in mapper.parameters():
            assert param.grad is not None, f"param.grad is None for {param}"
            assert (
                param.grad.shape == param.shape
            ), f"param.grad.shape ({param.grad.shape}) != param.shape ({param.shape}) for {param}"


class TestGraphTransformerBackwardMapper(TestGraphTransformerBaseMapper):
    @pytest.fixture
    def mapper(self, mapper_init, fake_graph):
        (
            in_channels_src,
            in_channels_dst,
            hidden_dim,
            out_channels_dst,
            cpu_offload,
            activation,
            trainable_size,
            _num_heads,
            _mlp_hidden_ratio,
        ) = mapper_init
        return GraphTransformerBackwardMapper(
            in_channels_src=in_channels_src,
            in_channels_dst=in_channels_dst,
            hidden_dim=hidden_dim,
            out_channels_dst=out_channels_dst,
            cpu_offload=cpu_offload,
            activation=activation,
            sub_graph=fake_graph,
            trainable_size=trainable_size,
        )

    def test_pre_process(self, mapper, mapper_init, pair_tensor):
        x = pair_tensor
        (
            in_channels_src,
            _in_channels_dst,
            hidden_dim,
            _out_channels_dst,
            _cpu_offload,
            _activation,
            _trainable_size,
            _num_heads,
            _mlp_hidden_ratio,
        ) = mapper_init
        shard_shapes = [list(x[0].shape)], [list(x[1].shape)]

        x_src, x_dst, shapes_src, shapes_dst = mapper.pre_process(x, shard_shapes)
        assert x_src.shape == torch.Size([self.BIG_GRID_SIZE, in_channels_src]), (
            f"x_src.shape ({x_src.shape}) != torch.Size"
            f"([self.BIG_GRID_SIZE, in_channels_src]) ({torch.Size([self.BIG_GRID_SIZE, in_channels_src])})"
        )
        assert x_dst.shape == torch.Size([self.GRID_SIZE, hidden_dim]), (
            f"x_dst.shape ({x_dst.shape}) != torch.Size"
            f"([self.GRID_SIZE, hidden_dim]) ({torch.Size([self.GRID_SIZE, hidden_dim])})"
        )
        assert shapes_src == [[self.BIG_GRID_SIZE, hidden_dim]]
        assert shapes_dst == [[self.GRID_SIZE, hidden_dim]]

    def test_post_process(self, mapper, mapper_init):
        (
            _in_channels_src,
            _in_channels_dst,
            hidden_dim,
            out_channels_dst,
            _cpu_offload,
            _activation,
            _trainable_size,
            _num_heads,
            _mlp_hidden_ratio,
        ) = mapper_init
        x_dst = torch.rand(self.GRID_SIZE, hidden_dim)
        shapes_dst = [list(x_dst.shape)]

        result = mapper.post_process(x_dst, shapes_dst)
        assert (
            torch.Size([self.GRID_SIZE, out_channels_dst]) == result.shape
        ), f"[self.GRID_SIZE, out_channels_dst] ({[self.GRID_SIZE, out_channels_dst]}) != result.shape ({result.shape})"

    def test_forward_backward(self, mapper_init, mapper, pair_tensor):
        (
            in_channels_src,
            _in_channels_dst,
            hidden_dim,
            out_channels_dst,
            _cpu_offload,
            _activation,
            _trainable_size,
            _num_heads,
            _mlp_hidden_ratio,
        ) = mapper_init
        pair_tensor
        shard_shapes = [list(pair_tensor[0].shape)], [list(pair_tensor[1].shape)]
        batch_size = 1

        # Different size for x_dst, as the Backward mapper changes the channels in shape in pre-processor
        x = (
            torch.rand(self.BIG_GRID_SIZE, hidden_dim),
            torch.rand(self.GRID_SIZE, in_channels_src),
        )

        result = mapper.forward(x, batch_size, shard_shapes)
        assert result.shape == torch.Size([self.GRID_SIZE, out_channels_dst])

        # Dummy loss
        target = torch.rand(self.GRID_SIZE, out_channels_dst)
        loss_fn = nn.MSELoss()

        loss = loss_fn(result, target)

        # Check loss
        assert loss.item() >= 0

        loss.backward()

        # Check gradients
        assert mapper.trainable.trainable.grad is not None
        assert mapper.trainable.trainable.grad.shape == mapper.trainable.trainable.shape

        for param in mapper.parameters():
            assert param.grad is not None, f"param.grad is None for {param}"
            assert (
                param.grad.shape == param.shape
            ), f"param.grad.shape ({param.grad.shape}) != param.shape ({param.shape}) for {param}"
