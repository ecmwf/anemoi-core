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

import pytest
import torch
from torch import nn
from torch_geometric.data import HeteroData

from anemoi.models.layers.mapper import GraphInterpolationBackwardMapper
from anemoi.models.layers.mapper import GraphInterpolationBaseMapper
from anemoi.models.layers.mapper import GraphInterpolationForwardMapper
from anemoi.models.layers.mapper import SparseProjector


@dataclass
class MapperConfig:
    in_channels_src: int = 3
    in_channels_dst: int = 3
    hidden_dim: int = 256
    edge_attribute_name: str = "edge_attr"


class TestGraphInterpolation:
    """Test the GraphInterpolation class."""

    NUM_EDGES: int = 150
    NUM_SRC_NODES: int = 100
    NUM_DST_NODES: int = 200
    OUT_CHANNELS_DST: int = 5

    @pytest.fixture
    def mapper_init(self):
        return MapperConfig()

    @pytest.fixture
    def pair_tensor(self, mapper_init):
        return (
            torch.rand(self.NUM_SRC_NODES, mapper_init.in_channels_src),
            torch.rand(self.NUM_DST_NODES, mapper_init.in_channels_dst),
        )

    @pytest.fixture
    def fake_graph(self) -> HeteroData:
        """Fake graph."""
        graph = HeteroData()
        graph[("nodes", "to", "nodes")].edge_index = torch.concat(
            [
                torch.randint(0, self.NUM_SRC_NODES, (1, self.NUM_EDGES)),
                torch.randint(0, self.NUM_DST_NODES, (1, self.NUM_EDGES)),
            ],
            axis=0,
        )
        graph[("nodes", "to", "nodes")].edge_attr1 = torch.rand((self.NUM_EDGES, 1))
        graph[("nodes", "to", "nodes")].edge_attr2 = torch.rand((self.NUM_EDGES, 32))
        return graph


class TestGraphInterpolationBaseMapper(TestGraphInterpolation):
    """Test the GraphInterpolationBaseMapper class."""

    @pytest.fixture
    def mapper(self, mapper_init, fake_graph):
        return GraphInterpolationBaseMapper(
            **asdict(mapper_init),
            out_channels_dst=self.OUT_CHANNELS_DST,
            sub_graph=fake_graph[("nodes", "to", "nodes")],
            src_grid_size=self.NUM_SRC_NODES,
            dst_grid_size=self.NUM_DST_NODES,
        )

    def test_initialization(self, mapper):
        assert isinstance(mapper, GraphInterpolationBaseMapper)
        assert isinstance(mapper.project, SparseProjector)
        assert len(list(mapper.parameters())) == 0, "GraphInterpolationBaseMapper should not have parameters"


class TestGraphInterpolationForwardMapper(TestGraphInterpolation):
    """Test the GraphInterpolationForwardMapper class."""

    @pytest.fixture
    def mapper(self, mapper_init, fake_graph):
        return GraphInterpolationForwardMapper(
            **asdict(mapper_init),
            sub_graph=fake_graph[("nodes", "to", "nodes")],
            src_grid_size=self.NUM_SRC_NODES,
            dst_grid_size=self.NUM_DST_NODES,
        )

    def test_initialization(self, mapper):
        assert isinstance(mapper, GraphInterpolationBaseMapper)
        assert isinstance(mapper.project, SparseProjector)
        assert len(list(mapper.parameters())) in [0, 2]  # no parameters or Linear.{weight,bias}

    def test_forward_backward(self, mapper_init, mapper, pair_tensor):
        x = pair_tensor
        batch_size = 1
        shard_shapes = [list(x[0].shape)], [list(x[1].shape)]

        x_src, x_dst = mapper.forward(x, batch_size, shard_shapes)
        assert x_src.shape == torch.Size([self.NUM_SRC_NODES, mapper_init.in_channels_src])
        assert x_dst.shape == torch.Size([self.NUM_DST_NODES, mapper_init.hidden_dim])

        # Dummy loss
        target = torch.rand(self.NUM_DST_NODES, mapper_init.hidden_dim)
        loss_fn = nn.MSELoss()

        loss = loss_fn(x_dst, target)

        # Check loss
        assert loss.item() >= 0

        loss.backward()

        # Check gradients
        for param in mapper.parameters():
            assert param.grad is not None, f"param.grad is None for {param}"
            assert (
                param.grad.shape == param.shape
            ), f"param.grad.shape ({param.grad.shape}) != param.shape ({param.shape}) for {param}"


class TestGraphInterpolationBackwardMapper(TestGraphInterpolation):
    """Test the GraphInterpolationBackwardMapper class."""

    def test_initialization(self, mapper):
        assert isinstance(mapper, GraphInterpolationBaseMapper)
        assert isinstance(mapper.project, SparseProjector)
        assert len(list(mapper.parameters())) in [0, 2]  # no parameters or Linear.{weight,bias}

    @pytest.fixture
    def mapper(self, mapper_init, fake_graph):
        return GraphInterpolationBackwardMapper(
            **asdict(mapper_init),
            out_channels_dst=self.OUT_CHANNELS_DST,
            sub_graph=fake_graph[("nodes", "to", "nodes")],
            src_grid_size=self.NUM_SRC_NODES,
            dst_grid_size=self.NUM_DST_NODES,
        )

    def test_forward_backward(self, mapper_init, mapper, pair_tensor):
        shard_shapes = [list(pair_tensor[0].shape)], [list(pair_tensor[1].shape)]
        batch_size = 1

        # Different size for x_dst, as the Backward mapper changes the channels in shape in pre-processor
        x = (
            torch.rand(self.NUM_SRC_NODES, mapper_init.hidden_dim),
            torch.rand(self.NUM_DST_NODES, mapper_init.in_channels_src),
        )

        result = mapper.forward(x, batch_size, shard_shapes)
        assert result.shape == torch.Size([self.NUM_DST_NODES, self.OUT_CHANNELS_DST])

        # Dummy loss
        target = torch.rand(self.NUM_DST_NODES, self.OUT_CHANNELS_DST)
        loss_fn = nn.MSELoss()

        loss = loss_fn(result, target)

        # Check loss
        assert loss.item() >= 0

        loss.backward()

        # Check gradients
        for param in mapper.parameters():
            assert param.grad is not None, f"param.grad is None for {param}"
            assert (
                param.grad.shape == param.shape
            ), f"param.grad.shape ({param.grad.shape}) != param.shape ({param.shape}) for {param}"
